#include "tensor.h"
#include "IR.h"
#include "backend.h"
#include "CPUBackend.h"
#include "CUDABackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Used outside timed regions to consume the returned result.
volatile float benchmarkSink = 0.0f;

struct Statistics {
    double minimum;
    double median;
    double mean;
    double maximum;
};

tensor makeMatrix(int rows, int columns, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

    std::vector<float> data(
        static_cast<std::size_t>(rows) * columns
    );

    for (float& value : data) {
        value = distribution(generator);
    }

    return tensor(data, {rows, columns});
}

IR makeNetwork(
    const tensor& input,
    const tensor& weight,
    const tensor& bias,
    bool fused
) {
    IR ir;

    IRInstruction inputInstruction{};
    inputInstruction.operation = "Input";
    inputInstruction.output = 0;
    inputInstruction.value = input;
    ir.addInstruction(inputInstruction);

    IRInstruction weightInstruction{};
    weightInstruction.operation = "Constant";
    weightInstruction.output = 1;
    weightInstruction.value = weight;
    ir.addInstruction(weightInstruction);

    IRInstruction biasInstruction{};
    biasInstruction.operation = "Constant";
    biasInstruction.output = 2;
    biasInstruction.value = bias;
    ir.addInstruction(biasInstruction);

    if (fused) {
        IRInstruction fusedInstruction{};
        fusedInstruction.operation = "FusedMatMulAdd";
        fusedInstruction.inputs = {0, 1, 2};
        fusedInstruction.output = 3;
        ir.addInstruction(fusedInstruction);

        IRInstruction reluInstruction{};
        reluInstruction.operation = "ReLU";
        reluInstruction.inputs = {3};
        reluInstruction.output = 4;
        ir.addInstruction(reluInstruction);

        ir.setOutputID(4);
    }
    else {
        IRInstruction matmulInstruction{};
        matmulInstruction.operation = "MatMul";
        matmulInstruction.inputs = {0, 1};
        matmulInstruction.output = 3;
        ir.addInstruction(matmulInstruction);

        IRInstruction addInstruction{};
        addInstruction.operation = "Add";
        addInstruction.inputs = {3, 2};
        addInstruction.output = 4;
        ir.addInstruction(addInstruction);

        IRInstruction reluInstruction{};
        reluInstruction.operation = "ReLU";
        reluInstruction.inputs = {4};
        reluInstruction.output = 5;
        ir.addInstruction(reluInstruction);

        ir.setOutputID(5);
    }

    return ir;
}

void verifyClose(
    const std::string& name,
    const tensor& actual,
    const tensor& expected
) {
    if (actual.getShape() != expected.getShape()) {
        throw std::runtime_error(name + ": shape mismatch");
    }

    const auto actualData = actual.getData();
    const auto expectedData = expected.getData();

    if (actualData.size() != expectedData.size()) {
        throw std::runtime_error(name + ": data size mismatch");
    }

    // Longer dot products can accumulate more rounding error.
    const float absoluteTolerance = 1e-4f;
    const float relativeTolerance = 1e-4f;
    float maximumError = 0.0f;

    for (std::size_t i = 0; i < actualData.size(); ++i) {
        float actualValue = actualData[i];
        float expectedValue = expectedData[i];

        if (!std::isfinite(actualValue) ||
            !std::isfinite(expectedValue)) {
            throw std::runtime_error(name + ": non-finite result");
        }

        float error = std::fabs(actualValue - expectedValue);
        float tolerance =
            absoluteTolerance +
            relativeTolerance * std::fabs(expectedValue);

        if (error > tolerance) {
            throw std::runtime_error(
                name + ": mismatch at index " +
                std::to_string(i) +
                ", expected " + std::to_string(expectedValue) +
                ", got " + std::to_string(actualValue)
            );
        }

        maximumError = std::max(maximumError, error);
    }

    std::cout << "  Verified " << name
              << ", max absolute error = "
              << std::scientific << maximumError
              << std::defaultfloat << '\n';
}

void consumeResult(const tensor& result) {
    const auto data = result.getData();

    if (!data.empty()) {
        benchmarkSink = data.front() + data.back();
    }
}

double measureOnce(Backend& backend, const IR& ir) {
    const auto start = std::chrono::steady_clock::now();

    tensor result = backend.execute(ir);

    const auto stop = std::chrono::steady_clock::now();

    // This extra read/copy is outside the measured interval.
    consumeResult(result);

    return std::chrono::duration<double, std::milli>(
        stop - start
    ).count();
}

Statistics summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());

    double total = 0.0;

    for (double sample : samples) {
        total += sample;
    }

    const std::size_t middle = samples.size() / 2;

    const double median =
        samples.size() % 2 == 0
        ? (samples[middle - 1] + samples[middle]) / 2.0
        : samples[middle];

    return {
        samples.front(),
        median,
        total / static_cast<double>(samples.size()),
        samples.back()
    };
}

int main(int argc, char* argv[]) {
    try {
        std::vector<int> sizes;

        // Optional arguments select matrix sizes:
        // ./benchmark_backends 128 256 512
        for (int i = 1; i < argc; ++i) {
            int size = std::stoi(argv[i]);

            if (size <= 0) {
                throw std::invalid_argument(
                    "Matrix sizes must be positive"
                );
            }

            sizes.push_back(size);
        }

        if (sizes.empty()) {
            sizes = {128, 256, 512};
        }

        const int warmups = 3;
        const int runs = 15;

        CPUBackend cpu;
        CUDABackend gpu;

        std::ofstream summaryFile("benchmark_summary.csv");
        std::ofstream samplesFile("benchmark_samples.csv");

        if (!summaryFile || !samplesFile) {
            throw std::runtime_error(
                "Could not open benchmark CSV files"
            );
        }

        summaryFile
            << "size,path,warmups,runs,min_ms,median_ms,"
               "mean_ms,max_ms,speedup_vs_cpu_unfused\n";

        samplesFile << "size,path,round,latency_ms\n";

        summaryFile << std::setprecision(10);
        samplesFile << std::setprecision(10);

        for (int size : sizes) {
            std::cout << "\nMatrix size: "
                      << size << " x " << size << '\n';

            tensor input = makeMatrix(size, size, 101);
            tensor weight = makeMatrix(size, size, 202);
            tensor bias = makeMatrix(size, size, 303);

            IR unfusedIR = makeNetwork(
                input, weight, bias, false
            );

            IR fusedIR = makeNetwork(
                input, weight, bias, true
            );

            // Verify every path before collecting timings.
            tensor expected = cpu.execute(unfusedIR);

            verifyClose(
                "CPU fused",
                cpu.execute(fusedIR),
                expected
            );

            verifyClose(
                "CUDA fused",
                gpu.execute(fusedIR),
                expected
            );

            const std::string names[3] = {
                "cpu_unfused",
                "cpu_fused",
                "cuda_fused"
            };

            Backend* backends[3] = {&cpu, &cpu, &gpu};
            const IR* programs[3] = {
                &unfusedIR, &fusedIR, &fusedIR
            };

            std::vector<double> samples[3];

            for (auto& pathSamples : samples) {
                pathSamples.reserve(runs);
            }

            // Warm up all paths before measurement.
            for (int round = 0; round < warmups; ++round) {
                for (int path = 0; path < 3; ++path) {
                    tensor result =
                        backends[path]->execute(*programs[path]);

                    consumeResult(result);
                }
            }

            // Rotate execution order to reduce order-related bias.
            for (int round = 0; round < runs; ++round) {
                for (int offset = 0; offset < 3; ++offset) {
                    int path = (round + offset) % 3;

                    samples[path].push_back(
                        measureOnce(
                            *backends[path],
                            *programs[path]
                        )
                    );
                }
            }

            Statistics stats[3] = {
                summarize(samples[0]),
                summarize(samples[1]),
                summarize(samples[2])
            };

            std::cout << std::fixed << std::setprecision(3)
                      << std::left << std::setw(16) << "Path"
                      << std::right
                      << std::setw(12) << "Median ms"
                      << std::setw(12) << "Min ms"
                      << std::setw(12) << "Max ms"
                      << std::setw(12) << "Speedup"
                      << '\n';

            for (int path = 0; path < 3; ++path) {
                double speedup =
                    stats[0].median / stats[path].median;

                std::cout
                    << std::left << std::setw(16) << names[path]
                    << std::right
                    << std::setw(12) << stats[path].median
                    << std::setw(12) << stats[path].minimum
                    << std::setw(12) << stats[path].maximum
                    << std::setw(12) << speedup
                    << '\n';

                summaryFile
                    << size << ',' << names[path] << ','
                    << warmups << ',' << runs << ','
                    << stats[path].minimum << ','
                    << stats[path].median << ','
                    << stats[path].mean << ','
                    << stats[path].maximum << ','
                    << speedup << '\n';

                for (int round = 0; round < runs; ++round) {
                    samplesFile
                        << size << ',' << names[path] << ','
                        << round + 1 << ','
                        << samples[path][round] << '\n';
                }
            }

            // Preserve completed sizes if a later size fails.
            summaryFile.flush();
            samplesFile.flush();

            if (!summaryFile || !samplesFile) {
                throw std::runtime_error(
                    "Failed to write benchmark results"
                );
            }
        }

        std::cout
            << "\nSaved benchmark_summary.csv"
            << " and benchmark_samples.csv\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}