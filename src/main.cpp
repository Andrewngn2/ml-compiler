#include "tensor.h"
#include "node.h"
#include "graph.h"
#include "IR.h"
#include "CPUBackend.h"

#ifdef MINITENSOR_WITH_CUDA
#include "CUDABackend.h"
#endif

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void checkResult(
    const std::string& name,
    const tensor& actual,
    const std::vector<float>& expected
) {
    if (actual.getShape() != std::vector<int>{2, 2}) {
        throw std::runtime_error(name + ": incorrect shape");
    }

    const auto data = actual.getData();

    if (data.size() != expected.size()) {
        throw std::runtime_error(name + ": incorrect data size");
    }

    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i]) ||
            std::fabs(data[i] - expected[i]) > 1e-5f) {
            throw std::runtime_error(
                name + ": incorrect value at index " +
                std::to_string(i)
            );
        }
    }

    std::cout << "PASS: " << name << '\n';
}

void demonstrateNetwork() {
    std::cout << "=== MatMul + Add + ReLU ===\n";

    node input(
        "Input",
        tensor({1, 2, 3, 4}, {2, 2})
    );

    node weight(
        "Constant",
        tensor({6, 7, 6, 7}, {2, 2})
    );

    node bias(
        "Constant",
        tensor({1, 2, 1, 2}, {2, 2})
    );

    node matmul("MatMul");
    matmul.addInput(&input);
    matmul.addInput(&weight);

    node add("Add");
    add.addInput(&matmul);
    add.addInput(&bias);

    node relu("ReLU");
    relu.addInput(&add);

    Graph graph;
    graph.addNode(&input);
    graph.addNode(&weight);
    graph.addNode(&bias);
    graph.addNode(&matmul);
    graph.addNode(&add);
    graph.addNode(&relu);
    graph.setOutputNode(&relu);

    std::cout << "\nIR before optimization:\n";
    IR originalIR = graph.lowerToIR();
    originalIR.print();

    graph.optimize();

    if (add.getOperation() != "FusedMatMulAdd") {
        throw std::runtime_error("MatMul + Add fusion failed");
    }

    IR optimizedIR = graph.lowerToIR();

    std::cout << "\nIR after optimization:\n";
    optimizedIR.print();

    std::cout << "\nInstructions: "
              << originalIR.getInstructions().size()
              << " -> "
              << optimizedIR.getInstructions().size()
              << '\n';

    CPUBackend cpu;

    checkResult(
        "unoptimized CPU result",
        cpu.execute(originalIR),
        {19, 23, 43, 51}
    );

    tensor cpuResult = cpu.execute(optimizedIR);

    std::cout << "\nOptimized CPU result:\n";
    cpuResult.print();

    checkResult(
        "optimized CPU result",
        cpuResult,
        {19, 23, 43, 51}
    );

#ifdef MINITENSOR_WITH_CUDA
    CUDABackend gpu;
    tensor gpuResult = gpu.execute(optimizedIR);

    std::cout << "\nCUDA result from the same optimized IR:\n";
    gpuResult.print();

    checkResult(
        "CUDA result",
        gpuResult,
        {19, 23, 43, 51}
    );
#else
    std::cout << "\nCUDA execution disabled in this build.\n";
#endif
}

void demonstrateConstantFolding() {
    std::cout << "\n=== Constant folding ===\n";

    node left(
        "Constant",
        tensor({1, 2, 3, 4}, {2, 2})
    );

    node right(
        "Constant",
        tensor({5, 6, 7, 8}, {2, 2})
    );

    node add("Add");
    add.addInput(&left);
    add.addInput(&right);

    Graph graph;
    graph.addNode(&left);
    graph.addNode(&right);
    graph.addNode(&add);
    graph.setOutputNode(&add);

    std::cout << "\nIR before constant folding:\n";
    graph.lowerToIR().print();

    graph.optimize();
    IR ir = graph.lowerToIR();

    if (add.getOperation() != "Constant" ||
        ir.getInstructions().size() != 1) {
        throw std::runtime_error("Constant folding failed");
    }

    std::cout << "\nIR after constant folding:\n";
    ir.print();

    CPUBackend cpu;
    tensor result = cpu.execute(ir);

    std::cout << "\nFolded result:\n";
    result.print();

    checkResult(
        "constant folding",
        result,
        {6, 8, 10, 12}
    );
}

int main() {
    try {
        demonstrateNetwork();
        demonstrateConstantFolding();

        std::cout << "\nMiniTensor demo completed successfully.\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}