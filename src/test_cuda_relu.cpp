#include "cudaKernels.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void testReLU(
    const std::string& name,
    const tensor& input
) {
    tensor expected = input.ReLU();
    tensor actual = cudaRelu(input);

    if (actual.getShape() != expected.getShape()) {
        throw std::runtime_error(name + ": shape mismatch");
    }

    const auto expectedData = expected.getData();
    const auto actualData = actual.getData();

    if (actualData.size() != expectedData.size()) {
        throw std::runtime_error(name + ": data size mismatch");
    }

    for (std::size_t i = 0; i < actualData.size(); ++i) {
        if (!std::isfinite(actualData[i]) ||
            std::fabs(actualData[i] - expectedData[i]) > 1e-6f) {
            throw std::runtime_error(
                name + ": incorrect result at index " +
                std::to_string(i)
            );
        }
    }

    std::cout << "PASS: " << name << '\n';
}

int main() {
    try {
        testReLU(
            "negative, zero, and positive values",
            tensor({-2, 0, 3, -1, 4, -5}, {2, 3})
        );

        testReLU(
            "single element",
            tensor({-7}, {1, 1})
        );

        // 257 elements require more than one block.
        std::vector<float> mediumData(257);

        for (std::size_t i = 0; i < mediumData.size(); ++i) {
            mediumData[i] = static_cast<float>(
                static_cast<int>(i % 21) - 10
            );
        }

        testReLU(
            "multiple blocks and partial final block",
            tensor(mediumData, {1, 257})
        );

        // Exceeds 1024 blocks * 256 threads.
        // Some threads must process a second element.
        std::vector<float> largeData(300001);

        for (std::size_t i = 0; i < largeData.size(); ++i) {
            largeData[i] = static_cast<float>(
                static_cast<int>(i % 101) - 50
            );
        }

        testReLU(
            "grid-stride loop",
            tensor(largeData, {1, 300001})
        );

        std::cout << "All CUDA ReLU tests passed!\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}