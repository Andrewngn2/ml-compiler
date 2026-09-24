#include "tensor.h"
#include "node.h"
#include "graph.h"
#include "IR.h"
#include "CPUBackend.h"
#include "CUDABackend.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Check both shape and numerical values.
void expectTensor(
    const std::string& name,
    const tensor& actual,
    const std::vector<float>& expectedData,
    const std::vector<int>& expectedShape
) {
    if (actual.getShape() != expectedShape) {
        throw std::runtime_error(name + ": shape mismatch");
    }

    const auto data = actual.getData();

    if (data.size() != expectedData.size()) {
        throw std::runtime_error(name + ": data size mismatch");
    }

    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i]) ||
            std::fabs(data[i] - expectedData[i]) > 1e-6f) {
            throw std::runtime_error(
                name + ": mismatch at index " +
                std::to_string(i) +
                ", expected " + std::to_string(expectedData[i]) +
                ", got " + std::to_string(data[i])
            );
        }
    }

    std::cout << "PASS: " << name << '\n';
}

// Verify that execution rejects an invalid request.
// The test's own failure is thrown outside the catch block.
void expectRejected(
    const std::string& name,
    Backend& backend,
    const IR& ir,
    const std::unordered_map<int, tensor>& bindings,
    const std::string& expectedMessage
) {
    try {
        backend.execute(ir, bindings);
    }
    catch (const std::runtime_error& error) {
        const std::string message = error.what();

        if (message.find(expectedMessage) == std::string::npos) {
            throw std::runtime_error(
                name + ": unexpected error: " + message
            );
        }

        std::cout << "PASS: " << name << '\n';
        return;
    }

    throw std::runtime_error(name + ": expected an exception");
}

void testInputAndBindings() {
    node input("Input", tensor({-2, 3, -1, 4}, {2, 2}));

    node relu("ReLU");
    relu.addInput(&input);

    Graph graph;
    graph.addNode(&input);
    graph.addNode(&relu);
    graph.setOutputNode(&relu);

    IR ir = graph.lowerToIR();

    CPUBackend cpu;
    CUDABackend gpu;

    // Exercise virtual dispatch through the shared interface.
    Backend& backend = gpu;

    expectTensor(
        "CPU baseline",
        cpu.execute(ir),
        {0, 3, 0, 4},
        {2, 2}
    );

    expectTensor(
        "CUDA through Backend interface",
        backend.execute(ir),
        {0, 3, 0, 4},
        {2, 2}
    );

    // This graph lowers to Input ID 0 and ReLU ID 1.
    std::unordered_map<int, tensor> bindings;
    bindings.emplace(
        0,
        tensor({5, -6, 7, -8}, {2, 2})
    );

    expectTensor(
        "CUDA runtime input",
        backend.execute(ir, bindings),
        {5, 0, 7, 0},
        {2, 2}
    );

    expectTensor(
        "stored input remains unchanged",
        backend.execute(ir),
        {0, 3, 0, 4},
        {2, 2}
    );

    bindings.at(0) = tensor({1, 2, 3}, {1, 3});

    expectRejected(
        "reject incorrect input shape",
        backend,
        ir,
        bindings,
        "Runtime input shape"
    );

    bindings.clear();
    bindings.emplace(
        1, // ReLU's ID, not an Input ID.
        tensor({1, 2, 3, 4}, {2, 2})
    );

    expectRejected(
        "reject binding to ReLU",
        backend,
        ir,
        bindings,
        "Runtime binding does not refer to an Input"
    );

    bindings.clear();
    bindings.emplace(
        99, // Does not exist.
        tensor({1, 2, 3, 4}, {2, 2})
    );

    expectRejected(
        "reject nonexistent input ID",
        backend,
        ir,
        bindings,
        "Runtime binding does not refer to an Input"
    );
}

void testConstant() {
    node constant(
        "Constant",
        tensor({-4, 2, 0, 9}, {2, 2})
    );

    node relu("ReLU");
    relu.addInput(&constant);

    Graph graph;
    graph.addNode(&constant);
    graph.addNode(&relu);
    graph.setOutputNode(&relu);

    IR ir = graph.lowerToIR();
    CUDABackend gpu;

    expectTensor(
        "Constant followed by CUDA ReLU",
        gpu.execute(ir),
        {0, 2, 0, 9},
        {2, 2}
    );

    std::unordered_map<int, tensor> bindings;
    bindings.emplace(
        0, // Constant ID, which must not be replaceable.
        tensor({1, 2, 3, 4}, {2, 2})
    );

    expectRejected(
        "reject binding to Constant",
        gpu,
        ir,
        bindings,
        "Runtime binding does not refer to an Input"
    );
}

void testConsecutiveReLU() {
    node input(
        "Input",
        tensor({-8, 0, 5, -3, 2, 7}, {2, 3})
    );

    node firstRelu("ReLU");
    firstRelu.addInput(&input);

    node secondRelu("ReLU");
    secondRelu.addInput(&firstRelu);

    Graph graph;

    // Registration order should not affect lowering.
    graph.addNode(&secondRelu);
    graph.addNode(&input);
    graph.addNode(&firstRelu);
    graph.setOutputNode(&secondRelu);

    IR ir = graph.lowerToIR();
    CUDABackend gpu;

    expectTensor(
        "consecutive CUDA ReLU operations",
        gpu.execute(ir),
        {0, 0, 5, 0, 2, 7},
        {2, 3}
    );
}

void testUnsupportedOperation() {
    node left("Input", tensor({1, 2, 3, 4}, {2, 2}));
    node right("Input", tensor({5, 6, 7, 8}, {2, 2}));

    node add("Add");
    add.addInput(&left);
    add.addInput(&right);

    Graph graph;
    graph.addNode(&left);
    graph.addNode(&right);
    graph.addNode(&add);
    graph.setOutputNode(&add);

    // Do not optimize: this test deliberately retains Add.
    IR ir = graph.lowerToIR();
    CUDABackend gpu;

    expectRejected(
        "reject unsupported Add operation",
        gpu,
        ir,
        {},
        "CUDA backend does not support operation: Add"
    );
}

int main() {
    try {
        testInputAndBindings();
        testConstant();
        testConsecutiveReLU();
        testUnsupportedOperation();

        std::cout << "\nAll CUDA backend tests passed!\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}