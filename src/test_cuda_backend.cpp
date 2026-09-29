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
#include "cudaKernels.h"
#include <algorithm>

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


void testFusedNetwork() {
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

    graph.optimize();

    if (add.getOperation() != "FusedMatMulAdd") {
        throw std::runtime_error("Network was not fused");
    }

    IR ir = graph.lowerToIR();

    CPUBackend cpu;
    CUDABackend gpu;

    expectTensor(
        "CPU fused network",
        cpu.execute(ir),
        {19, 23, 43, 51},
        {2, 2}
    );

    expectTensor(
        "CUDA fused network",
        gpu.execute(ir),
        {19, 23, 43, 51},
        {2, 2}
    );

    // This graph assigns its sole Input value ID 0.
    std::unordered_map<int, tensor> bindings;
    bindings.emplace(
        0,
        tensor({-1, -2, 3, 4}, {2, 2})
    );

    expectTensor(
        "CUDA fused network with replacement input",
        gpu.execute(ir, bindings),
        {0, 0, 43, 51},
        {2, 2}
    );
}

void expectClose(
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

    const float absoluteTolerance = 1e-5f;
    const float relativeTolerance = 1e-4f;
    float maxError = 0.0f;

    for (std::size_t i = 0; i < actualData.size(); ++i) {
        float actualValue = actualData[i];
        float expectedValue = expectedData[i];

        if (!std::isfinite(actualValue) ||
            !std::isfinite(expectedValue)) {
            throw std::runtime_error(
                name + ": non-finite value at index " +
                std::to_string(i)
            );
        }

        float error = std::fabs(actualValue - expectedValue);
        float allowedError =
            absoluteTolerance +
            relativeTolerance * std::fabs(expectedValue);

        if (error > allowedError) {
            throw std::runtime_error(
                name + ": mismatch at index " +
                std::to_string(i) +
                ", expected " + std::to_string(expectedValue) +
                ", got " + std::to_string(actualValue)
            );
        }

        maxError = std::max(maxError, error);
    }

    std::cout << "PASS: " << name
              << " (max absolute error: " << maxError << ")\n";
}

// Produce repeatable mixed-sign fractional data.
tensor makeTestMatrix(int rows, int columns, int seed) {
    std::vector<float> data(
        static_cast<std::size_t>(rows) * columns
    );

    for (std::size_t i = 0; i < data.size(); ++i) {
        int value =
            (static_cast<int>(i % 29) * 7 + seed) % 29 - 14;

        data[i] = static_cast<float>(value) / 10.0f;
    }

    return tensor(data, {rows, columns});
}
void testKnownRectangularProduct() {
    tensor a(
        {1, 2, 3,
         4, 5, 6},
        {2, 3}
    );

    tensor b(
        { 1, -1,
          2,  0,
         -1,  3},
        {3, 2}
    );

    tensor bias(
        {1, -2,
         3, -4},
        {2, 2}
    );

    tensor expected(
        {3, 6,
         11, 10},
        {2, 2}
    );

    expectClose(
        "known rectangular CPU product",
        fusedMatMulAdd2d(a, b, bias),
        expected
    );

    expectClose(
        "known rectangular CUDA product",
        cudaFusedMatMulAdd(a, b, bias),
        expected
    );
}

void testRectangularCase(int rows, int shared, int columns) {
    tensor a = makeTestMatrix(rows, shared, 1);
    tensor b = makeTestMatrix(shared, columns, 5);
    tensor bias = makeTestMatrix(rows, columns, 9);

    tensor expected = fusedMatMulAdd2d(a, b, bias);
    tensor actual = cudaFusedMatMulAdd(a, b, bias);

    std::string name =
        "CUDA rectangular " +
        std::to_string(rows) + "x" +
        std::to_string(shared) + " times " +
        std::to_string(shared) + "x" +
        std::to_string(columns);

    expectClose(name, actual, expected);
}
void expectInvalidFusedShape(
    const std::string& name,
    const tensor& a,
    const tensor& b,
    const tensor& bias
) {
    try {
        cudaFusedMatMulAdd(a, b, bias);
    }
    catch (const std::invalid_argument& error) {
        std::cout << "PASS: " << name
                  << " (" << error.what() << ")\n";
        return;
    }

    throw std::runtime_error(
        name + ": invalid shapes were accepted"
    );
}

void testInvalidFusedShapes() {
    tensor a = makeTestMatrix(2, 3, 1);
    tensor b = makeTestMatrix(3, 4, 2);
    tensor bias = makeTestMatrix(2, 4, 3);

    expectInvalidFusedShape(
        "reject incompatible matrix dimensions",
        a,
        makeTestMatrix(2, 4, 4),
        bias
    );

    // Same element count as the correct bias, but wrong shape.
    expectInvalidFusedShape(
        "reject incorrect bias shape",
        a,
        b,
        makeTestMatrix(4, 2, 5)
    );
}

int main() {
    try {
        testInputAndBindings();
        testConstant();
        testConsecutiveReLU();
        testUnsupportedOperation();
        testFusedNetwork();
        testKnownRectangularProduct();
        testRectangularCase(3, 5, 7);
        testRectangularCase(17, 19, 23);
        testInvalidFusedShapes();
        std::cout << "\nAll CUDA backend tests passed!\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}