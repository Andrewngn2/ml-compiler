#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.h"
#include "node.h"
#include "graph.h"
#include "IR.h"
#include "CPUBackend.h"

void expectTensor(
    const std::string& testName,
    const tensor& actual,
    const std::vector<float>& expectedData,
    const std::vector<int>& expectedShape
) {
    if (actual.getShape() != expectedShape) {
        throw std::runtime_error(testName + ": incorrect shape");
    }

    const auto data = actual.getData();

    if (data.size() != expectedData.size()) {
        throw std::runtime_error(testName + ": incorrect data size");
    }

    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i]) ||
            std::fabs(data[i] - expectedData[i]) > 1e-5f) {
            throw std::runtime_error(
                testName + ": mismatch at index " +
                std::to_string(i) +
                ", expected " + std::to_string(expectedData[i]) +
                ", got " + std::to_string(data[i])
            );
        }
    }

    std::cout << "PASS: " << testName << '\n';
}

// A fresh graph is constructed for each call.
void testNetwork(bool optimize) {
    node input("Input", tensor({1, 2, 3, 4}, {2, 2}));
    node weight("Input", tensor({6, 7, 6, 7}, {2, 2}));
    node bias("Input", tensor({1, 2, 1, 2}, {2, 2}));

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

    if (optimize) {
        graph.optimize();

        // Ensure this test actually exercises fused dispatch.
        if (add.getOperation() != "FusedMatMulAdd") {
            throw std::runtime_error(
                "Optimized network: MatMul + Add was not fused"
            );
        }
    }

    IR ir = graph.lowerToIR();
    CPUBackend backend;

    expectTensor(
        optimize ? "optimized network" : "unoptimized network",
        backend.execute(ir),
        {19, 23, 43, 51},
        {2, 2}
    );
}

void testConstantFolding() {
    node left("Constant", tensor({1, 2, 3, 4}, {2, 2}));
    node right("Constant", tensor({5, 6, 7, 8}, {2, 2}));

    node add("Add");
    add.addInput(&left);
    add.addInput(&right);

    Graph graph;
    graph.addNode(&left);
    graph.addNode(&right);
    graph.addNode(&add);
    graph.setOutputNode(&add);
    graph.optimize();

    // Ensure this test actually exercises a folded Constant.
    if (add.getOperation() != "Constant") {
        throw std::runtime_error(
            "Constant folding: Add was not folded"
        );
    }

    IR ir = graph.lowerToIR();
    CPUBackend backend;

    expectTensor(
        "constant folding and Constant dispatch",
        backend.execute(ir),
        {6, 8, 10, 12},
        {2, 2}
    );
}

void testReLU() {
    node input(
        "Input",
        tensor({-2, 0, 3, -1, 4, -5}, {2, 3})
    );

    node relu("ReLU");
    relu.addInput(&input);

    Graph graph;
    graph.addNode(&input);
    graph.addNode(&relu);
    graph.setOutputNode(&relu);

    IR ir = graph.lowerToIR();
    CPUBackend backend;

    expectTensor(
        "ReLU with negative, zero, and positive values",
        backend.execute(ir),
        {0, 0, 3, 0, 4, 0},
        {2, 3}
    );
}

int main() {
    try {
        testNetwork(false);
        testNetwork(true);
        testConstantFolding();
        testReLU();

        std::cout << "\nAll CPU backend tests passed!\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}