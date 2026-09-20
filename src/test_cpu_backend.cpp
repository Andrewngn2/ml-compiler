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
void testGraphLowering() {
    node input("Input", tensor({-2, 3, -1, 4}, {2, 2}));

    // The same input is used twice.
    node add("Add");
    add.addInput(&input);
    add.addInput(&input);

    node relu("ReLU");
    relu.addInput(&add);

    // This node does not contribute to the output.
    node unused("Input", tensor({100, 100, 100, 100}, {2, 2}));

    Graph graph;

    // Deliberately register nodes out of dependency order.
    graph.addNode(&relu);
    graph.addNode(&unused);
    graph.addNode(&add);
    graph.addNode(&input);
    graph.setOutputNode(&relu);

    // Skip optimization to test lowering directly.
    IR ir = graph.lowerToIR();
    const auto& instructions = ir.getInstructions();

    // Only Input, Add, and ReLU should be emitted.
    if (instructions.size() != 3) {
        throw std::runtime_error(
            "Lowering: expected exactly 3 instructions"
        );
    }

    if (instructions[0].operation != "Input" ||
        instructions[1].operation != "Add" ||
        instructions[2].operation != "ReLU") {
        throw std::runtime_error(
            "Lowering: incorrect dependency order"
        );
    }

    // Both Add operands must reuse the same input ID.
    if (instructions[1].inputs != std::vector<int>{0, 0}) {
        throw std::runtime_error(
            "Lowering: shared input was not reused"
        );
    }

    if (instructions[2].inputs != std::vector<int>{1}) {
        throw std::runtime_error(
            "Lowering: ReLU references the wrong value"
        );
    }

    if (ir.getOutputID() != 2) {
        throw std::runtime_error(
            "Lowering: incorrect output ID"
        );
    }

    CPUBackend backend;

    expectTensor(
        "reachable nodes, shared inputs, and dependency order",
        backend.execute(ir),
        {0, 6, 0, 8},
        {2, 2}
    );

    ir.print();
}
void testRuntimeInputs() {
    node input("Input", tensor({-2, 3, -1, 4}, {2, 2}));

    node relu("ReLU");
    relu.addInput(&input);

    Graph graph;
    graph.addNode(&input);
    graph.addNode(&relu);
    graph.setOutputNode(&relu);

    // Lower only once.
    IR ir = graph.lowerToIR();
    CPUBackend backend;

    // In this graph, Input is value 0 and ReLU is value 1.
    expectTensor(
        "original input",
        backend.execute(ir),
        {0, 3, 0, 4},
        {2, 2}
    );

    std::unordered_map<int, tensor> bindings;
    bindings.emplace(
        0,
        tensor({5, -6, 7, -8}, {2, 2})
    );

    expectTensor(
        "replacement runtime input",
        backend.execute(ir, bindings),
        {5, 0, 7, 0},
        {2, 2}
    );

    // The override must not change the stored IR input.
    expectTensor(
        "original input preserved",
        backend.execute(ir),
        {0, 3, 0, 4},
        {2, 2}
    );

    bindings.at(0) = tensor({1, 2, 3}, {3});

    bool rejectedShape = false;

    try {
        backend.execute(ir, bindings);
    }
    catch (const std::runtime_error&) {
        rejectedShape = true;
    }

    if (!rejectedShape) {
        throw std::runtime_error(
            "Runtime inputs: incompatible shape was accepted"
        );
    }

    // Value 1 is ReLU, so it cannot be overridden as an input.
    bindings.clear();
    bindings.emplace(
        1,
        tensor({1, 2, 3, 4}, {2, 2})
    );

    bool rejectedID = false;

    try {
        backend.execute(ir, bindings);
    }
    catch (const std::runtime_error&) {
        rejectedID = true;
    }

    if (!rejectedID) {
        throw std::runtime_error(
            "Runtime inputs: non-Input ID was accepted"
        );
    }

    std::cout << "PASS: runtime input validation\n";
}

int main() {
    try {
        testNetwork(false);
        testNetwork(true);
        testConstantFolding();
        testReLU();
        testGraphLowering();
        std::cout << "\nAll CPU backend tests passed!\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}