#include "tensor.h"
#include "node.h"
#include "optimizer.h"
#include <iostream>

int main() {

    tensor input_tensor({1, 2, 3, 4}, {2, 2});
    tensor weight_tensor({6, 7, 6, 7}, {2, 2});
    tensor bias_tensor({1, 2, 1, 2}, {2, 2});

    node input_node("Input", input_tensor);
    node weight_node("Input", weight_tensor);
    node bias_node("Input", bias_tensor);

    node matmul_node("MatMul");
    node add_node("Add");
    node relu_node("ReLU");

    matmul_node.addInput(&input_node);
    matmul_node.addInput(&weight_node);

    add_node.addInput(&matmul_node);
    add_node.addInput(&bias_node);

    relu_node.addInput(&add_node);

    std::cout << "===== BEFORE OPTIMIZATION =====\n";
    relu_node.printGraph();

    // Execute original graph
    tensor original_result = relu_node.execute();

    std::cout << "\nOriginal result:\n";
    original_result.print();

    // Fuse MatMul + Add
    fuseMatMulAdd(&add_node);

    std::cout << "\n===== AFTER OPTIMIZATION =====\n";
    relu_node.printGraph();

    // Execute optimized graph
    tensor optimized_result = relu_node.execute();

    std::cout << "\nOptimized result:\n";
    optimized_result.print();

    return 0;
}

