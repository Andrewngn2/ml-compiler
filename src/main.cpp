#include "tensor.h"
#include "node.h"
#include "optimizer.h"
#include <iostream>
#include <chrono>

tensor createMatrix(int rows, int cols){
    std::vector<float> data(rows*cols, 1.0f);
    return tensor(data,{rows,cols});
}

int main() {
    int size = 2048;

    tensor input_tensor = createMatrix(size, size);
    tensor weight_tensor = createMatrix(size,size);
    tensor bias_tensor = createMatrix(size,size);

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


    // -----------------------------
    // ORIGINAL GRAPH
    // -----------------------------

    std::cout << "===== ORIGINAL GRAPH =====\n";
    relu_node.printGraph();

    auto start = std::chrono::high_resolution_clock::now();
    tensor original_result = relu_node.execute();
    auto end = std::chrono::high_resolution_clock::now();
    auto original_time = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "\nOriginal result:\n";
    


    // -----------------------------
    // OPTIMIZE
    // -----------------------------

    fuseMatMulAdd(&add_node);

    const int iterations = 10;

    // -----------------------------
    // BENCHMARK ORIGINAL
    // -----------------------------

    // Rebuild the original graph
    add_node.setOperation("Add");

    std::vector<node*> original_inputs = {&matmul_node,&bias_node};

    add_node.setInputs(original_inputs);

    long long original_total = 0;

    for (int i = 0; i < iterations; ++i) {

        auto start = std::chrono::high_resolution_clock::now();

        tensor result = relu_node.execute();

        auto end = std::chrono::high_resolution_clock::now();

        original_total += std::chrono::duration_cast<std::chrono::microseconds>( end - start).count();
    }

    double original_average = static_cast<double>(original_total) / iterations;


    // -----------------------------
    // BENCHMARK OPTIMIZED
    // -----------------------------

    fuseMatMulAdd(&add_node);

    long long optimized_total = 0;

    for (int i = 0; i < iterations; ++i) {

        auto start = std::chrono::high_resolution_clock::now();

        tensor result = relu_node.execute();

        auto end = std::chrono::high_resolution_clock::now();

        optimized_total +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                end - start
            ).count();
    }

    double optimized_average =
        static_cast<double>(optimized_total) / iterations;


    // -----------------------------
    // RESULTS
    // -----------------------------

    std::cout << "\n===== BENCHMARK =====\n";

    std::cout << "Matrix size: "
            << size << " x " << size << "\n";

    std::cout << "Iterations: "
            << iterations << "\n";

    std::cout << "Original average: "
            << original_average << " microseconds\n";

    std::cout << "Optimized average: "
            << optimized_average << " microseconds\n";

    double speedup =
        original_average / optimized_average;

    std::cout << "Speedup: "
            << speedup << "x\n";
    

    return 0;
}

