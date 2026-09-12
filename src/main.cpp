#include "tensor.h"
#include "node.h"
#include "optimizer.h"
#include <iostream>
#include <chrono>
#include <tuple>

tensor createMatrix(int rows, int cols){
    std::vector<float> data(rows*cols, 1.0f);
    return tensor(data,{rows,cols});
}

int main() {
    int size = 512;

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
    int runs = 5;

    long long average_time,max_time,min_time;
    std::tie(average_time,max_time,min_time) = benchmarkMatMul(input_tensor, weight_tensor, runs,1);
    std::cout << "naive matmul:\n";
    std::cout << "Average time: "
            << average_time
            << " microseconds\n";
    std::cout<< "Max time:" << max_time << "microseconds\n";
    std::cout<< "Min time:" <<min_time <<"microseconds\n";
    
     long long optim_average_time,optim_max_time, optim_min_time;
    std::tie(optim_average_time,optim_max_time,optim_min_time) = benchmarkMatMul(input_tensor, weight_tensor, runs,0);
    std::cout <<"optimized matmul-contiguous memory access\n";
    std::cout << "Average time: "
            << optim_average_time
            << " microseconds\n";
    std::cout<< "Max time:" << optim_max_time << "microseconds\n";
    std::cout<< "Min time:" << optim_min_time <<"microseconds\n";
    


   

    return 0;
}

