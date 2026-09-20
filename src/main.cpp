#include "tensor.h"
#include "node.h"
#include "optimizer.h"
#include <iostream>
#include <chrono>
#include <tuple>
#include "graph.h"
#include "CPUBackend.h"

tensor createMatrix(int rows, int cols){
    std::vector<float> data(rows*cols, 1.0f);
    return tensor(data,{rows,cols});
}

int main() {

    // ============================================================
    // TEST 1: MatMul + Add Fusion
    // ============================================================

    std::cout << "===== TEST 1: MatMul + Add Fusion =====" << std::endl;

    tensor input_tensor(
        {1, 2,
         3, 4},
        {2, 2}
    );

    tensor weight_tensor(
        {6, 7,
         6, 7},
        {2, 2}
    );

    tensor bias_tensor(
        {1, 2,
         1, 2},
        {2, 2}
    );

    // Create input nodes
    node input_node("Input", input_tensor);
    node weight_node("Input", weight_tensor);
    node bias_node("Input", bias_tensor);

    // MatMul node
    node matmul_node("MatMul");
    matmul_node.addInput(&input_node);
    matmul_node.addInput(&weight_node);

    // Add node
    node add_node("Add");
    add_node.addInput(&matmul_node);
    add_node.addInput(&bias_node);

    // ReLU node
    node relu_node("ReLU");
    relu_node.addInput(&add_node);

    // Create graph
    Graph graph;

    graph.addNode(&input_node);
    graph.addNode(&weight_node);
    graph.addNode(&bias_node);
    graph.addNode(&matmul_node);
    graph.addNode(&add_node);
    graph.addNode(&relu_node);

    graph.setOutputNode(&relu_node);

    std::cout << "\nGraph before optimization:\n";
    graph.printNodes();

    std::cout << "\nResult before optimization:\n";
    graph.execute().print();

    // Optimize
    graph.optimize();

    
    IR ir = graph.lowerToIR();

    std::cout << "\nIR:\n";
    ir.print();

    CPUBackend cpu;

    tensor result = cpu.execute(ir);

    std::cout << "\nCPU Backend result:\n";
    result.print();

    std::cout << "\nGraph after optimization:\n";
    graph.printNodes();

    std::cout << "\nResult after optimization:\n";
    graph.execute().print();


    // ============================================================
    // TEST 2: Constant Folding
    // ============================================================

    std::cout << "\n\n===== TEST 2: Constant Folding =====" << std::endl;

    tensor constant_a(
        {1, 2,
         3, 4},
        {2, 2}
    );

    tensor constant_b(
        {5, 6,
         7, 8},
        {2, 2}
    );

    // Constant nodes
    node constant_node_a("Constant", constant_a);
    node constant_node_b("Constant", constant_b);

    // Add node
    node constant_add("Add");
    constant_add.addInput(&constant_node_a);
    constant_add.addInput(&constant_node_b);

    // Graph
    Graph constant_graph;

    constant_graph.addNode(&constant_node_a);
    constant_graph.addNode(&constant_node_b);
    constant_graph.addNode(&constant_add);

    constant_graph.setOutputNode(&constant_add);

    std::cout << "\nGraph before optimization:\n";
    constant_graph.printNodes();

    std::cout << "\nResult before optimization:\n";
    constant_graph.execute().print();

    // Optimize
    constant_graph.optimize();

    std::cout << "\nGraph after optimization:\n";
    constant_graph.printNodes();

    std::cout << "\nResult after optimization:\n";
    constant_graph.execute().print();



    return 0;
}

