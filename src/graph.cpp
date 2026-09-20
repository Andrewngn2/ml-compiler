#include <iostream>
#include "graph.h"
#include <stdexcept>
#include "optimizer.h"
#include "graph.h"
#include "constantFoldingPass.h"
#include "matMulAddFusionPass.h"
#include <iostream>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <cstddef>
void Graph::addNode(node* newNode) {
    nodes.push_back(newNode);
}
void Graph::printNodes() const {
    for (node* currentNode : nodes) {
        std::cout << currentNode->getOperation() << std::endl;
    }

}
tensor Graph::execute() {
    if (outputNode == nullptr) {
        throw std::runtime_error("Graph has no output node");
    }

    return outputNode->execute();
}
void Graph::setOutputNode(node* output) {
    outputNode = output;
}
void Graph::optimize() {

    if (outputNode == nullptr) {
        throw std::runtime_error("Graph has no output node");
    }

    Optimizer optimizer;

    ConstantFoldingPass constantFolding;
    MatMulAddFusionPass matmulFusion;

    optimizer.addPass(&constantFolding);
    optimizer.addPass(&matmulFusion);

    optimizer.optimize(outputNode);
}

IR Graph::lowerToIR() const {
    if (outputNode == nullptr) {
        throw std::runtime_error("Graph has no output node");
    }

    IR ir;

    // Tracks nodes that already have an IR value.
    std::unordered_map<node*, int> valueIDs;

    // Tracks nodes currently being visited to detect cycles.
    std::unordered_set<node*> visiting;

    std::function<int(node*)> lowerNode;
        //Capture needed surrounding local variables by reference.
    lowerNode = [&](node* currentNode) -> int {
        if (currentNode == nullptr) {
            throw std::runtime_error("Graph contains a null input");
        }
            //iterator which acts like a node pointer
        auto existing = valueIDs.find(currentNode);
            //existing returns .end() when key is absent
        if (existing != valueIDs.end()) {
            return existing->second;//The value-IR ID
        }
        //in unordered set .count(value) returns 0 if absnet and 1 if present
        if (visiting.count(currentNode) != 0) {
            throw std::runtime_error("Graph contains a cycle");
        }

        visiting.insert(currentNode);

        IRInstruction instruction;//pointer use ->
        instruction.operation = currentNode->getOperation();
        //instructionis an object
        // Input and Constant carry stored tensor values.
        if (instruction.operation == "Input" ||
            instruction.operation == "Constant") {
            instruction.value = currentNode->getValue();
        }
        else {
            // Emit dependencies before emitting this operation.
            for (node* input : currentNode->getInputs()) {
                instruction.inputs.push_back(lowerNode(input));
            }
        }

        int id = static_cast<int>(ir.getInstructions().size());
        instruction.output = id;

        ir.addInstruction(instruction);
        valueIDs.emplace(currentNode, id);
        visiting.erase(currentNode);

        return id;
    };

    int outputID = lowerNode(outputNode);
    ir.setOutputID(outputID);

    return ir;
}