#include <iostream>
#include "graph.h"
#include <stdexcept>
#include "optimizer.h"
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

    optimizeNode(outputNode);
}

