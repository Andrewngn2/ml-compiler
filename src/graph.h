
#pragma once

#include <vector>
#include "tensor.h"
#include "node.h"
//graph is a list of pointers to nodes.
class Graph {
private:
    std::vector<node*> nodes;
    node* outputNode = nullptr;
    

public:
    void addNode(node* newNode);
    void printNodes() const;
    tensor execute();
    void setOutputNode(node* output);
    void optimize();
};
