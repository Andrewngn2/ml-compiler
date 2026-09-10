#include <vector>
#include "tensor.h"
#include <string>
#pragma once
class node {
    private:
        std::string operation;
        std::vector<node*> inputs;
        tensor value;
    public:
        node(std::string operation, tensor value);
        node(std::string operation);
        void addInput(node* input);
        void printGraph(int depth = 0) const;
        std::string getOperation() const;
        std::vector<node*> getInputs() const;
        void print() const;
        tensor execute();
        void setOperation(std:: string newOperation);
        void setInputs(std::vector<node*> newInputs);
    };