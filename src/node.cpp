#include "node.h"
#include <iostream>
#include "tensor.h"
#include <stdexcept>


node::node(std::string operation, tensor value)
: operation(operation), value(value){}

node::node(std::string operation)
: operation(operation){}

void node::print()const{
    std::cout << "operation:" << operation << std::endl;

    std::cout << "Inputs: ";

    for (node* input : inputs)
    {
        std::cout << input-> operation <<" ";
    }
    
      std::cout << std::endl;
}
void node::addInput(node* input){
    inputs.push_back(input);
}
tensor node::execute(){
    if(operation == "Input"){
        return value;
    }
    if(operation == "MatMul"){// uses recursion to get a and b tensor to matmul
        tensor a = inputs[0]->execute();
        tensor b = inputs[1]->execute();
        return matMul2d(a,b);
    }
    if(operation == "Add"){
        tensor a = inputs[0]-> execute();
        tensor b = inputs[1] ->execute();
        return a.add(b);
    }
    if(operation == "ReLU"){
        tensor a = inputs[0]->execute();
        return a.ReLU();
    }
    if(operation == "FusedMatMulAdd"){
        tensor input = inputs[0]->execute();
        tensor weight = inputs[1]->execute();
        tensor bias = inputs[2]->execute();

        return fusedMatMulAdd2d(input, weight, bias);
    }

    throw std::invalid_argument("Unknown operation");
}
void node::printGraph(int depth) const{
    for(int i=0;i<depth; ++i)
        {
            std::cout << " ";
        }
    std::cout << operation <<std::endl;

    for(node* input: inputs){
        input->printGraph(depth +1);
    }

}
std::string node::getOperation() const{
    return operation;
}
std::vector<node*> node::getInputs() const{
    return inputs;
}
void node::setOperation(std::string newOperation){
    operation = newOperation;
}
void node::setInputs(std::vector<node*> newInputs){
    inputs = newInputs;
}