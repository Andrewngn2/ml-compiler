#include "optimizer.h"

bool canFuseMatMulAdd(node* addNode) {
    //checks if it is an add node
    if (addNode->getOperation() != "Add") {
        return false;
    }
    //gets inputs from add node
    std::vector<node*> inputs = addNode->getInputs();
    //makes sure it has two inputs
    if (inputs.size() != 2) {
        return false;
    }
    // checks if either one is matmul
    if (inputs[0]->getOperation() == "MatMul") {
        return true;
    }

    if (inputs[1]->getOperation() == "MatMul") {
        return true;
    }

    return false;
}

void fuseMatMulAdd(node* addNode){

    if(!canFuseMatMulAdd(addNode)){
        return;
    }
//gets inputs from add
    std::vector<node*> addInputs = addNode->getInputs();
//creates matmulnode instance and bias node instane
    node* matmulNode =nullptr;
    node* biasNode = nullptr;

    if(addInputs[0]->getOperation() == "MatMul"){
        matmulNode = addInputs[0];
        biasNode = addInputs[1];
    } 
    else{
        matmulNode = addInputs[1];
        biasNode = addInputs[0];
    }
    //gets matmulinputs
    std::vector<node*> matmulInputs = matmulNode ->getInputs();

    node* inputNode = matmulInputs[0];
    node* weightNode = matmulInputs[1];
//combines all inputs
    std::vector<node*> fusedInputs={ inputNode, weightNode, biasNode};
//transform
    addNode->setOperation("FusedMatMulAdd");
    addNode->setInputs(fusedInputs);

}
void optimizeNode(node* currentNode, OptimizationPass* pass) {

    std::vector<node*> inputs = currentNode->getInputs();

    for (node* input : inputs) {
        optimizeNode(input, pass);
    }

    pass->run(currentNode);
}
bool isConstant(node* currentNode) {
    return currentNode->getOperation() == "Constant";
}

bool canFoldAdd(node* addNode) {

    if (addNode->getOperation() != "Add") {
        return false;
    }

    std::vector<node*> inputs = addNode->getInputs();

    if (inputs.size() != 2) {
        return false;
    }

    return isConstant(inputs[0]) && isConstant(inputs[1]);
}
void node::setValue(tensor newValue) {
    value = newValue;
}

void foldAdd(node* addNode) {

    if (!canFoldAdd(addNode)) {
        return;
    }

    std::vector<node*> inputs = addNode->getInputs();

    tensor a = inputs[0]->getValue();
    tensor b = inputs[1]->getValue();

    tensor result = a.add(b);

    addNode->setOperation("Constant");
    addNode->setInputs({});
    addNode->setValue(result);
}

void Optimizer::addPass(OptimizationPass* pass) {
    passes.push_back(pass);
}
void Optimizer::optimize(node* outputNode) {

    for (OptimizationPass* pass : passes) {
        optimizeNode(outputNode,pass);
    }
}