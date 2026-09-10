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