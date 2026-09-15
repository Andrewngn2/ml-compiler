#pragma once
#include "optimizationPass.h"
#include "node.h"

bool canFuseMatMulAdd(node* addNode);

void fuseMatMulAdd(node* addNode);
bool isConstant(node* currentNode);
bool canFoldAdd(node* addNode);
void foldAdd(node* addNode);
void optimizeNode(node* currentNode, OptimizationPass* pass);

class Optimizer {
private:
    std::vector<OptimizationPass*> passes;

public:
    void addPass(OptimizationPass* pass);
    void optimize(node* outputNode);
};
