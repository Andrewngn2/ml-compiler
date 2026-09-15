#pragma once

#include "optimizationPass.h"

class MatMulAddFusionPass : public OptimizationPass {
public:
    void run(node* currentNode) override;
};