#pragma once

#include "optimizationPass.h"
//inheritance
class ConstantFoldingPass : public OptimizationPass {
public:
    void run(node* currentNode) override;
};