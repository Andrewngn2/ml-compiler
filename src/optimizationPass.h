#pragma once

#include "node.h"

class OptimizationPass {
public:
    virtual void run(node* currentNode) = 0;
};