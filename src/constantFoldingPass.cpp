#include "constantFoldingPass.h"
#include "optimizer.h"

void ConstantFoldingPass::run(node* currentNode) {

    if (canFoldAdd(currentNode)) {
        foldAdd(currentNode);
    }
}