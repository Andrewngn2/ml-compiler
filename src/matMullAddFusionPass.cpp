#include "matMulAddFusionPass.h"
#include "optimizer.h"

void MatMulAddFusionPass::run(node* currentNode) {

    if (canFuseMatMulAdd(currentNode)) {
        fuseMatMulAdd(currentNode);
    }
}