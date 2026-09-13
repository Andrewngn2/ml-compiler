#pragma once

#include "node.h"

bool canFuseMatMulAdd(node* addNode);

void fuseMatMulAdd(node* addNode);
void optimizeNode(node* currentNode);