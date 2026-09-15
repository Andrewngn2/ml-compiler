#include "IR.h"
#include <iostream>

void IR::addInstruction(IRInstruction instruction) {
    instructions.push_back(instruction);
}

void IR::print() const {
    for (const IRInstruction& instruction : instructions) {
        std::cout << instruction.output
                  << ": "
                  << instruction.operation
                  << std::endl;
    }
}