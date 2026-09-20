#include "IR.h"
#include <iostream>
#include "tensor.h"

void IR::addInstruction(IRInstruction instruction) {
    instructions.push_back(instruction);
}

void IR::print() const {

    for (const IRInstruction& instruction : instructions) {

        std::cout << instruction.output
                  << ": "
                  << instruction.operation;
            //checks if it has
        if (!instruction.inputs.empty()) {

            std::cout << " [";

            for (int i = 0; i < instruction.inputs.size(); ++i) {

                std::cout << instruction.inputs[i];

                if (i + 1 < instruction.inputs.size()) {
                    std::cout << ", ";
                }
            }

            std::cout << "]";
        }
        std::vector<int> shape = instruction.value.getShape();

        if (!shape.empty()) {
            std::cout << " shape=[";
            
            for (int i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];

                if (i + 1 < shape.size()) {
                    std::cout << ", ";
                }
            }

            std::cout << "]";
        }

        std::cout << std::endl;
    }
}
const std::vector<IRInstruction>& IR::getInstructions() const {
    return instructions;
}

void IR::setOutputID(int id) {
    outputID = id;
}

int IR::getOutputID() const {
    return outputID;
}