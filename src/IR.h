#pragma once

#include <string>
#include <vector>
#include "tensor.h"

struct IRInstruction {
    std::string operation;
    std::vector<int> inputs;
    int output;
    tensor value;
};
class IR {
private:
    std::vector<IRInstruction> instructions;
    int outputID = -1;

public:
    void addInstruction(IRInstruction instruction);
    void print() const;
    void setOutputID(int id);
    int getOutputID() const;
    const std::vector<IRInstruction>& getInstructions() const;
    
};