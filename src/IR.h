#pragma once

#include <string>
#include <vector>

struct IRInstruction {
    std::string operation;
    std::vector<int> inputs;
    int output;
};
class IR {
private:
    std::vector<IRInstruction> instructions;

public:
    void addInstruction(IRInstruction instruction);
    void print() const;
};