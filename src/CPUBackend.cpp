#include "CPUBackend.h"
#include <vector>
#include <stdexcept>
#include "IR.h"
#include "tensor.h"
tensor CPUBackend::execute(const IR& ir, const std::unordered_map<int, tensor>& inputBindings) {

    
    const std::vector<IRInstruction>& instructions = ir.getInstructions();


    std::vector<tensor> values(instructions.size());

    for (const auto& binding : inputBindings) {
    bool foundInput = false;
        //validation checks if id refers to input instruction and if it has right shape
    for (const auto& instruction : instructions) {
        if (instruction.output == binding.first &&
            instruction.operation == "Input") {
            foundInput = true;

            if (binding.second.getShape() !=
                instruction.value.getShape()) {
                throw std::runtime_error(
                    "Runtime input shape does not match IR input"
                );
            }

            break;
        }
    }

    if (!foundInput) {
        throw std::runtime_error(
            "Runtime binding does not refer to an Input instruction"
        );
    }
}

    for(const IRInstruction& instruction : instructions){

        if (instruction.operation == "Input") {
            auto binding = inputBindings.find(instruction.output);

            if (binding != inputBindings.end()) {
                values[instruction.output] = binding->second;
            }
            else {
                values[instruction.output] = instruction.value;
            }
        }
        else if (instruction.operation == "Constant") {
            values[instruction.output] = instruction.value;
        }
        else if(instruction.operation =="FusedMatMulAdd")
        {
            int inputID = instruction.inputs[0];
            int weightID= instruction.inputs[1];
            int biasID = instruction.inputs[2];

            tensor input = values[inputID];
            tensor weight = values[weightID];
            tensor bias = values[biasID];

            values[instruction.output] = fusedMatMulAdd2d(input, weight, bias);
        }
        else if(instruction.operation == "ReLU"){
            int inputID = instruction.inputs[0];
            tensor input = values[inputID];
            values[instruction.output] = input.ReLU();
        }
        else if (instruction.operation == "MatMul") {
            int leftID = instruction.inputs[0];
            int rightID = instruction.inputs[1];

            values[instruction.output] =
            matMul2d(values[leftID], values[rightID]);
        }
        else if (instruction.operation == "Add") {
            int leftID = instruction.inputs[0];
            int rightID = instruction.inputs[1];

            values[instruction.output] =
            values[leftID].add(values[rightID]);
        }
        else{
            throw std::invalid_argument("CPU Backend does not support operation: " + instruction.operation);
        }

    }
    return values.back();
}

