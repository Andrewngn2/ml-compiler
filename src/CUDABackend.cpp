#include "CUDABackend.h"
#include "cudaKernels.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

tensor CUDABackend::execute(
    const IR& ir,
    const std::unordered_map<int, tensor>& inputBindings
) {
    const auto& instructions = ir.getInstructions();

    std::vector<tensor> values(instructions.size());

    // Tracks which values have actually been produced.
    std::vector<bool> ready(instructions.size(), false);

    auto validID = [&](int id) -> bool {
        return id >= 0 &&
               static_cast<std::size_t>(id) < values.size();
    };

    // Validate runtime input replacements before execution.
    for (const auto& binding : inputBindings) {
        bool foundInput = false;

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
                "Runtime binding does not refer to an Input"
            );
        }
    }

    for (const auto& instruction : instructions) {
        int outputID = instruction.output;

        if (!validID(outputID)) {
            throw std::runtime_error("Invalid IR output ID");
        }

        if (ready[outputID]) {
            throw std::runtime_error("Duplicate IR output ID");
        }

        if (instruction.operation == "Input") {
            auto binding = inputBindings.find(outputID);

            if (binding != inputBindings.end()) {
                values[outputID] = binding->second;
            }
            else {
                values[outputID] = instruction.value;
            }
        }
        else if (instruction.operation == "Constant") {
            values[outputID] = instruction.value;
        }
        else if (instruction.operation == "ReLU") {
            if (instruction.inputs.size() != 1) {
                throw std::runtime_error(
                    "ReLU requires exactly one input"
                );
            }

            int inputID = instruction.inputs[0];

            if (!validID(inputID) || !ready[inputID]) {
                throw std::runtime_error(
                    "ReLU input has not been produced"
                );
            }

            values[outputID] = cudaRelu(values[inputID]);
        }
        else {
            throw std::runtime_error(
                "CUDA backend does not support operation: " +
                instruction.operation
            );
        }

        ready[outputID] = true;
    }

    int outputID = ir.getOutputID();

    if (!validID(outputID) || !ready[outputID]) {
        throw std::runtime_error("IR has no valid result");
    }

    return values[outputID];
}