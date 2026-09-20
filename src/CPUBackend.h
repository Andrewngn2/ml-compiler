#pragma once
#include <unordered_map>
#include "IR.h"
#include "tensor.h"

class CPUBackend {
public:
    tensor execute(const IR& ir,    //maps ir input ID 
        const std::unordered_map<int, tensor>& inputBindings= {});
};