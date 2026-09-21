#pragma once
#include <unordered_map>
#include "IR.h"
#include "tensor.h"
#include "backend.h"

class CPUBackend : public Backend {
public:
    tensor execute(const IR& ir,    //maps ir input ID 
        const std::unordered_map<int, tensor>& inputBindings= {}) override;
};