#pragma once

#include "backend.h"

class CUDABackend : public Backend {
public:
    tensor execute(
        const IR& ir,
        const std::unordered_map<int, tensor>& inputBindings = {}
    ) override;
};