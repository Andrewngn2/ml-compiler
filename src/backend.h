#pragma once

#include <unordered_map>

#include "tensor.h"
#include "IR.h"

class Backend {
public:
    virtual ~Backend() = default;

    virtual tensor execute(
        const IR& ir,
        const std::unordered_map<int, tensor>& inputBindings = {}
    ) = 0;
};