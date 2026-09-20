#pragma once

#include "IR.h"
#include "tensor.h"

class CPUBackend {
public:
    tensor execute(const IR& ir);
};