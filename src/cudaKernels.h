#pragma once

#include "tensor.h"

tensor cudaRelu(const tensor& input);
tensor cudaFusedMatMulAdd(const tensor& a,const tensor& b,const tensor& bias);