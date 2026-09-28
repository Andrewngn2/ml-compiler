#include "cudaKernels.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstddef>
// Helpers used only inside this file.
namespace {

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") +
            cudaGetErrorString(status)
        );
    }
}

__global__ void reluKernel(
    const float* input,
    float* output,
    std::size_t count
) {
    std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    // Each thread processes elements separated by the grid size.
    std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (; i < count; i += stride) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}


__global__ void fusedMatMulAddKernel(
    const float* a,
    const float* b,
    const float* bias,
    float* output,
    std::size_t outputCount,
    int sharedDimension,
    int outputColumns
) {
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (; index < outputCount; index += stride) {
        std::size_t row = index / outputColumns;
        std::size_t col = index % outputColumns;

        float sum = 0.0f;

        for (int k = 0; k < sharedDimension; ++k) {
            sum +=
                a[row * sharedDimension + k] *
                b[static_cast<std::size_t>(k) * outputColumns + col];
        }

        output[index] = sum + bias[index];
    }
}

// Owns one GPU allocation and frees it when the object leaves scope.
class DeviceBuffer {
public:
    float* data = nullptr;

    explicit DeviceBuffer(std::size_t bytes) {
        checkCuda(cudaMalloc(
            reinterpret_cast<void**>(&data), bytes
        ));
    }

    ~DeviceBuffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }

    // Prevent two objects from owning the same allocation.
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

} // namespace

tensor cudaRelu(const tensor& input) {
    const std::vector<float> hostInput = input.getData();
    const std::vector<int> shape = input.getShape();

    const std::size_t count = hostInput.size();

    // Avoid allocating memory or launching a kernel for empty data.
    if (count == 0) {
        return input;
    }

    const std::size_t bytes = count * sizeof(float);
    std::vector<float> hostOutput(count);

    DeviceBuffer deviceInput(bytes);
    DeviceBuffer deviceOutput(bytes);

    checkCuda(cudaMemcpy(
        deviceInput.data, //access class pointer member
        hostInput.data(), //call vector method returning pointer to elements
        bytes,
        cudaMemcpyHostToDevice
    ));

    const int threadsPerBlock = 256;

    const std::size_t requiredBlocks =
        1 + (count - 1) / threadsPerBlock;

    // Limit the grid size; the kernel loop handles remaining elements.
    const int blocks = static_cast<int>(
        requiredBlocks < 1024 ? requiredBlocks : 1024
    );

    reluKernel<<<blocks, threadsPerBlock>>>(
        deviceInput.data,
        deviceOutput.data,
        count
    );

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(
        hostOutput.data(),
        deviceOutput.data,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    return tensor(hostOutput, shape);
}

tensor cudaFusedMatMulAdd(
    const tensor& a,
    const tensor& b,
    const tensor& bias
) {
    const auto aShape = a.getShape();
    const auto bShape = b.getShape();
    const auto biasShape = bias.getShape();

    if (aShape.size() != 2 ||
        bShape.size() != 2 ||
        biasShape.size() != 2) {
        throw std::invalid_argument(
            "CUDA fused matmul + add requires 2D tensors"
        );
    }

    if (aShape[0] <= 0 || aShape[1] <= 0 ||
        bShape[0] <= 0 || bShape[1] <= 0 ||
        biasShape[0] <= 0 || biasShape[1] <= 0) {
        throw std::invalid_argument(
            "CUDA fused matmul + add requires positive dimensions"
        );
    }

    const int rows = aShape[0];
    const int sharedDimension = aShape[1];
    const int columns = bShape[1];

    if (sharedDimension != bShape[0]) {
        throw std::invalid_argument(
            "Matrix dimensions do not match for fused matmul + add"
        );
    }

    if (biasShape[0] != rows || biasShape[1] != columns) {
        throw std::invalid_argument(
            "Bias shape does not match matmul output"
        );
    }

    const auto aData = a.getData();
    const auto bData = b.getData();
    const auto biasData = bias.getData();

    const std::size_t aCount =
        static_cast<std::size_t>(rows) * sharedDimension;

    const std::size_t bCount =
        static_cast<std::size_t>(sharedDimension) * columns;

    const std::size_t outputCount =
        static_cast<std::size_t>(rows) * columns;

    if (aData.size() != aCount ||
        bData.size() != bCount ||
        biasData.size() != outputCount) {
        throw std::invalid_argument(
            "Tensor data size does not match shape"
        );
    }

    const std::size_t aBytes = aCount * sizeof(float);
    const std::size_t bBytes = bCount * sizeof(float);
    const std::size_t outputBytes = outputCount * sizeof(float);

    std::vector<float> outputData(outputCount);

    DeviceBuffer deviceA(aBytes);
    DeviceBuffer deviceB(bBytes);
    DeviceBuffer deviceBias(outputBytes);
    DeviceBuffer deviceOutput(outputBytes);

    checkCuda(cudaMemcpy(
        deviceA.data,
        aData.data(),
        aBytes,
        cudaMemcpyHostToDevice
    ));

    checkCuda(cudaMemcpy(
        deviceB.data,
        bData.data(),
        bBytes,
        cudaMemcpyHostToDevice
    ));

    checkCuda(cudaMemcpy(
        deviceBias.data,
        biasData.data(),
        outputBytes,
        cudaMemcpyHostToDevice
    ));

    const int threadsPerBlock = 256;

    const std::size_t requiredBlocks =
        1 + (outputCount - 1) / threadsPerBlock;

    const int blocks = static_cast<int>(
        requiredBlocks < 1024 ? requiredBlocks : 1024
    );

    fusedMatMulAddKernel<<<blocks, threadsPerBlock>>>(
        deviceA.data,
        deviceB.data,
        deviceBias.data,
        deviceOutput.data,
        outputCount,
        sharedDimension,
        columns
    );

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(
        outputData.data(),
        deviceOutput.data,
        outputBytes,
        cudaMemcpyDeviceToHost
    ));

    return tensor(outputData, {rows, columns});
}