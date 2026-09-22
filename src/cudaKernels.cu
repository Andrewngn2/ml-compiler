#include "cudaKernels.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

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