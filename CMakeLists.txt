cmake_minimum_required(VERSION 3.24)

project(MiniTensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MINITENSOR_ENABLE_CUDA "Build CUDA backend and benchmarks" OFF)

enable_testing()

# Shared tensor, graph, optimizer, IR, and CPU implementation.
add_library(minitensor_cpu
    src/tensor.cpp
    src/node.cpp
    src/graph.cpp
    src/optimizer.cpp
    src/constantFoldingPass.cpp
    src/matMulAddFusionPass.cpp
    src/IR.cpp
    src/CPUBackend.cpp
)

target_include_directories(minitensor_cpu PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_executable(test_cpu_backend
    src/test_cpu_backend.cpp
)

target_link_libraries(test_cpu_backend PRIVATE minitensor_cpu)

add_test(
    NAME cpu_backend
    COMMAND test_cpu_backend
)

add_executable(minitensor_demo
    src/main.cpp
)

target_link_libraries(minitensor_demo PRIVATE minitensor_cpu)

add_test(
    NAME demo
    COMMAND minitensor_demo
)

if(MINITENSOR_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(minitensor_cuda
        src/CUDABackend.cpp
        src/cudaKernels.cu
    )

    set_target_properties(minitensor_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    target_link_libraries(minitensor_cuda PUBLIC
        minitensor_cpu
        CUDA::cudart
    )
    target_link_libraries(minitensor_demo PRIVATE minitensor_cuda)

    target_compile_definitions(minitensor_demo PRIVATE
        MINITENSOR_WITH_CUDA
    )

    add_executable(test_cuda_relu
        src/test_cuda_relu.cpp
    )

    target_link_libraries(test_cuda_relu PRIVATE minitensor_cuda)

    add_executable(test_cuda_backend
        src/test_cuda_backend.cpp
    )

    target_link_libraries(test_cuda_backend PRIVATE minitensor_cuda)

    add_executable(benchmark_backends
        src/benchmark_backends.cpp
    )

    target_link_libraries(benchmark_backends PRIVATE minitensor_cuda)

    add_test(
        NAME cuda_relu
        COMMAND test_cuda_relu
    )

    add_test(
        NAME cuda_backend
        COMMAND test_cuda_backend
    )
endif()