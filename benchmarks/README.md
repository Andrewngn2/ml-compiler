# Benchmark records

Each run directory contains summary statistics, individual latency
samples, environment information, console output, and exit status.

## Recorded build

The published runs used the following command from the source directory:

```bash
nvcc -O3 -std=c++17 \
    benchmark_backends.cpp \
    tensor.cpp IR.cpp CPUBackend.cpp CUDABackend.cpp cudaKernels.cu \
    -o benchmark_backends
```

Execution:

```bash
/path/to/benchmark_backends 128 256 512 1024
```

## Methodology

- Workload: ReLU(A × B + bias).
- Square float32 inputs with full matrix bias.
- Seeds: 101 for input, 202 for weight, 303 for bias.
- Three warmups and fifteen measurements per path per matrix size.
- Correctness checked before timing.
- Path order rotated between measurement rounds.
- Host steady-clock timing around backend.execute().
- CUDA allocations, transfers, synchronization, and cleanup included.
- IR construction and correctness comparisons excluded.
- Speedup calculated from per-run median latency relative to CPU unfused.

Results reflect custom implementations on a shared computing system.
The CPU baseline is single-threaded and does not call an optimized BLAS
library. GPU kernel-only latency is not measured separately.

The CMake build is provided for reproducibility of the project.
The historical results above were produced with the explicit command
shown here; a different build configuration should be recorded as a new run.