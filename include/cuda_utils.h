#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            std::ostringstream oss; \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudaGetErrorString(error) \
                << " (" << cudaGetErrorName(error) << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define CUDA_KERNEL_CHECK() \
    do { \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)