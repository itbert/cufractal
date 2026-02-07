#include "fractal.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cmath>

__global__ void mandelbrot_kernel(
    float* output,
    int width,
    int height,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    int max_iter
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float real = xmin + x * (xmax - xmin) / width;
    float imag = ymin + y * (ymax - ymin) / height;
    
    float z_real = real;
    float z_imag = imag;
    int iter = 0;
    
    while (iter < max_iter && (z_real * z_real + z_imag * z_imag) < 4.0f) {
        float temp = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0f * z_real * z_imag + imag;
        z_real = temp;
        iter++;
    }
    
    float value;
    if (iter == max_iter) {
        value = 1.0f;
    } else {
        float log_zn = logf(z_real * z_real + z_imag * z_imag) / 2.0f;
        float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
        value = (iter + 1 - nu) / max_iter;
    }
    
    output[y * width + x] = fminf(fmaxf(value, 0.0f), 1.0f);
}

std::vector<float> mandelbrot_cuda(
    int width,
    int height,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    int max_iter
) {
    if (width <= 0 || height <= 0 || max_iter <= 0) {
        throw std::invalid_argument("Invalid dimensions or max_iter");
    }
    
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA-capable devices found");
    }
    
    std::vector<float> output(width * height);
    
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(float)));
    
    dim3 blockDim(16, 16);  // 256 потоков на блок
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    mandelbrot_kernel<<<gridDim, blockDim>>>(
        d_output,
        width,
        height,
        xmin,
        xmax,
        ymin,
        ymax,
        max_iter
    );
    
    CUDA_KERNEL_CHECK();
    
    CUDA_CHECK(cudaMemcpy(
        output.data(),
        d_output,
        width * height * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    
    CUDA_CHECK(cudaFree(d_output));
    
    return output;
}