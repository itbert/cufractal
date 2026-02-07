#include "fractal.h"
#include <cmath>
#include <stdexcept>

std::vector<float> mandelbrot_cpu(
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

    std::vector<float> output(width * height);

    const float dx = (xmax - xmin) / width;
    const float dy = (ymax - ymin) / height;

    for (int y = 0; y < height; ++y) {
        const float imag = ymin + y * dy;
        
        for (int x = 0; x < width; ++x) {
            const float real = xmin + x * dx;
            
            float z_real = real;
            float z_imag = imag;
            int iter = 0;
            
            while (iter < max_iter && (z_real * z_real + z_imag * z_imag) < 4.0f) {
                float temp = z_real * z_real - z_imag * z_imag + real;
                z_imag = 2.0f * z_real * z_imag + imag;
                z_real = temp;
                iter++;
            }
            
            float value = (iter < max_iter) 
                ? (static_cast<float>(iter) + 1.0f - std::log2(std::log2(z_real * z_real + z_imag * z_imag) / std::log(2.0f))) / max_iter
                : 1.0f;
            
            output[y * width + x] = std::min(std::max(value, 0.0f), 1.0f);
        }
    }

    return output;
}
