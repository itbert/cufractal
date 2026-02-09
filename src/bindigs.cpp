#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "fractal.h"
#include <stdexcept>

namespace py = pybind11;

py::array_t<float> mandelbrot_wrapper(
    int width,
    int height,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    int max_iter,
    bool use_cuda
) {
    std::vector<float> result;
    
    try {
        if (use_cuda) {
            result = mandelbrot_cuda(width, height, xmin, xmax, ymin, ymax, max_iter);
        } else {
            result = mandelbrot_cpu(width, height, xmin, xmax, ymin, ymax, max_iter);
        }
    } catch (const std::runtime_error& e) {
        if (use_cuda && std::string(e.what()).find("CUDA") != std::string::npos) {
            py::print("ne cuda");
            result = mandelbrot_cpu(width, height, xmin, xmax, ymin, ymax, max_iter);
        } else {
            throw;
        }
    }
    
    py::array_t<float> output({height, width});
    std::memcpy(output.mutable_data(), result.data(), result.size() * sizeof(float));
    
    return output;
}

PYBIND11_MODULE(cufractal, m) {
    m.doc() = "GPU yes";
    
    m.def("mandelbrot", 
        &mandelbrot_wrapper,
        py::arg("width") = 1024,
        py::arg("height") = 1024,
        py::arg("xmin") = -2.0f,
        py::arg("xmax") = 1.0f,
        py::arg("ymin") = -1.5f,
        py::arg("ymax") = 1.5f,
        py::arg("max_iter") = 1000,
        py::arg("use_cuda") = true,
        R"pbdoc(        
        Parameters
        ----------
        width : int
            Image width in pixels
        height : int
            Image height in pixels
        xmin : float
            Left boundary of complex plane
        xmax : float
            Right boundary of complex plane
        ymin : float
            Bottom boundary of complex plane
        ymax : float
            Top boundary of complex plane
        max_iter : int
            Maximum iterations per pixel
        use_cuda : bool
            Use GPU acceleration (falls back to CPU if unavailable)
        
        Returns
        -------
        numpy.ndarray
            2D array of shape (height, width) with values in [0.0, 1.0]
    )pbdoc");
    
    m.def("mandelbrot_cpu", &mandelbrot_cpu, "CPU-only implementation");
    m.def("mandelbrot_cuda", &mandelbrot_cuda, "CUDA implementation (throws if unavailable)");
}