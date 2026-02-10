import cufractal
import numpy as np
import pytest


def test_mandelbrot_shape():
    width, height = 128, 128
    result = cufractal.mandelbrot(width=width, height=height)

    assert result.shape == (height, width), f"Expected {(height, width)}, got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_mandelbrot_values_range():
    result = cufractal.mandelbrot(width=64, height=64, max_iter=100)

    assert np.all(result >= 0.0), "Values below 0.0 detected"
    assert np.all(result <= 1.0), "Values above 1.0 detected"


def test_mandelbrot_symmetry():
    img = cufractal.mandelbrot(
        width=128,
        height=128,
        xmin=-2.0,
        xmax=1.0,
        ymin=-1.0,
        ymax=1.0,
        max_iter=50
    )

    flipped = np.flipud(img)

    diff = np.abs(img - flipped).mean()
    assert diff < 0.01, f"Symmetry broken, mean diff: {diff}"


def test_cuda_fallback():
    result_cpu = cufractal.mandelbrot(use_cuda=False, width=64, height=64)
    result_gpu = cufractal.mandelbrot(use_cuda=True, width=64, height=64)

    diff = np.abs(result_cpu - result_gpu).mean()
    assert diff < 0.05, f"CUDA/CPU results differ too much: {diff}"


def test_invalid_inputs():
    with pytest.raises(RuntimeError):
        cufractal.mandelbrot(width=0, height=100)

    with pytest.raises(RuntimeError):
        cufractal.mandelbrot(width=100, height=0)

    with pytest.raises(RuntimeError):
        cufractal.mandelbrot(max_iter=0)


def test_small_resolution():
    result = cufractal.mandelbrot(width=1, height=1)
    assert result.shape == (1, 1)
    assert 0.0 <= result[0, 0] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
