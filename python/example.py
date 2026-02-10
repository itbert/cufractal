import cufractal
import matplotlib.pyplot as plt
import time
import numpy as np


width, height = 1024, 1024
max_iter = 1000

start = time.time()
image_gpu = cufractal.mandelbrot(
    width=width,
    height=height,
    max_iter=max_iter,
    use_cuda=True
)

gpu_time = time.time() - start
print(gpu_time*1000)

if width * height <= 256 * 256:
    start = time.time()
    image_cpu = cufractal.mandelbrot(
        width=width,
        height=height,
        max_iter=max_iter,
        use_cuda=False
    )
    cpu_time = time.time() - start
    print(f"CPU: {cpu_time*1000:.2f} мс (ускорение: {cpu_time/gpu_time:.1f}x)")

plt.figure(figsize=(10, 10))
plt.imshow(image_gpu, cmap='magma', extent=[-2.0, 1.0, -1.5, 1.5])

plt.colorbar(label='Нормализованное количество итераций')
plt.title(f'Множество Мандельброта ({width}x{height})')

plt.xlabel('Re')
plt.ylabel('Im')
plt.tight_layout()
output_file = f'mandelbrot_{width}x{height}.png'
plt.savefig(output_file, dpi=150)
print(f"save that: {output_file}")

plt.show()
