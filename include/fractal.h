#pragma once

#include <vector>
#include <cstdint>

/**
 * Генерация фрактала Мондельброта
 * @param width Ширина
 * @param height Высота
 * @param xmin Минимальная вещественная часть
 * @param xmax Максимальная вещественная часть
 * @param ymin Минимальная мнимая часть
 * @param ymax Максимальная мнимая часть
 * @param max_iter Максимальное количество итераций
 * @return Вектор нормализованных значений
 */
std::vector<float> mandelbrot_cpu(
    int width,
    int height,
    float xmin = -2.0f,
    float xmax = 1.0f,
    float ymin = -1.5f,
    float ymax = 1.5f,
    int max_iter = 1000
);


std::vector<float> mandelbrot_cuda(
    int width,
    int height,
    float xmin = -2.0f,
    float xmax = 1.0f,
    float ymin = -1.5f,
    float ymax = 1.5f,
    int max_iter = 1000
);