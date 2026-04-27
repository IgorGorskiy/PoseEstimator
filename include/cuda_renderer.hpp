#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>
#include <memory>

namespace pe {

// ── CUDA offscreen рендерер ───────────────────────────────────────────────
//
// Два прохода на GPU:
//   Pass 1: каждый CUDA-поток обрабатывает один треугольник грани →
//           атомарно записывает глубину в Z-буфер (device memory)
//   Pass 2: каждый CUDA-поток обрабатывает один отрезок ребра →
//           растеризует пиксели, проверяет Z-буфер, пишет маску
//
// Результат: cv::Mat CV_8UC1 (255 = видимое ребро, 0 = фон)
// Нет зависимости от оконной системы, OpenGL, Vulkan.

class CudaRenderer {
public:
    explicit CudaRenderer(CameraIntrinsics& K);
    ~CudaRenderer();

    CudaRenderer(const CudaRenderer&)            = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;

    // Загрузить геометрию модели в device memory
    void uploadModel(const Model3D& model);

    // Обновить параметры камеры
    void setCamera(const CameraIntrinsics& K);

    // Рендер. Возвращает CV_8UC1 маску рёбер на CPU.
    cv::Mat renderEdgeMask(const SE3& pose, int thickness = 1, float div = 1);

    bool isValid() const { return valid_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool valid_{false};
};

} // namespace pe
