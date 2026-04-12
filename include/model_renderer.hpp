#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>
#include <memory>
#include <vector>

namespace pe {

class CudaRenderer;

class ModelRenderer {
public:
    ModelRenderer(const CameraIntrinsics& K, const Model3D& model);
    ~ModelRenderer();

    /// Рендер рёбер с Z-тестом.
    /// Vulkan если доступен, иначе CPU-fallback (программный Z-буфер).
    void render(const SE3& pose, cv::Mat& out, int thickness = 3) const;

    /// Проекция рёбер в 2D полилинии для TCD метрики.
    /// Всегда CPU — оптимизатор читает координаты точек, не пиксели.
    std::vector<std::vector<cv::Point2f>>
    project(const SE3& pose, float* visibleRatio = nullptr) const;

    bool hasGpu() const { return cuda_ != nullptr; }
    bool m_newFrame;

private:
    CameraIntrinsics K_;
    const Model3D&   model_;
    std::unique_ptr<CudaRenderer> cuda_;

    static constexpr double near_ = 0.001;

    cv::Point2f projectPoint(const Vec3d& p_cam) const;
    cv::Mat     buildZBufferCPU(const SE3& pose) const;

    static void rasterizeTriangleCPU(
        const cv::Point2f& p0, float z0,
        const cv::Point2f& p1, float z1,
        const cv::Point2f& p2, float z2,
        cv::Mat& zbuf);

    bool isVisible(const Edge3D& edge, const SE3& pose) const;
};

} // namespace pe
