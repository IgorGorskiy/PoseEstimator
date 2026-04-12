#pragma once
#include "types.hpp"
#include "model_renderer.hpp"
#include "image_preprocessor.hpp"

namespace pe {

/// Метрика сопоставления проекции 3D-контуров с обнаруженными рёбрами.
///
/// Используется Truncated Chamfer Distance (TCD):
///   score = (1/N) * sum_i  min(DT(p_i), tau)
/// где p_i – пиксели спроецированных рёбер модели, DT – distance transform,
/// tau – порог усечения (cfg.chamfer_trunc).
///
/// Усечение tau делает метрику робастной к выбросам (частичная видимость,
/// перекрытие крепёжными изделиями).
class ContourMatcher {
public:
    ContourMatcher(const CameraIntrinsics& K,
                   const Model3D& model,
                   const PipelineConfig& cfg);

    /// Вычисляет TCD для заданной позы.
    /// \param pose          предполагаемая поза T_cam_model
    /// \param prep          результат предобработки текущего кадра
    /// \param visibleRatio  доля видимых рёбер [0..1]
    double score(const SE3& pose,
                 const ImagePreprocessor::Result& prep,
                 float* visibleRatio = nullptr) const;

    /// Многомасштабный поиск: сначала на уменьшенном изображении, потом уточнение
    //OptResult pyramidSearch(const ImagePreprocessor::Result& prep,
    //                    const CameraIntrinsics& K,
    //                    const Vec3d& searchCenter,
    //                    double tRange) const;

    /// Генерирует начальные гипотезы позы (равномерно по пространству
    /// трансляции + дискретизация ориентаций Икосаэдр/SO3-решётка).
    /// Используется при холодном старте (первый кадр).
    std::vector<SE3> generateInitialHypotheses(const Vec3d& searchCenter,
                                               double tRange,
                                               int    nTranslations,
                                               int    nRotations) const;

private:
    CameraIntrinsics K_;
    const Model3D&   model_;
    PipelineConfig   cfg_;
    ModelRenderer    renderer_;

    // Выборка точек вдоль полилинии (равномерно по длине)
    static std::vector<cv::Point2i>
    samplePolyline(const std::vector<cv::Point2f>& poly, int step);
};

} // namespace pe
