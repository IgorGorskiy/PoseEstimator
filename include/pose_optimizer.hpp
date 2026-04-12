#pragma once
#include "types.hpp"
#include "contour_matcher.hpp"
#include "model_renderer.hpp"

namespace pe {

/// Итеративная оптимизация позы методом Nelder–Mead (Downhill Simplex)
/// в пространстве PoseVec = [tx, ty, tz, rx, ry, rz].
///
/// Преимущества Nelder–Mead для нашей задачи:
///  - не требует градиентов (функция TCD негладкая)
///  - хорошо работает для 6 DOF при малом числе итераций
///  - лёгкий в реализации без сторонних библиотек
///
/// После Nelder–Mead опционально выполняется локальный поиск
/// методом Powell (чередование 1D-минимизаций по осям).

class PoseOptimizer {
public:
    PoseOptimizer(const CameraIntrinsics& K,
                  const Model3D& model,
                  const PipelineConfig& cfg);

    struct OptResult {
        SE3    pose;
        double score;
        int    iterations;
        bool   converged;
    };

    /// Уточняет позу начиная с initPose.
    OptResult optimize(const SE3& initPose,
                       const ImagePreprocessor::Result& prep) const;

    /// Полный поиск: генерирует гипотезы, отбирает topK лучших,
    /// оптимизирует каждую, возвращает лучшую.
    OptResult search(const ImagePreprocessor::Result& prep,
                     const Vec3d& searchCenter,
                     double tRange,
                     int nHypotheses,
                     int topK = 5) const;

private:
    PipelineConfig   cfg_;
    ContourMatcher   matcher_;

    using ScoreFn = std::function<double(const PoseVec&)>;

    // Nelder–Mead по 6D
    void nelderMead(OptResult& result, const PoseVec& x0, const ScoreFn& f,
        double alpha = 1.0, double gamma = 2.0, double rho = 0.5, double sigma = 0.5) const;

    // Начальный симплекс вокруг x0
    std::vector<PoseVec> buildSimplex(const PoseVec& x0) const;
};

} // namespace pe
