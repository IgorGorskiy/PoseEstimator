#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>

namespace pe {

/// Предобработка входного кадра:
///  1. Усреднение шума (Gaussian / bilateral)
///  2. CLAHE (адаптивная нормализация яркости) – помогает при сварочных бликах
///  3. Canny edge detection
///  4. Distance transform (DT) для быстрого подсчёта Chamfer distance
class ImagePreprocessor {
public:
    explicit ImagePreprocessor(const PipelineConfig& cfg);

    struct Result {
        cv::Mat edges;       // бинарное изображение рёбер (CV_8UC1)
        cv::Mat dt;          // distance transform от рёбер (CV_32F)
        cv::Mat dt_trunc;    // усечённый DT (CV_32F, max = cfg.chamfer_trunc)
    };

    Result process(const cv::Mat& frame) const;

    /// Статический метод: только для визуализации/отладки
    static cv::Mat visualize(const cv::Mat& frame, const Result& res);

private:
    PipelineConfig cfg_;
};

} // namespace pe
