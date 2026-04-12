#pragma once
#include "types.hpp"
#include "step_loader.hpp"
#include "model_renderer.hpp"
#include "image_preprocessor.hpp"
#include "contour_matcher.hpp"
#include "pose_optimizer.hpp"
#include "pose_filter.hpp"

#include <functional>
#include <optional>
#include <string>

namespace pe {

/// Главный пайплайн: обрабатывает видеопоток и возвращает
/// отфильтрованную оценку позы на каждый кадр.
///
/// Режимы работы:
///  - COLD_START   – глобальный поиск (первый кадр или потеря трека)
///  - TRACKING     – локальная оптимизация вблизи предсказанной позы
///
/// Потеря трека определяется по:
///  - score > threshold
///  - Mahalanobis distance фильтра > 5σ
///  - visibleRatio < min_visible_ratio

class Pipeline {
public:
    enum class Mode { COLD_START, TRACKING };

    /// Колбэк вызывается на каждый кадр с результатом и отладочными данными
    using FrameCallback = std::function<void(
        const PoseEstimate& pose,
        const cv::Mat& debugFrame,
        Mode mode
    )>;

    Pipeline(const std::string& stepPath,
             const CameraIntrinsics& K,
             const PipelineConfig& cfg);

    /// Запуск обработки видео из файла или камеры
    /// \param source  путь к файлу или "0","1",... для USB-камеры
    void run(const std::string& source,
             FrameCallback callback = nullptr,
             bool showWindow = true);

    /// Обработка одного кадра (для встраивания в внешний цикл)
    PoseEstimate processFrame(const cv::Mat& frame, double timestamp);

    /// Переключение в COLD_START (например, при смене детали)
    void reset();

    /// Установить начальную позу (из SolidWorks или другого источника).
    /// Поза используется как отправная точка оптимизатора вместо глобального поиска.
    /// \param pose         T_cam_model в метрах
    /// \param isExact      false = поза приблизительная, нужна локальная оптимизация
    ///                     true  = поза точная, сразу переходим в TRACKING
    void setInitialPose(const SE3& pose, bool isExact = false);

    /// Отрисовать рендер для произвольной позы на произвольном фоне.
    /// Используется для проверки правильности интерпретации координат.
    cv::Mat renderPoseDebug(const cv::Mat& background, const SE3& pose,
                            const std::string& label = "") const;

    /// Получить текущий отладочный рендер
    cv::Mat debugRender(const cv::Mat& frame,
                        const PoseEstimate& estimate) const;

private:
    CameraIntrinsics   K_;
    PipelineConfig     cfg_;
    Model3D            model_;
    ImagePreprocessor  preprocessor_;
    ModelRenderer      renderer_;
    ContourMatcher     matcher_;
    PoseOptimizer      optimizer_;
    PoseFilter         filter_;

    Mode               mode_{Mode::COLD_START};
    PoseEstimate       lastEstimate_;
    int                lostCount_{0};
    static constexpr int LOST_FRAMES_BEFORE_RESET = 10;

    // Начальная поза (из SolidWorks или другого источника)
    std::optional<SE3> initialPose_;
    bool               initialPoseExact_{false};

    // Параметры холодного поиска
    double coldSearchRange_{0.5};     // ±0.5 м диапазон поиска трансляции
    int    coldSearchHyp_{500};       // число гипотез

    // Порог score для определения "потери трека"
    double trackLossThreshold_;

    PoseEstimate coldStart(const ImagePreprocessor::Result& prep, double ts);
    PoseEstimate track(const ImagePreprocessor::Result& prep, double ts);
};

} // namespace pe
