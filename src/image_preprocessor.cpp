#include "image_preprocessor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>

namespace pe {

ImagePreprocessor::ImagePreprocessor(const PipelineConfig& cfg) : cfg_(cfg) {
    if (cfg_.blur_ksize % 2 == 0)
        throw std::invalid_argument("blur_ksize must be odd");
}

ImagePreprocessor::Result ImagePreprocessor::process(const cv::Mat& frame) const
{
    // ── 1. Приведение к оттенкам серого ──────────────────────────────────
    cv::Mat gray;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame.clone();

    // ── 2. CLAHE (адаптивное выравнивание гистограммы) ───────────────────
    // Особенно важно при сварке: яркая дуга + тёмный металл
    if (cfg_.use_clahe) {
        auto clahe = cv::createCLAHE(3.0, cv::Size(8,8));
        clahe->apply(gray, gray);
    }

    // ── 3. Размытие ───────────────────────────────────────────────────────
    // Bilateral filter сохраняет резкие границы лучше Gaussian,
    // но дороже по времени; используем Gaussian как баланс.
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred,
                     cv::Size(cfg_.blur_ksize, cfg_.blur_ksize), 0);

    // ── 4. Canny ─────────────────────────────────────────────────────────
    cv::Mat edges;
    cv::Canny(blurred, edges, cfg_.canny_low, cfg_.canny_high);

    // Лёгкая дилатация: "утолщаем" рёбра, чтобы DT был менее чувствителен
    // к субпиксельным сдвигам проекции
    if (cfg_.dt_dilate > 0) {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(cfg_.dt_dilate, cfg_.dt_dilate));
        cv::dilate(edges, edges, kernel);
    }

    // ── 5. Distance transform ─────────────────────────────────────────────
    // Входные данные для DT: инвертируем (0 = ребро, 255 = фон)
    cv::Mat inv;
    cv::bitwise_not(edges, inv);

    cv::Mat dt;
    cv::distanceTransform(inv, dt, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    // ── 6. Усечённый DT (Truncated Chamfer) ──────────────────────────────
    cv::Mat dt_trunc;
    dt.copyTo(dt_trunc);
    cv::threshold(dt_trunc, dt_trunc,
                  static_cast<double>(cfg_.chamfer_trunc),
                  static_cast<double>(cfg_.chamfer_trunc),
                  cv::THRESH_TRUNC);
    //cv::imshow("edges", edges);
    return {edges, dt, dt_trunc};
}

cv::Mat ImagePreprocessor::visualize(const cv::Mat& frame, const Result& res)
{
    cv::Mat vis;
    if (frame.channels() == 1)
        cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
    else
        vis = frame.clone();

    // Наложение рёбер синим цветом
    cv::Mat edgeMask;
    cv::cvtColor(res.edges, edgeMask, cv::COLOR_GRAY2BGR);
    cv::addWeighted(vis, 0.7, edgeMask, 0.3, 0, vis);
    return vis;
}

} // namespace pe
