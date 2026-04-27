#include "pipeline.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <chrono>

namespace pe {

Pipeline::Pipeline(const std::string& stepPath,
                   CameraIntrinsics& K,
                   const PipelineConfig& cfg)
    : K_(K)
    , cfg_(cfg)
    , model_(loadStep(stepPath))
    , preprocessor_(cfg)
    , renderer_(K, model_)
    , matcher_(K, model_, cfg)
    , optimizer_(K, model_, cfg)
    , filter_(cfg)
    , trackLossThreshold_(cfg.chamfer_trunc * 0.4)
{
    // Оценка начального расстояния по размеру модели
    Vec3d span = model_.bbox_max - model_.bbox_min;
    double modelSize = span.norm();
    coldSearchRange_ = modelSize * 1.5;
    std::cout << "[Pipeline] Model size: " << modelSize << " m\n";
    std::cout << "[Pipeline] Search range: ±" << coldSearchRange_ << " m\n";
}

void Pipeline::reset() {
    mode_      = Mode::COLD_START;
    lostCount_ = 0;
    initialPose_.reset();
    std::cout << "[Pipeline] Reset to COLD_START\n";
}

void Pipeline::setInitialPose(const SE3& pose, bool isExact) {
    initialPose_      = pose;
    initialPoseExact_ = isExact;
    std::cout << "[Pipeline] Initial pose set. isExact=" << isExact << "\n";
    auto t = pose.translation();
    Eigen::AngleAxisd aa(pose.rotation());
    std::printf("  T=[%.4f %.4f %.4f]  R=%.2f deg @ [%.3f %.3f %.3f]\n",
        t.x(), t.y(), t.z(),
        aa.angle() * 180.0 / M_PI,
        aa.axis().x(), aa.axis().y(), aa.axis().z());
}

cv::Mat Pipeline::renderPoseDebug(const cv::Mat& background,
                                   const SE3& pose,
                                   const std::string& label) const
{
    cv::Mat vis;
    if (background.empty()) {
        vis = cv::Mat::zeros(K_.height, K_.width, CV_8UC3);
    } else {
        if (background.channels() == 1)
            cv::cvtColor(background, vis, cv::COLOR_GRAY2BGR);
        else
            vis = background.clone();
    }

    // Рендер рёбер модели
    cv::Mat contour;
    renderer_.render(pose, contour);

    // Наложение жёлтым (отличается от зелёного трекинга)
    cv::Mat yellow = cv::Mat::zeros(vis.size(), vis.type());
    yellow.setTo(cv::Scalar(0, 0, 255), contour);
    cv::addWeighted(vis, 0.7, yellow, 1, 0, vis);
    // Подпись
    auto t = pose.translation();
    Eigen::AngleAxisd aa(pose.rotation());
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "%s  T=[%.3f %.3f %.3f]m  R=%.1fdeg",
        label.c_str(),
        t.x(), t.y(), t.z(),
        aa.angle() * 180.0 / M_PI);
    cv::putText(vis, buf, {10, 40},
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);

    // Подсчёт видимых рёбер
    float vr = 0;
    auto prep = preprocessor_.process(background);
    auto score = matcher_.score(pose, prep, &vr);
    std::cout << "DEBUG POSE SCORE = " << score << "\n";
    char buf2[64];
    std::snprintf(buf2, sizeof(buf2), "VisibleEdges=%.1f%%", vr * 100.f);
    cv::putText(vis, buf2, {10, 70},
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 200, 255), 2);
    return vis;
}

// ── Холодный старт: глобальный поиск ─────────────────────────────────────

PoseEstimate Pipeline::coldStart(const ImagePreprocessor::Result& prep, double ts)
{
    PoseEstimate est;
    est.timestamp = ts;

    // ── Если задана начальная поза ────────────────────────────────────────
    if (initialPose_.has_value()) {
        SE3 initP = initialPose_.value();

        if (initialPoseExact_) {
            // Точная поза — сразу переходим в трекинг
            float vr;
            est.score = matcher_.score(initP, prep, &vr);
            est.pose  = initP;
            est.valid = true;
            filter_.reset(initP, ts);
            mode_      = Mode::TRACKING;
            lostCount_ = 0;
            initialPose_.reset();
            std::cout << "[Pipeline] Exact initial pose accepted."
                      << " score=" << est.score << " vr=" << vr << "\n";
            lastEstimate_ = est;
            return est;
        }

        // Приблизительная поза — локальная оптимизация вокруг неё
        std::cout << "[Pipeline] Refining approximate initial pose...\n";

        // Несколько кандидатов: базовая поза + вариации
        PoseVec base = se3ToPoseVec(initP);
        std::vector<SE3> candidates;
        candidates.push_back(initP);

        // ВАРИАЦИИ ИДУТ НАХУЙ, Я ЗАЕБАЛСЯ ЖДАТЬ ПОКА ЭТО ГОВНО ИСПОЛНЯЕТСЯ!!!!
        // Вариации ±5 см по трансляции и ±10° по вращению
        // for (int i = 0; i < 3; ++i) {
        //     for (double dv : {-0.05, 0.05}) {
        //         PoseVec v = base; v(i) += dv;
        //         candidates.push_back(poseVecToSE3(v));
        //     }
        // }
        // for (int i = 3; i < 6; ++i) {
        //     for (double dv : {-0.17, 0.17}) {  // ~10 deg
        //         PoseVec v = base; v(i) += dv;
        //         candidates.push_back(poseVecToSE3(v));
        //     }
        // }

        PipelineConfig cfg_local = cfg_;
        cfg_local.init_step_t   = 0.05;
        cfg_local.init_step_r   = 0.02;
        cfg_local.max_iterations = 500;
        PoseOptimizer opt_local(K_, model_, cfg_local);

        PoseOptimizer::OptResult best;
        best.score = 1e18;
        for (const auto& c : candidates) {
            auto res = opt_local.optimize(c, prep);
            if (res.score < best.score) best = res;
        }

        std::cout << "[Pipeline] Refined score=" << best.score
                  << " (was approx)\n";

        float vr;
        matcher_.score(best.pose, prep, &vr);

        // После уточнения начальная поза израсходована
        initialPose_.reset();

        if (best.score < trackLossThreshold_ * 2.0) {
            filter_.reset(best.pose, ts);
            est.pose  = best.pose;
            est.score = best.score;
            est.valid = true;
            mode_      = Mode::TRACKING;
            lostCount_ = 0;
            std::cout << "[Pipeline] Track acquired from initial pose!"
                      << " score=" << best.score << " vr=" << vr << "\n";
        } else {
            est.valid = false;
            std::cout << "[Pipeline] Initial pose refinement failed."
                      << " score=" << best.score
                      << " threshold=" << trackLossThreshold_*2.0 << "\n";
            // Падаем в глобальный поиск ниже
        }
        lastEstimate_ = est;
        if (est.valid) return est;
    }

    // ── Глобальный поиск (если нет начальной позы или уточнение не помогло) ─
    Vec3d span  = model_.bbox_max - model_.bbox_min;
    double dist = span.maxCoeff() / (2.0 * std::tan(0.4));
    Vec3d  center = {0, 0, dist};

    auto res = optimizer_.search(prep, center,
                                 coldSearchRange_,
                                 coldSearchHyp_, 5);

    est.score = res.score;
    float vr;
    matcher_.score(res.pose, prep, &vr);
    std::cout << "[DIAG] coldSearch score=" << res.score
              << " threshold=" << trackLossThreshold_ * 2.0
              << " visRatio=" << vr << "\n";
    std::cout << "[DIAG] searchCenter=" << center.transpose()
              << " range=" << coldSearchRange_ << "\n";
    std::cout << "[DIAG] edges pixel count="
              << cv::countNonZero(prep.edges) << "\n";

    if (res.score < trackLossThreshold_ * 2.0) {
        filter_.reset(res.pose, ts);
        est.pose  = res.pose;
        est.valid = true;
        mode_     = Mode::TRACKING;
        lostCount_ = 0;
        std::cout << "[Pipeline] Track acquired! Score=" << res.score << "\n";
    } else {
        est.valid = false;
        std::cout << "[Pipeline] Track NOT acquired. Score=" << res.score << "\n";
    }
    lastEstimate_ = est;
    return est;
}

// ── Режим слежения ────────────────────────────────────────────────────────

PoseEstimate Pipeline::track(const ImagePreprocessor::Result& prep, double ts)
{
    // Горячий старт оптимизатора от предсказания фильтра
    SE3 predicted = filter_.predictedPose();

    // Локальная оптимизация (малый диапазон)
    PipelineConfig localCfg = cfg_;
    //localCfg.init_step_t = 0.02;
    //localCfg.init_step_r = 0.01;

    static PoseOptimizer localOpt(K_, model_, localCfg);
    auto res = localOpt.optimize(predicted, prep);

    float vr;

    PoseEstimate est;
    est.timestamp = ts;
    est.score     = res.score;

    // НА ГПУ vr НЕ СЧИТАЕТСЯ
    bool trackOk = (res.score < trackLossThreshold_);// &&
                   //(vr >= cfg_.min_visible_ratio);

    if (trackOk) {
        SE3 filtered = filter_.update(res.pose, res.score, ts);
        est.pose  = cfg_.use_kalman ? filtered : res.pose;
        auto poseV = se3ToPoseVec(est.pose);
        std::cout << "[Pipeline] Track pose: [ " << poseV[0];
        for (int i = 1; i < 6; i++)
            std::cout << "; " << poseV[i];
        std::cout << " ]\n";
        est.valid = true;
        lostCount_ = 0;
    } else {
        // Пропуск плохого кадра: используем предсказание фильтра
        filter_.predict(ts);
        est.pose  = filter_.predictedPose();
        est.valid = false;
        ++lostCount_;
        std::cout << "[Pipeline] Track degraded (score=" << res.score
                  << " vr=" << vr << " lost=" << lostCount_ << ")\n";
        if (lostCount_ >= LOST_FRAMES_BEFORE_RESET) {
            std::cout << "[Pipeline] Track lost! Returning to COLD_START\n";
            mode_ = Mode::COLD_START;
        }
    }
    PoseVec estV = se3ToPoseVec(est.pose);
    PoseVec lastestV = se3ToPoseVec(lastEstimate_.pose);
    auto shift = estV - lastestV;
    std::cout << "[pipenline] End of processing frame. Pose shift: x=" << shift[0] << " y=" << shift[1] << " z=" << shift[2]
        << " a=" << shift[3] << " b=" << shift[4] << " c=" << shift[5] << "\n";
    lastEstimate_ = est;
    getchar();
    return est;
}

// ── Обработка одного кадра ────────────────────────────────────────────────

PoseEstimate Pipeline::processFrame(const cv::Mat& frame, double timestamp)
{
    auto prep = preprocessor_.process(frame);

    if (mode_ == Mode::COLD_START)
        return coldStart(prep, timestamp);
    else
        return track(prep, timestamp);
}

// ── Отладочная визуализация ───────────────────────────────────────────────

cv::Mat Pipeline::debugRender(const cv::Mat& frame,
                               const PoseEstimate& est) const
{
    cv::Mat vis;
    if (frame.channels() == 1)
        cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
    else
        vis = frame.clone();

    if (!est.valid) {
        cv::putText(vis, "NO TRACK", {10, 40},
                    cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    cv::Scalar(0,0,255), 2);
        return vis;
    }

    // Рендер контуров модели
    cv::Mat contour;
    renderer_.render(est.pose, contour);

    // Наложение зелёным
    cv::Mat green = cv::Mat::zeros(vis.size(), vis.type());
    green.setTo(cv::Scalar(0,255,0), contour);
    cv::addWeighted(vis, 1.0, green, 0.6, 0, vis);

    // Текст
    char buf[128];
    std::snprintf(buf, sizeof(buf), "Score=%.2f", est.score);
    cv::putText(vis, buf, {10, 40},
                cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(0,255,0), 2);

    // Оси координат модели
    Vec3d origin = model_.centroid;
    double axisLen = (model_.bbox_max - model_.bbox_min).norm() * 0.15;
    std::vector<Vec3d> axisEnds = {
        origin + est.pose.rotation().inverse() * Vec3d(axisLen,0,0),
        origin + est.pose.rotation().inverse() * Vec3d(0,axisLen,0),
        origin + est.pose.rotation().inverse() * Vec3d(0,0,axisLen)
    };

    // ... (проекция осей остаётся на усмотрение реализации cv::projectPoints)

    return vis;
}

// ── Видеоцикл ─────────────────────────────────────────────────────────────

void Pipeline::run(const std::string& source,
                   FrameCallback callback,
                   bool showWindow)
{
    cv::VideoCapture cap;

    // Пробуем открыть как устройство (цифра) или файл
    bool opened = false;
    if (source.size() == 1 && std::isdigit(source[0])) {
        opened = cap.open(source[0] - '0');
    } else {
        opened = cap.open(source);
    }

    if (!opened) {
        throw std::runtime_error("Cannot open video source: " + source);
    }

    std::cout << "[Pipeline] Video opened: " << source << "\n";
    std::cout << "[Pipeline] Press 'q' to quit, 'r' to reset tracker\n";

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    cv::Mat frame;
    while (true) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "[Pipeline] End of stream\n";
            break;
        }

        auto now  = Clock::now();
        double ts = std::chrono::duration<double>(now - t0).count();

        auto est  = processFrame(frame, ts);
        auto dbg  = debugRender(frame, est);

        if (callback) callback(est, dbg, mode_);

        if (showWindow) {
            cv::imshow("Pose Estimator", dbg);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
            if (key == 'r') { reset(); std::cout << "[User] Manual reset\n"; }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::cout << "[FRAME] time = " << ms(t0, t1) << " ms\n";
    }
    cap.release();
    if (showWindow) cv::destroyAllWindows();
}

} // namespace pe
