#pragma once
#include <array>
#include <vector>
#include <optional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace pe {

// ── Базовые геометрические типы ──────────────────────────────────────────

using Vec2d  = Eigen::Vector2d;
using Vec3d  = Eigen::Vector3d;
using Vec6d  = Eigen::Matrix<double, 6, 1>;
using Mat3d  = Eigen::Matrix3d;
using Mat4d  = Eigen::Matrix4d;
using SE3    = Eigen::Isometry3d;           // аффинное преобразование (R|t)

// Представление позы: ось-угол (aa) + трансляция
// state = [tx, ty, tz, rx, ry, rz]  (ось-угол, угол = |r|)
using PoseVec = Vec6d;

inline SE3 poseVecToSE3(const PoseVec& v) {
    SE3 T = SE3::Identity();
    T.translation() = v.head<3>();
    Eigen::AngleAxisd aa(v.tail<3>().norm(),
                         v.tail<3>().normalized());
    if (v.tail<3>().norm() > 1e-9)
        T.linear() = aa.toRotationMatrix();
    return T;
}

inline PoseVec se3ToPoseVec(const SE3& T) {
    PoseVec v;
    v.head<3>() = T.translation();
    Eigen::AngleAxisd aa(T.rotation());
    v.tail<3>() = aa.axis() * aa.angle();
    return v;
}

// ── Параметры камеры ─────────────────────────────────────────────────────

struct CameraIntrinsics {
    double fx{}, fy{}, cx{}, cy{};
    std::array<double,5> distCoeffs{};   // k1,k2,p1,p2,k3
    int width{}, height{};

    cv::Mat toCvMatrix() const {
        cv::Mat K = (cv::Mat_<double>(3,3) <<
            fx, 0, cx,
             0, fy, cy,
             0,  0,  1);
        return K;
    }
    cv::Mat distCoeffsCv() const {
        return cv::Mat(1,5,CV_64F,
            const_cast<double*>(distCoeffs.data())).clone();
    }
};

// ── 3D модель после загрузки STEP ────────────────────────────────────────

struct Edge3D {
    std::vector<Vec3d> pts;
    bool isSharp{true};
};

// Треугольная грань для Z-буфера
struct Face3D {
    Vec3d v0, v1, v2;
};

struct Model3D {
    std::vector<Edge3D> edges;
    std::vector<Face3D> faces;   // треугольные грани для Z-буфера
    Vec3d               bbox_min;
    Vec3d               bbox_max;
    Vec3d               centroid;
};

// ── Результат оценки позы ─────────────────────────────────────────────────

struct PoseEstimate {
    SE3    pose;          // T_cam_model: модель → камера
    double score{};       // метрика качества (чем меньше, тем лучше)
    bool   valid{false};
    double timestamp{};   // секунды
};

// ── Конфигурация пайплайна ────────────────────────────────────────────────

struct PipelineConfig {
    // Предобработка
    int    canny_low{50};
    int    canny_high{150};
    int    blur_ksize{5};
    bool   use_clahe{true};

    // Сопоставление
    float  chamfer_trunc{30.f};   // порог усечённого расстояния Шамфера (px)
    int    dt_dilate{3};          // дилатация рёбер перед distance transform

    // Оптимизатор
    double init_step_t{0.005};    // начальный шаг по трансляции (м)
    double init_step_r{0.02};     // начальный шаг по углу (рад)
    int    max_iterations{500};
    double convergence_tol{1e-6};

    // Фильтр Калмана
    bool   use_kalman{true};
    double proc_noise_t{1e-4};    // дисперсия шума процесса, трансляция
    double proc_noise_r{1e-3};    // дисперсия шума процесса, вращение
    double meas_noise_t{1e-3};
    double meas_noise_r{1e-2};

    // Видимость
    float  min_visible_ratio{0.15f};   // если <15% рёбер видно – пропуск
};

} // namespace pe
