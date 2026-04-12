#include "pose_filter.hpp"
#include <iostream>
#include <cmath>

namespace pe {

PoseFilter::PoseFilter(const PipelineConfig& cfg) : cfg_(cfg) {
    x_.setZero();
    P_.setIdentity();
    P_ *= 1.0;  // большая начальная неопределённость

    // Шум процесса Q
    Q_.setZero();
    for (int i = 0; i < 3; ++i) Q_(i,   i)   = cfg_.proc_noise_t;
    for (int i = 3; i < 6; ++i) Q_(i,   i)   = cfg_.proc_noise_r;
    for (int i = 6; i < 9; ++i) Q_(i,   i)   = cfg_.proc_noise_t * 10;
    for (int i = 9; i < 12; ++i) Q_(i,  i)   = cfg_.proc_noise_r * 10;

    // Шум измерения R
    R_.setZero();
    for (int i = 0; i < 3; ++i) R_(i, i) = cfg_.meas_noise_t;
    for (int i = 3; i < 6; ++i) R_(i, i) = cfg_.meas_noise_r;
}

PoseFilter::StateMat PoseFilter::makeF(double dt) {
    StateMat F = StateMat::Identity();
    // x_new = x_old + v*dt
    for (int i = 0; i < 6; ++i)
        F(i, i+6) = dt;
    return F;
}

PoseFilter::HMat PoseFilter::makeH() {
    HMat H = HMat::Zero();
    for (int i = 0; i < 6; ++i) H(i, i) = 1.0;
    return H;
}

void PoseFilter::reset(const SE3& initPose, double timestamp) {
    x_.setZero();
    x_.head<6>() = se3ToPoseVec(initPose);
    P_.setIdentity();
    P_ *= 0.1;
    prevTime_     = timestamp;
    initialized_  = true;
    lastMahal_    = 0;
    std::cout << "[KalmanFilter] Reset at t=" << timestamp << "\n";
}

void PoseFilter::predict(double timestamp) {
    if (!initialized_) return;
    double dt = timestamp - prevTime_;
    if (dt <= 0) return;

    StateMat F = makeF(dt);
    x_ = F * x_;
    P_ = F * P_ * F.transpose() + Q_ * dt;
    prevTime_ = timestamp;
}

SE3 PoseFilter::update(const SE3& meas, double score, double timestamp) {
    predict(timestamp);

    HMat H = makeH();

    MeasVec z = se3ToPoseVec(meas);

    // Адаптация R: если score высокий (плохое совпадение) → увеличиваем шум
    double scoreRatio = std::min(score / cfg_.chamfer_trunc, 1.0);
    MeasMat R_adapt = R_ * (1.0 + 10.0 * scoreRatio * scoreRatio);

    // Инновация
    MeasVec innov = z - H * x_;

    // Нормализация угловой части инновации в [-π, π]
    for (int i = 3; i < 6; ++i) {
        while (innov(i) >  M_PI) innov(i) -= 2*M_PI;
        while (innov(i) < -M_PI) innov(i) += 2*M_PI;
    }

    // Ковариация инновации
    MeasMat S = H * P_ * H.transpose() + R_adapt;

    // Расстояние Махаланобиса для детектирования выбросов
    lastMahal_ = std::sqrt(innov.transpose() * S.llt().solve(innov));

    // Порог выброса: если Mahalanobis > 5σ – игнорируем измерение
    static constexpr double MAHAL_THRESH = 5.0;
    if (lastMahal_ > MAHAL_THRESH && initialized_) {
        std::cout << "[KalmanFilter] Outlier rejected (Mahal=" << lastMahal_ << ")\n";
        return poseVecToSE3(x_.head<6>());
    }

    // Gain Калмана
    Eigen::Matrix<double, NS, NM> K =
        P_ * H.transpose() * S.inverse();

    // Обновление состояния
    x_ = x_ + K * innov;
    P_ = (StateMat::Identity() - K * H) * P_;

    // Симметризация P (численная стабильность)
    P_ = 0.5 * (P_ + P_.transpose());

    prevTime_ = timestamp;
    return poseVecToSE3(x_.head<6>());
}

SE3 PoseFilter::predictedPose() const {
    return poseVecToSE3(x_.head<6>());
}

} // namespace pe
