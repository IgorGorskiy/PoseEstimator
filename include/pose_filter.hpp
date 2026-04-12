#pragma once
#include "types.hpp"
#include <Eigen/Dense>

namespace pe {

/// Расширенный фильтр Калмана (EKF) для фильтрации позы в видеопотоке.
///
/// Модель:
///   Состояние: x = [tx, ty, tz, rx, ry, rz,   ── поза (ось-угол)
///                   vtx, vty, vtz, vrx, vry, vrz]  ── скорости
///   Размер состояния: 12
///   Размер измерения: 6
///
/// Модель движения: равномерное прямолинейное (Constant Velocity).
///   x_k = F * x_{k-1} + w,   w ~ N(0, Q)
///   z_k = H * x_k + v,       v ~ N(0, R)
///
/// Почему полезна фильтрация:
///   - Робот движется плавно → кинематическая связанность кадров
///   - Подавление шума оптимизации (locкal minima)
///   - Предсказание позы для следующего кадра → быстрый старт оптимизации
///   - Детектирование аномальных скачков позы (вес измерения снижается)
///
/// Примечание: для SE(3) строго нужен EKF или UKF на группе Ли,
/// здесь используется линеаризованное приближение в ось-угол координатах,
/// что достаточно при малых угловых скоростях (типично для манипулятора).

class PoseFilter {
public:
    explicit PoseFilter(const PipelineConfig& cfg);

    /// Сброс фильтра с начальной позой
    void reset(const SE3& initPose, double timestamp);

    /// Шаг предсказания (без измерения) – вызывается при пропущенном кадре
    void predict(double timestamp);

    /// Шаг обновления с новым измерением позы
    /// \param meas      измеренная поза от оптимизатора
    /// \param score     метрика качества (TCD); используется для адаптивного R
    /// \param timestamp время измерения (секунды)
    /// \return          отфильтрованная поза
    SE3 update(const SE3& meas, double score, double timestamp);

    /// Предсказание позы на следующий кадр (для горячего старта оптимизатора)
    SE3 predictedPose() const;

    bool isInitialized() const { return initialized_; }

    /// Маханаланобис-расстояние последнего измерения (для детекции выбросов)
    double lastMahalanobisDistance() const { return lastMahal_; }

private:
    static constexpr int NS = 12;  // размер состояния
    static constexpr int NM = 6;   // размер измерения

    using StateVec = Eigen::Matrix<double, NS, 1>;
    using StateMat = Eigen::Matrix<double, NS, NS>;
    using MeasVec  = Eigen::Matrix<double, NM, 1>;
    using MeasMat  = Eigen::Matrix<double, NM, NM>;
    using HMat     = Eigen::Matrix<double, NM, NS>;

    PipelineConfig cfg_;

    StateVec x_;   // состояние
    StateMat P_;   // ковариация состояния
    StateMat Q_;   // шум процесса
    MeasMat  R_;   // шум измерения

    double prevTime_{0};
    bool   initialized_{false};
    double lastMahal_{0};

    // Матрица измерения H: берём только первые 6 компонент состояния
    static HMat makeH();

    // Построить матрицу перехода F для интервала dt
    static StateMat makeF(double dt);
};

} // namespace pe
