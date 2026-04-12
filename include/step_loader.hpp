#pragma once
#include "types.hpp"
#include <string>

namespace pe {

/// Загружает STEP-файл через OpenCASCADE, извлекает B-Rep рёбра,
/// тесселирует их в полилинии и возвращает Model3D.
///
/// Рёбра классифицируются:
///  - "резкие" (угол между смежными гранями > sharpAngleDeg) → всегда рисуются
///  - "плавные" (касательный переход)                        → silhouette-рёбра
///
/// \param path          путь к .step / .stp файлу
/// \param deflection    линейное отклонение тесселяции (м), влияет на кол-во точек
/// \param sharpAngleDeg угол в градусах для классификации рёбер
Model3D loadStep(const std::string& path,
                 double deflection    = 30.0,
                 double sharpAngleDeg = 30.0);

} // namespace pe
