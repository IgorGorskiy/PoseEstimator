#include "contour_matcher.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <conio.h>

namespace pe {

ContourMatcher::ContourMatcher(const CameraIntrinsics& K,
                               const Model3D& model,
                               const PipelineConfig& cfg)
    : K_(K), model_(model), cfg_(cfg), renderer_(K, model) {}

// ── Вычисление Truncated Chamfer Distance ────────────────────────────────

double ContourMatcher::score(const SE3& pose,
                             const ImagePreprocessor::Result& prep,
                             float* visibleRatio) const
{
    const cv::Mat& dt = prep.dt_trunc;   // CV_32F, усечённый DT
    const int W = dt.cols, H = dt.rows;
    //if (H != K_.height || W != K_.width) {
    //    std::cout << "H = " << H << "; K_.height = " << K_.height << "; W = " << W << "; K_.width = " << K_.width << "\n";
    //    throw("IMAGE SCALE INCORRECT");
    //    return 1000;
    //}
    double sumDist = 0.0;
    int    count = 0;
    bool gpu = true;
    if (gpu) {
        cv::Mat contours;
        renderer_.render(pose, contours, 1);
        cv::Point2i p;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                p.x = x;
                p.y = y;
                if (contours.at<uchar>(p) == 255)
                    sumDist += static_cast<double>(dt.at<float>(p));
                ++count;
            }
        }
    }
    else {
        float vr = 0.f;
        auto polys = renderer_.project(pose, &vr);

        if (visibleRatio) *visibleRatio = vr;

        if (polys.empty()) return static_cast<double>(cfg_.chamfer_trunc);
        const int step = 2;                  // шаг выборки в пикселях
        for (const auto& poly : polys) {
            auto pts = samplePolyline(poly, step);
            for (const auto& p : pts) {
                if (p.x < 0 || p.x >= W || p.y < 0 || p.y >= H) continue;
                sumDist += static_cast<double>(dt.at<float>(p));
                ++count;
            }
        }
    }
    if (count == 0) return static_cast<double>(cfg_.chamfer_trunc);
    return sumDist / count;
}

// ── Выборка точек вдоль полилинии ────────────────────────────────────────

std::vector<cv::Point2i>
ContourMatcher::samplePolyline(const std::vector<cv::Point2f>& poly, int step)
{
    std::vector<cv::Point2i> result;
    if (poly.size() < 2) return result;

    double accum = 0.0;
    cv::Point2f prev = poly[0];
    result.push_back({static_cast<int>(prev.x), static_cast<int>(prev.y)});

    for (size_t i = 1; i < poly.size(); ++i) {
        cv::Point2f cur = poly[i];
        double dx = cur.x - prev.x, dy = cur.y - prev.y;
        double len = std::sqrt(dx*dx + dy*dy);

        double t = (step - accum);
        while (t <= len) {
            double ratio = t / len;
            cv::Point2i sample = {
                static_cast<int>(prev.x + ratio * dx),
                static_cast<int>(prev.y + ratio * dy)
            };
            result.push_back(sample);
            t += step;
        }
        accum = std::fmod(accum + len, static_cast<double>(step));
        prev = cur;
    }
    return result;
}

// ── Генерация начальных гипотез ───────────────────────────────────────────
//
// Для ориентаций используем вершины икосаэдра (20 вершин) как начальную
// дискретизацию SO(3). При увеличении nRotations применяем дополнительные
// повороты вокруг каждой из трёх осей.

namespace {

// 12 вершин икосаэдра (нормированные)
static const std::vector<Eigen::Vector3d> ICOSAHEDRON_VERTICES = {
    { 0,  1,  1.618}, { 0, -1,  1.618}, { 0,  1, -1.618}, { 0, -1, -1.618},
    { 1,  1.618, 0}, {-1,  1.618, 0}, { 1, -1.618, 0}, {-1, -1.618, 0},
    { 1.618, 0,  1}, {-1.618, 0,  1}, { 1.618, 0, -1}, {-1.618, 0, -1}
};

std::vector<Eigen::Matrix3d> generateRotations(int n) {
    std::vector<Eigen::Matrix3d> rots;
    rots.push_back(Eigen::Matrix3d::Identity());

    // Икосаэдр: направления взгляда
    for (const auto& v : ICOSAHEDRON_VERTICES) {
        Eigen::Vector3d dir = v.normalized();
        // Строим систему из dir
        Eigen::Vector3d up = std::abs(dir.z()) < 0.9 ?
            Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitY();
        Eigen::Vector3d right = dir.cross(up).normalized();
        up = right.cross(dir).normalized();

        // Вращения вокруг dir на шаги 2π/n
        int nAz = std::max(1, n / 12);
        for (int k = 0; k < nAz; ++k) {
            double angle = 2*M_PI * k / nAz;
            Eigen::AngleAxisd roll(angle, dir);
            Eigen::Matrix3d R;
            R.col(0) = roll * right;
            R.col(1) = roll * up;
            R.col(2) = dir;
            rots.push_back(R);
        }
    }
    return rots;
}

} // anonymous namespace

std::vector<SE3>
ContourMatcher::generateInitialHypotheses(const Vec3d& center,
                                          double tRange,
                                          int nT,
                                          int nR) const
{
    auto rots = generateRotations(nR);

    std::vector<SE3> hypotheses;

    // Равномерная сетка трансляций вместо случайных
    int gridN = std::max(1, (int)std::cbrt(nT));
    double step = (gridN > 1) ? (2.0 * tRange / (gridN - 1)) : 0.0;

    for (int ix = 0; ix < gridN; ++ix)
    for (int iy = 0; iy < gridN; ++iy)
    for (int iz = 0; iz < gridN; ++iz) {
        Vec3d t = center + Vec3d(
            -tRange + ix * step,
            -tRange + iy * step,
            -tRange + iz * step);

        // Z должен быть положительным (объект перед камерой)
        if (t.z() <= 0.05) continue;

        for (const auto& R : rots) {
            SE3 T = SE3::Identity();
            T.linear()      = R;
            T.translation() = t;
            hypotheses.push_back(T);
        }
    }
    return hypotheses;
}

} // namespace pe
