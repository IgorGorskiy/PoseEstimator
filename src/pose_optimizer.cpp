#include "pose_optimizer.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>

namespace pe {
    extern int firstframeINT = 1;

PoseOptimizer::PoseOptimizer(CameraIntrinsics& K,
                             const Model3D& model,
                             const PipelineConfig& cfg)
    : cfg_(cfg), matcher_(K, model, cfg) {
}

// ── Начальный симплекс ────────────────────────────────────────────────────

std::vector<PoseVec> PoseOptimizer::buildSimplex(const PoseVec& x0) const
{
    // n+1 = 7 вершин в 6D
    auto myX = x0;
    for (int i = 0; i < 6; ++i) {
        double step = (i < 3) ? cfg_.init_step_t : cfg_.init_step_r;
        myX(i) -= step / 6;
    }
    std::vector<PoseVec> simplex(7, myX);
    for (int i = 0; i < 6; ++i) {
        double step = (i < 3) ? cfg_.init_step_t : cfg_.init_step_r;
        simplex[i+1](i) += step;
    }
    return simplex;
}

// ── Nelder–Mead ───────────────────────────────────────────────────────────

void PoseOptimizer::nelderMead(OptResult &result, const PoseVec& x0, const ScoreFn& f,
    double alpha, double gamma, double rho, double sigma) const
{
    auto t0 = std::chrono::high_resolution_clock::now();
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b-a).count();
    };
    //double alpha = 1.0;   // reflection
    //double gamma = 2.0;   // expansion
    //double rho   = 0.5;   // contraction
    //double sigma = 0.5;   // shrink

    const int n = 6;

    // Вершины симплекса и их значения
    auto simplex = buildSimplex(x0);
    std::vector<double> fval(n+1);
    int div = 2;
    for (int i = 0; i <= n; ++i) fval[i] = f(simplex[i], div);

    result.iterations = 0;
    result.converged  = false;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
        auto t2 = std::chrono::high_resolution_clock::now();
        // ── Сортировка (best=0, worst=n) ──────────────────────────────
        std::vector<int> idx(n+1);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&fval](int a, int b){ return fval[a] < fval[b]; });

        std::vector<PoseVec> s2(n+1);
        std::vector<double>  f2(n+1);
        for (int i = 0; i <= n; ++i) { s2[i] = simplex[idx[i]]; f2[i] = fval[idx[i]]; }
        simplex = s2; fval = f2;

        // ── Критерий сходимости ────────────────────────────────────────
        double spread = 0;
        for (int i = 1; i <= n; ++i)
            spread = std::max(spread, (simplex[i] - simplex[0]).norm());
        if (spread < cfg_.convergence_tol) {
            result.converged = true;
            break;
        }
        if (spread < 0.01)
            div = 2;
        else if  (spread < 0.005)
            div = 1;
        else if (spread < 0.003)
            div = 1;
        // ── Центроид (без худшей вершины) ──────────────────────────────
        PoseVec centroid = PoseVec::Zero();
        for (int i = 0; i < n; ++i) centroid += simplex[i];
        centroid /= n;

        // ── Отражение ──────────────────────────────────────────────────
        PoseVec xr = centroid + alpha * (centroid - simplex[n]);
        double  fr = f(xr, div);
        if (fr < fval[0]) {
            // Расширение
            PoseVec xe = centroid + gamma * (xr - centroid);
            double  fe = f(xe, div);
            simplex[n] = (fe < fr) ? xe : xr;
            fval[n]    = (fe < fr) ? fe : fr;
        } else if (fr < fval[n-1]) {
            simplex[n] = xr; fval[n] = fr;
        } else {
            // Сжатие
            bool doShrink = true;
            if (fr < fval[n]) {
                PoseVec xc = centroid + rho * (xr - centroid);
                double  fc = f(xc, div);
                if (fc < fr) { simplex[n] = xc; fval[n] = fc; doShrink = false; }
            } else {
                PoseVec xc = centroid + rho * (simplex[n] - centroid);
                double  fc = f(xc, div);
                if (fc < fval[n]) { simplex[n] = xc; fval[n] = fc; doShrink = false; }
            }
            if (doShrink) {
                // Сжатие всего симплекса к лучшей точке
                for (int i = 1; i <= n; ++i) {
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0]);
                    fval[i]    = f(simplex[i], div);
                }
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        //std::cout << "[nelderMead] iter " << result.iterations << " time =" << ms(t2, t3) << "ms; score = " << fval[0] << "; spread = " << spread << " \n";
        ++result.iterations;
    }
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "[nelderMead] prep time = " << ms(t0,t1) 
              << "; cycle exec time = " << ms(t1,t6) 
              << " ms; iterations = " << result.iterations << "\n";
    result.pose  = poseVecToSE3(simplex[0]);
    result.score = fval[0];
}

std::vector<PoseVec> generateGrid(
    const PoseVec& x0,
    int n1, int n2, int n3,   // количество шагов в обе стороны для первых 3 координат
    int n4, int n5, int n6,   // для последних 3 координат
    double t,                 // шаг для первых 3 координат
    double r                  // шаг для последних 3 координат
) {
    std::vector<PoseVec> grid;

    for (int i = -n1; i <= n1; ++i)
        for (int j = -n2; j <= n2; ++j)
            for (int k = -n3; k <= n3; ++k)
                for (int l = -n4; l <= n4; ++l)
                    for (int m = -n5; m <= n5; ++m)
                        for (int n = -n6; n <= n6; ++n) {
                            PoseVec p;

                            p[0] = x0[0] + i * t;
                            p[1] = x0[1] + j * t;
                            p[2] = x0[2] + k * t;

                            p[3] = x0[3] + l * r;
                            p[4] = x0[4] + m * r;
                            p[5] = x0[5] + n * r;

                            grid.push_back(p);
                        }

    return grid;
}

// ── Интерфейс optimize ────────────────────────────────────────────────────

PoseOptimizer::OptResult
PoseOptimizer::optimize(const SE3& initPose,
                        const ImagePreprocessor::Result& prep) const
{
    firstframeINT = 1;
    PoseVec x0 = se3ToPoseVec(initPose);
    auto scoreFn = [this, &prep](const PoseVec& v, int div) -> double {
        float vr;
        double s = matcher_.score(poseVecToSE3(v), prep, &vr, div);
        // Штраф за почти полную невидимость
        // УДАЛЕНО, ТАК КАК НА ГПУ ВИДИМОСТЬ РЁБЕР НЕ ПОДСЧИТЫВАЕТСЯ
        //if (vr < cfg_.min_visible_ratio)
        //    s += cfg_.chamfer_trunc * (1.0 - vr / cfg_.min_visible_ratio);
        return s;
    };
    bool multithread = false;
    if (!multithread) {
        double alpha = 1.0;   // reflection
        double gamma = 2.0;   // expansion
        double rho = 0.5;   // contraction
        double sigma = 0.5;   // shrink
        OptResult result[7];
        auto simplex = buildSimplex(x0);
        double min = 0;
        auto minresult = result[0];
        auto minPoint = simplex[0];
        /*for (int i = 0; i < 7; ++i) {
            PoseOptimizer::nelderMead(result[i], simplex[i], scoreFn, alpha, gamma, rho, sigma);
            if (result[i].score < minresult.score){
                minresult = result[i];
                minPoint = simplex[i];
            }
        }*/
        //auto t0 = std::chrono::high_resolution_clock::now();
        //auto ms = [](auto a, auto b) {
        //    return std::chrono::duration<double, std::milli>(b - a).count();
        //};
        //int n = 1;
        //std::vector<PoseVec> grid = generateGrid(x0, n, n, n, n, n, n, cfg_.init_step_t, cfg_.init_step_r);
        //double minScore = scoreFn(x0, 10);
        //auto minPose = x0;
        //for (PoseVec curX0 : grid) {
        //    auto curScore = scoreFn(curX0, 10);
        //    if (curScore < minScore) {
        //        minScore = curScore;
        //        minPose = curX0;
        //    }
        //}
        //auto shift = minPose - x0;
        //auto t1 = std::chrono::high_resolution_clock::now();
        //std::cout << "[grid] time = " << ms(t0, t1) << "\n";
        //std::cout << "[grid] pose shift: x=" << shift[0] << " y=" << shift[1] << " z=" << shift[2]
        //    << " a=" << shift[3] << " b=" << shift[4] << " c=" << shift[5] << "\n";
        PoseOptimizer::nelderMead(minresult, x0, scoreFn, alpha, gamma, rho, sigma);
        return minresult;
    }
    std::vector<double> alphaV{0.95, 1.05};
    std::vector<double> gammaV{1.9, 2.1};
    std::vector<double> rhoV{0.47, 0.53};
    std::vector<double> sigmaV{0.47, 0.53};
    std::vector<OptResult> results(16);
    std::vector<std::thread> workers;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {

                    int idx = i * 8 + j * 4 + k * 2 + l;

                    workers.emplace_back(
                        &PoseOptimizer::nelderMead,
                        this,  // <-- ВАЖНО
                        std::ref(results[idx]), // <-- ссылка!
                        std::cref(x0),
                        std::cref(scoreFn),
                        alphaV[i],
                        gammaV[j],
                        rhoV[k],
                        sigmaV[l]
                    );
                }
            }
        }
    }
    for (auto& t : workers) {
        if (t.joinable()) { // Safety check to prevent errors
            t.join();
        }
        else {
            std::cout << "ERROR: tread did not join" << std::endl;
        }
    }
    double minscore = results[0].score;
    int minID = 0;
    OptResult finalRes = results[0];
    std::cout << "[optimize] Optimisation finished. Scores: ";
    for (int i = 0; i < 16; i++) {
        auto res = results[i];
        std::cout << res.score << " ";
        if (res.score < minscore) {
            minscore = res.score;
            minID = i;
            finalRes = res;
        }
    }
    std::cout << "\n" << "[optimize] Best score: " << finalRes.score << std::endl;
    return finalRes;
    //return nelderMead(x0, scoreFn);
}

// ── Полный поиск (гипотезы → отбор → уточнение) ──────────────────────────

PoseOptimizer::OptResult
PoseOptimizer::search(const ImagePreprocessor::Result& prep,
                      const Vec3d& searchCenter,
                      double tRange,
                      int nHypotheses,
                      int topK) const
{
    std::cout << "[Optimizer] Generating " << nHypotheses << " hypotheses...\n";
    auto hyps = matcher_.generateInitialHypotheses(
        searchCenter, tRange, nHypotheses / 100, 100);

    // Быстрая оценка всех гипотез
    std::vector<std::pair<double, SE3>> scored;
    scored.reserve(hyps.size());
    for (const auto& h : hyps) {
        double s = matcher_.score(h, prep);
        scored.push_back({s, h});
    }
    std::partial_sort(scored.begin(),
                      scored.begin() + std::min(topK, (int)scored.size()),
                      scored.end(),
                      [](auto& a, auto& b){ return a.first < b.first; });

    // Оптимизация topK лучших гипотез
    OptResult best;
    best.score = 1e18;

    int k = std::min(topK, (int)scored.size());
    for (int i = 0; i < k; ++i) {
        std::cout << "[Optimizer] Refining hypothesis " << i+1 << "/" << k
                  << "  (initial score=" << scored[i].first << ")\n";
        auto res = optimize(scored[i].second, prep);
        if (res.score < best.score) best = res;
    }

    std::cout << "[Optimizer] Best score: " << best.score
              << "  converged=" << best.converged << "\n";
    return best;
}

} // namespace pe
