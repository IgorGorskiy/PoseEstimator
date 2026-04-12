#include "pipeline.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <optional>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

pe::CameraIntrinsics loadCamera(const YAML::Node& n) {
    pe::CameraIntrinsics K;
    K.fx = n["fx"].as<double>(); K.fy = n["fy"].as<double>();
    K.cx = n["cx"].as<double>(); K.cy = n["cy"].as<double>();
    K.width = n["width"].as<int>(); K.height = n["height"].as<int>();
    if (n["dist_coeffs"]) {
        auto d = n["dist_coeffs"];
        for (int i = 0; i < 5 && i < (int)d.size(); ++i)
            K.distCoeffs[i] = d[i].as<double>();
    }
    return K;
}

pe::PipelineConfig loadConfig(const YAML::Node& n) {
    pe::PipelineConfig cfg;
    auto g = [&](const char* key, auto& val) {
        if (n[key]) val = n[key].as<std::decay_t<decltype(val)>>();
    };
    g("canny_low", cfg.canny_low); g("canny_high", cfg.canny_high);
    g("blur_ksize", cfg.blur_ksize); g("use_clahe", cfg.use_clahe);
    g("chamfer_trunc", cfg.chamfer_trunc); g("dt_dilate", cfg.dt_dilate);
    g("init_step_t", cfg.init_step_t); g("init_step_r", cfg.init_step_r);
    g("max_iterations", cfg.max_iterations);
    g("convergence_tol", cfg.convergence_tol);
    g("use_kalman", cfg.use_kalman);
    g("proc_noise_t", cfg.proc_noise_t); g("proc_noise_r", cfg.proc_noise_r);
    g("meas_noise_t", cfg.meas_noise_t); g("meas_noise_r", cfg.meas_noise_r);
    g("min_visible_ratio", cfg.min_visible_ratio);
    return cfg;
}

// ── Загрузка начальной позы из SolidWorks ────────────────────────────────
//
// СК SolidWorks: X вправо, Y вверх, Z из экрана
// СК камеры OpenCV: X вправо, Y вниз, Z вглубь
//
// Алгоритм:
//   R_world_cam_sw = Rx(rx) * Ry(ry) * Rz(rz)  [intrinsic XYZ]
//   T_cam_world_sw = inv(T_world_cam_sw)
//   sw2cv = diag(1,-1,-1)
//   T_cam_model = sw2cv * T_cam_world_sw

std::optional<pe::SE3> loadInitialPose(const YAML::Node& pnode) {
    if (!pnode || !pnode["initial_pose"]) return std::nullopt;
    const auto& ip = pnode["initial_pose"];
    if (!ip["enabled"] || !ip["enabled"].as<bool>()) return std::nullopt;

    double tx = ip["position_mm"][0].as<double>() * 0.001;
    double ty = ip["position_mm"][1].as<double>() * 0.001;
    double tz = ip["position_mm"][2].as<double>() * 0.001;
    double rx = ip["rotation_deg"][0].as<double>() * M_PI / 180.0;
    double ry = ip["rotation_deg"][1].as<double>() * M_PI / 180.0;
    double rz = ip["rotation_deg"][2].as<double>() * M_PI / 180.0;

    // Intrinsic XYZ: применяем Rx, потом Ry, потом Rz
    Eigen::Matrix3d R_sw =
        (Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ())).toRotationMatrix();

    // T_cam_world в СК SolidWorks
    Eigen::Matrix3d Rcw = R_sw.transpose();
    Eigen::Vector3d tcw = -Rcw * Eigen::Vector3d(tx, ty, tz);

    // Перевод SW -> OpenCV: инвертируем Y и Z
    Eigen::Matrix3d sw2cv; sw2cv << 1,0,0, 0,-1,0, 0,0,-1;
    pe::SE3 T = pe::SE3::Identity();
    T.linear()      = sw2cv * Rcw;
    T.translation() = sw2cv * tcw;

    std::cout << "[InitPose] SW pos(m): " << tx <<" "<< ty <<" "<< tz << "\n";
    std::cout << "[InitPose] SW rot(deg): "
              << ip["rotation_deg"][0].as<double>() << " "
              << ip["rotation_deg"][1].as<double>() << " "
              << ip["rotation_deg"][2].as<double>() << "\n";
    auto t = T.translation();
    Eigen::AngleAxisd aa(T.rotation());
    std::printf("[InitPose] T_cam_model: T=[%.3f %.3f %.3f]  R=%.1f deg\n",
        t.x(), t.y(), t.z(), aa.angle()*180.0/M_PI);
    if (t.z() <= 0)
        std::cout << "[InitPose] WARNING: Z<=0, объект за камерой!\n"
                  << "  Попробуй поменять знак rotation_deg или порядок углов.\n";
    return T;
}

void printPose(const pe::PoseEstimate& est) {
    if (!est.valid) { std::cout << "  [INVALID]\n"; return; }
    auto t = est.pose.translation();
    Eigen::AngleAxisd aa(est.pose.rotation());
    std::printf("  T=[%.4f %.4f %.4f]  R=%.2f deg @ [%.3f %.3f %.3f]  score=%.2f\n",
        t.x(), t.y(), t.z(),
        aa.angle()*180.0/M_PI,
        aa.axis().x(), aa.axis().y(), aa.axis().z(), est.score);
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "");
    char ** argvv = new char*[4];
    if (argc < 4) {
        std::cout << "Usage: pose_estimator <config.yaml> <model.step> <video>\n";
        argvv[0] = "Release/pose_estimator_gpu.exe"; // бесполезно. Наверно. Я не уверен.
        argvv[1] = "config/cameranew2.yaml";
        argvv[2] = "resources/robot2.STEP";
        argvv[3] = "resources/robot22.mp4";
        argv = argvv;
    }

    std::cout << "argc: " << argc << "\n";
for (int i = 0; i < argc; ++i)
    std::cout << "argv[" << i << "] = [" << argv[i] << "]\n";


cv::VideoCapture cap;
std::string src = argv[3];
bool opened = (src.size()==1 && std::isdigit(src[0]))
        ? cap.open(src[0]-'0') : cap.open(src);
std::cout << "cap.open result: " << opened << "\n";
std::cout << "isOpened: " << cap.isOpened() << "\n";

    YAML::Node yaml;
    try { yaml = YAML::LoadFile(argv[1]); }
    catch (const std::exception& e) {
        std::cerr << "Cannot load config: " << e.what() << "\n"; return 1;
    }

    auto K        = loadCamera(yaml["camera"]);
    auto cfg      = loadConfig(yaml["pipeline"]);
    auto initPose = loadInitialPose(yaml["pipeline"]);

    std::cout << "Camera: " << K.width << "x" << K.height
              << "  fx=" << K.fx << "\n";
    try {
        pe::Pipeline pipeline(argv[2], K, cfg);

        // ── Статический рендер начальной позы ────────────────────────────
        if (initPose.has_value()) {
            std::cout << "\n=== Static render of SolidWorks initial pose ===\n"
                      << "Yellow contour = модель в позиции SW.\n"
                      << "Если совпадает со сборкой — координаты верны.\n"
                      << "Нажмите любую клавишу для продолжения, ESC для выхода.\n\n";

            // проверка
//             std::string src2 = argv[3];
// bool opened2 = cap.open(src2, cv::CAP_MSMF);
// std::cout << "CAP_MSMF opened: " << opened2 << "\n";
// if (!opened2) {
//     opened2 = cap.open(src2, cv::CAP_ANY);
//     std::cout << "CAP_ANY opened: " << opened2 << "\n";
// }


// // Попробуем явно через FFMPEG бэкенд
// cv::VideoCapture cap2(src, cv::CAP_FFMPEG);
// std::cout << "CAP_FFMPEG opened: " << cap2.isOpened() << "\n";

// // Попробуем через Media Foundation
// cv::VideoCapture cap3(src, cv::CAP_MSMF);
// std::cout << "CAP_MSMF opened: " << cap3.isOpened() << "\n";


            cv::Mat firstFrame;
            if (opened) { cap.read(firstFrame); cap.release(); }

            cv::Mat render = pipeline.renderPoseDebug(
                firstFrame, initPose.value(), "SW initial pose");
            cv::imshow("Initial Pose Check (any key = continue, ESC = exit)", render);
            cv::imwrite("debug_initial_pose.png", render);
            std::cout << "Saved: debug_initial_pose.png\n\n";
            std::cout << "Как читать результат:\n"
                      << "  Контур совпадает со сборкой -> OK, координаты верны\n"
                      << "  Контур отражён              -> поменяй знак у одного из rotation_deg\n"
                      << "  Контур смещён               -> поменяй знак у position_mm\n"
                      << "  Контур повёрнут не так      -> поменяй порядок rotation_deg\n"
                      << "    Варианты порядка: [rx,ry,rz] [rz,ry,rx] [ry,rx,rz] [rx,rz,ry]\n\n";

            int key = cv::waitKey(0);
            cv::destroyAllWindows();
            if (key == 27) return 0;

            pipeline.setInitialPose(initPose.value(), false);
        }

        // ── Трекинг ───────────────────────────────────────────────────────
        int fi = 0;

        // Временная диагностика - удали после проверки
// cv::VideoCapture test("C:/amyfiles/projects/poseEstimation/pose_estimator/build/Release/robot22.mp4");
// if (test.isOpened())
//     std::cout << "Backend: " << test.getBackendName() << "\n";
// else
//     std::cout << "Backend: not opened\n";
// std::cout << "Opened: " << test.isOpened() << "\n";
// if (test.isOpened()) {
//     std::cout << "FPS: " << test.get(cv::CAP_PROP_FPS) << "\n";
//     std::cout << "Width: " << test.get(cv::CAP_PROP_FRAME_WIDTH) << "\n";
// }
// test.release();


        pipeline.run(argv[3],
            [&](const pe::PoseEstimate& est, const cv::Mat&,
                pe::Pipeline::Mode mode) {
                if (fi % 10 == 0) {
                    std::cout << "Frame " << fi << "  "
                              << (mode==pe::Pipeline::Mode::TRACKING
                                  ? "TRACKING" : "COLD_START")
                              << "  t=" << est.timestamp << "s\n";
                    printPose(est);
                }
                ++fi;
            }, true);
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n"; return 1;
    }
    return 0;
}
