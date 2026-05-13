#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include "pipeline.hpp"
#include "vk.hpp"
#include "types.hpp"
#include "step_loader.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp >
#include <iostream>
#include <stdexcept>
#include <string>
#include <optional>
#include <vector>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <cstring>
#define GLM_ENABLE_EXPERIMENTAL
// #define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <numbers>
#include <windows.h>

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

std::optional<pe::SE3> loadInitialPoseVulcan(const YAML::Node& pnode) {
    if (!pnode || !pnode["initial_pose_vulcan"]) std::cout << "loadInitialPoseVulcan ERROR\n";
    const auto& ip = pnode["initial_pose_vulcan"];
    if (!ip["enabled"] || !ip["enabled"].as<bool>()) std::cout << "loadInitialPoseVulcan not enabled\n";
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
    Eigen::Matrix3d sw2cv; sw2cv << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    pe::SE3 T = pe::SE3::Identity();
    T.linear() = sw2cv * Rcw;
    T.translation() = sw2cv * tcw;

    std::cout << "[InitPose] SW pos(m): " << tx << " " << ty << " " << tz << "\n";
    std::cout << "[InitPose] SW rot(deg): "
        << ip["rotation_deg"][0].as<double>() << " "
        << ip["rotation_deg"][1].as<double>() << " "
        << ip["rotation_deg"][2].as<double>() << "\n";
    auto t = T.translation();
    Eigen::AngleAxisd aa(T.rotation());
    std::printf("[InitPose] T_cam_model: T=[%.3f %.3f %.3f]  R=%.1f deg\n",
        t.x(), t.y(), t.z(), aa.angle() * 180.0 / M_PI);
    if (t.z() <= 0)
        std::cout << "[InitPose] WARNING: Z<=0, объект за камерой!\n"
        << "  Попробуй поменять знак rotation_deg или порядок углов.\n";
    return T;
}

std::vector<double> loadInitialPoseVulcanVector(const YAML::Node& pnode) {
    if (!pnode || !pnode["initial_pose_vulcan"]) std::cout << "loadInitialPoseVulcan ERROR\n";
    const auto& ip = pnode["initial_pose_vulcan"];
    if (!ip["enabled"] || !ip["enabled"].as<bool>()) std::cout << "loadInitialPoseVulcan not enabled\n";
    std::vector<double> vec(6);
    vec[0] = ip["position_mm"][0].as<double>() * 0.001;
    vec[1] = ip["position_mm"][1].as<double>() * 0.001;
    vec[2] = ip["position_mm"][2].as<double>() * 0.001;
    vec[3] = ip["rotation_deg"][0].as<double>();
    vec[4] = ip["rotation_deg"][1].as<double>();
    vec[5] = ip["rotation_deg"][2].as<double>();
    return vec;
}

void poseToSwParams(const pe::SE3& pose,
    Eigen::Vector3d& pos_mm,
    Eigen::Vector3d& rot_deg)
{
    Eigen::Matrix3d sw2cv; sw2cv << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    Eigen::Matrix3d Rcw_sw = sw2cv * pose.rotation();
    pe::Vec3d           tcw_sw = sw2cv * pose.translation();
    Eigen::Matrix3d R = Rcw_sw.transpose();   // R_world_cam = R_sw
    pe::Vec3d           t_sw = -R * tcw_sw;
    pos_mm = t_sw * 1000.0;

    // R = Rx(a) * Ry(b) * Rz(c)  — intrinsic XYZ
    // R(0,2) =  sin(b)
    // R(1,2) = -sin(a)*cos(b)
    // R(2,2) =  cos(a)*cos(b)
    // R(0,1) = -cos(b)*sin(c)
    // R(0,0) =  cos(b)*cos(c)
    double sin_b = std::max(-1.0, std::min(1.0, R(0, 2)));
    double ry = std::asin(sin_b);
    double cos_b = std::cos(ry);

    double rx, rz;
    if (std::abs(cos_b) > 1e-6) {
        rx = std::atan2(-R(1, 2) / cos_b, R(2, 2) / cos_b);
        rz = std::atan2(-R(0, 1) / cos_b, R(0, 0) / cos_b);
    }
    else {
        // Gimbal lock: ry = ±90°
        // При ry=+90: R = [[0, sin(a-c), cos(a-c)], ...]
        rx = std::atan2(R(1, 0), R(1, 1));
        rz = 0;
    }
    rot_deg = Eigen::Vector3d(rx, ry, rz) * 180.0 / M_PI;
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

void vec3d_push(std::vector<float>& vec, const pe::Vec3d& v) {
    vec.push_back(v.x());
    vec.push_back(v.y());
    vec.push_back(v.z());
}

struct CameraPose {
    double x, y, z;       // позиция камеры в СК модели
    double rx, ry, rz;    // углы Euler XYZ intrinsic (радианы)
};

CameraPose poseToCamera(const pe::SE3& pose) {
    // T_cam_model → T_model_cam
    const Eigen::Matrix3d R = pose.rotation();
    const Eigen::Vector3d t = pose.translation();

    // Позиция камеры в СК модели: -R^T * t
    Eigen::Matrix3d Rt = R.transpose();
    Eigen::Vector3d pos = -Rt * t;

    // Euler XYZ intrinsic из R^T:
    // R^T = Rx(rx) * Ry(ry) * Rz(rz)
    // R^T = | cy*cz        -cy*sz         sy     |
    //       | sx*sy*cz+cx*sz  -sx*sy*sz+cx*cz  -sx*cy |
    //       | -cx*sy*cz+sx*sz   cx*sy*sz+sx*cz   cx*cy  |
    // Из R^T[0,2] = sin(ry) → ry
    // Из R^T[1,2] / R^T[2,2] → rx
    // Из R^T[0,1] / R^T[0,0] → rz

    double ry = std::asin(std::max(-1.0, std::min(1.0, Rt(0, 2))));
    double cos_ry = std::cos(ry);

    double rx, rz;
    if (std::abs(cos_ry) > 1e-6) {
        rx = std::atan2(-Rt(1, 2) / cos_ry, Rt(2, 2) / cos_ry);
        rz = std::atan2(-Rt(0, 1) / cos_ry, Rt(0, 0) / cos_ry);
    }
    else {
        // Gimbal lock: ry = ±90°
        rx = std::atan2(Rt(2, 1), Rt(1, 1));
        rz = 0.0;
    }

    return { pos.x(), pos.y(), pos.z(), rx, ry, rz };
}

int main(int argc, char* argv[]) {
    VK::set_resolution(1920, 1080);
    VK::init();
    VK::set_line_width(2.0f);
    pe::Model3D model = pe::loadStep("robot2.STEP");
    std::vector<float> faces, edges;

    for (const auto& face : model.faces) {
        vec3d_push(faces, face.v0);
        vec3d_push(faces, face.v1);
        vec3d_push(faces, face.v2);
    }

    for (const auto& edge : model.edges) {
        const auto& pts = edge.pts;
        for (size_t i = 1; i < pts.size(); i++) {
            vec3d_push(edges, pts[i]);
            vec3d_push(edges, pts[i - 1]);
        }
    }

    VK::upload_faces(faces);
    VK::upload_edges(edges);
    cv::Mat img;

    setlocale(LC_ALL, "");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    std::string argvv[4];
    if (argc < 4) {
        std::cout << "Usage: pose_estimator <config.yaml> <model.step> <video>\n";
        argvv[0] = "Release/pose_estimator_gpu.exe"; // бесполезно. Наверно. Я не уверен.
        argvv[1] = "config/cameranew2.yaml";
        argvv[2] = "resources/robot2.STEP";
        argvv[3] = "resources/robot22.mp4";
        //argvv[0] = "Release/pose_estimator_gpu.exe"; // бесполезно. Наверно. Я не уверен.
        //argvv[1] = "config/cameranew2leg.yaml";
        //argvv[2] = "resources/legpart.STEP";
        //argvv[3] = "resources/vidleg.mp4";

        //argvv = argvv;
    }

    std::cout << "argc: " << argc << "\n";
for (int i = 0; i < argc; ++i)
    std::cout << "argvv[" << i << "] = [" << argvv[i] << "]\n";


cv::VideoCapture cap;
std::string src = argvv[3];
bool opened = (src.size()==1 && std::isdigit(src[0]))
        ? cap.open(src[0]-'0') : cap.open(src);
std::cout << "cap.open result: " << opened << "\n";
std::cout << "isOpened: " << cap.isOpened() << "\n";

    YAML::Node yaml;
    try { yaml = YAML::LoadFile(argvv[1]); }
    catch (const std::exception& e) {
        std::cerr << "Cannot load config: " << e.what() << "\n"; return 1;
    }

    auto K        = loadCamera(yaml["camera"]);
    auto cfg      = loadConfig(yaml["pipeline"]);
    auto initPose = loadInitialPose(yaml["pipeline"]);
    Eigen::Vector3d pose_t;
    Eigen::Vector3d pose_r;
    poseToSwParams(initPose.value(), pose_t, pose_r);
    auto VP = loadInitialPoseVulcanVector(yaml["pipeline"]);
    VP[0] = -pose_t[0]/1000;
    VP[1] = pose_t[1] / 1000;
    VP[2] = -pose_t[2] / 1000;
    VP[3] = pose_r[0] + 3;
    VP[4] = -pose_r[2];
    VP[5] = pose_r[1];

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = std::chrono::high_resolution_clock::now();
    img = VK::draw(
        VP[0], VP[1], VP[2],
        VP[3], VP[4], VP[5]
    );
    img = VK::draw(
        VP[0], VP[1], VP[2],
        VP[3], VP[4], VP[5]
    );
    img = VK::draw(
        VP[0], VP[1], VP[2],
        VP[3], VP[4], VP[5]
    );
    img = VK::draw(
        VP[0], VP[1], VP[2],
        VP[3], VP[4], VP[5]
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    cv::Mat dst2 = img;

    std::cout << "[VULCAN] time =" << ms(t0, t1) << " ms\n";

    cv::imwrite("test.png", img);
    cv::imwrite("test2.png", dst2);

    std::cout << "Camera: " << K.width << "x" << K.height
              << "  fx=" << K.fx << "\n";
    try {
        pe::Pipeline pipeline(argvv[2], K, cfg);

        // ── Статический рендер начальной позы ────────────────────────────
        if (initPose.has_value()) {
            std::cout << "\n=== Static render of SolidWorks initial pose ===\n"
                      << "Yellow contour = модель в позиции SW.\n"
                      << "Если совпадает со сборкой — координаты верны.\n"
                      << "Нажмите любую клавишу для продолжения, ESC для выхода.\n\n";

            cv::Mat firstFrame;
            if (opened) { cap.read(firstFrame); cap.release(); }

            cv::Mat render = pipeline.renderPoseDebug(
                img, initPose.value(), "SW initial pose");
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


        pipeline.run(argvv[3],
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
