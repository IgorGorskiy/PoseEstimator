#include "cuda_renderer.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Geometry>

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <limits>

__global__ void kernelTest() {
    // пустое ядро
}

// ── Макрос проверки CUDA ──────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            throw std::runtime_error(                                       \
                std::string("[CUDA] ") + cudaGetErrorString(_e) +          \
                " at " __FILE__ ":" + std::to_string(__LINE__));           \
        }                                                                   \
    } while(0)

namespace pe {
    

// ── Структуры для передачи на GPU ─────────────────────────────────────────
// (не используем Eigen внутри .cu — только plain structs)

struct GpuVec3 { float x, y, z; };
struct GpuTri  { GpuVec3 v0, v1, v2; };
struct GpuSeg  { GpuVec3 a, b; };           // отрезок ребра

struct GpuCamera {
    float fx, fy, cx, cy;
    int   width, height;
    // dist coeffs (k1,k2,p1,p2,k3)
    float k1, k2, p1, p2, k3;
};

struct GpuPose {
    float R[9];   // row-major 3x3
    float t[3];
};

// ── Device-функции ────────────────────────────────────────────────────────

__device__ __forceinline__
GpuVec3 transformPoint(const GpuPose& pose, const GpuVec3& p) {
    return {
        pose.R[0]*p.x + pose.R[1]*p.y + pose.R[2]*p.z + pose.t[0],
        pose.R[3]*p.x + pose.R[4]*p.y + pose.R[5]*p.z + pose.t[1],
        pose.R[6]*p.x + pose.R[7]*p.y + pose.R[8]*p.z + pose.t[2]
    };
}

__device__ __forceinline__
void projectPoint(const GpuCamera& cam, const GpuVec3& pc,
                  float& u, float& v)
{
    float x = pc.x / pc.z, y = pc.y / pc.z;
    float r2 = x*x + y*y;
    float r4 = r2*r2, r6 = r2*r4;
    float rad = 1.f + cam.k1*r2 + cam.k2*r4 + cam.k3*r6;
    float xd  = x*rad + 2.f*cam.p1*x*y + cam.p2*(r2+2.f*x*x);
    float yd  = y*rad + cam.p1*(r2+2.f*y*y) + 2.f*cam.p2*x*y;
    u = cam.fx*xd + cam.cx;
    v = cam.fy*yd + cam.cy;
}

// ── Kernel 1: растеризация треугольников → Z-буфер ────────────────────────
//
// Каждый поток = один треугольник.
// Для каждого пикселя в AABB треугольника:
//   - проверяем барицентрические координаты
//   - интерполируем глубину
//   - атомарно записываем минимум в Z-буфер (int32 = глубина * 1e6)

__global__ void kernelRasterizeFaces(
    const GpuTri*    tris,
    int              numTris,
    const GpuCamera  cam,
    const GpuPose    pose,
    int*             zbuf)      // [height * width], int32, единицы = глубина*1e6
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTris) return;

    const GpuTri& tri = tris[tid];

    // Трансформируем вершины в СК камеры
    GpuVec3 c0 = transformPoint(pose, tri.v0);
    GpuVec3 c1 = transformPoint(pose, tri.v1);
    GpuVec3 c2 = transformPoint(pose, tri.v2);

    constexpr float NEAR = 0.001f;
    if (c0.z < NEAR && c1.z < NEAR && c2.z < NEAR) return;
    // Простое отсечение: зажимаем вершины за near-плоскостью
    if (c0.z < NEAR) c0.z = NEAR;
    if (c1.z < NEAR) c1.z = NEAR;
    if (c2.z < NEAR) c2.z = NEAR;

    // Проецируем
    float x0,y0, x1,y1, x2,y2;
    projectPoint(cam, c0, x0, y0);
    projectPoint(cam, c1, x1, y1);
    projectPoint(cam, c2, x2, y2);

    // AABB
    int minX = max(0,   (int)floorf(fminf(x0, fminf(x1, x2))));
    int maxX = min(cam.width-1,  (int)ceilf(fmaxf(x0, fmaxf(x1, x2))));
    int minY = max(0,   (int)floorf(fminf(y0, fminf(y1, y2))));
    int maxY = min(cam.height-1, (int)ceilf(fmaxf(y0, fmaxf(y1, y2))));

    // Площадь треугольника (знаковая — для барицентрических координат)
    float denom = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2);
    if (fabsf(denom) < 1e-6f) return;
    float invDenom = 1.f / denom;

    for (int py = minY; py <= maxY; ++py) {
        for (int px = minX; px <= maxX; ++px) {
            float fx = px + 0.5f, fy = py + 0.5f;

            // Барицентрические координаты
            float w0 = ((y1-y2)*(fx-x2) + (x2-x1)*(fy-y2)) * invDenom;
            float w1 = ((y2-y0)*(fx-x2) + (x0-x2)*(fy-y2)) * invDenom;
            float w2 = 1.f - w0 - w1;

            if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;

            // Интерполируем глубину
            float z = w0*c0.z + w1*c1.z + w2*c2.z;

            // Атомарный минимум: конвертируем в int (глубина * 1e6)
            int zi = (int)(z * 1000000.f);
            atomicMin(&zbuf[py * cam.width + px], zi);
        }
    }
}

// ── Kernel 2: растеризация отрезков рёбер с Z-тестом ─────────────────────
//
// Каждый поток = один отрезок ребра.
// Рисуем отрезок по алгоритму Брезенхема на GPU,
// для каждого пикселя проверяем Z-буфер.

__global__ void kernelRasterizeEdges(
    const GpuSeg* segs,
    int             numSegs,
    const GpuCamera cam,
    const GpuPose   pose,
    const int* zbuf,
    unsigned char* mask,       // [height * width]
    int             thickness)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numSegs) return;
    const GpuSeg& seg = segs[tid];
    constexpr float NEAR = 0.001f;
    constexpr float BIAS = 0.003f;  // 3 мм Z-bias
    GpuVec3 ca = transformPoint(pose, seg.a);
    GpuVec3 cb = transformPoint(pose, seg.b);
    // Отсекаем по near-плоскости
    if (ca.z < NEAR && cb.z < NEAR) return;
    if (ca.z < NEAR) ca.z = NEAR;
    if (cb.z < NEAR) cb.z = NEAR;
    float u0, v0, u1, v1;
    projectPoint(cam, ca, u0, v0);
    projectPoint(cam, cb, u1, v1);
    // Брезенхем в пиксельных координатах
    int x0 = (int)roundf(u0), y0 = (int)roundf(v0);
    int x1 = (int)roundf(u1), y1 = (int)roundf(v1);
    int dx = abs(x1 - x0), dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    int steps = dx + dy + 1;
    for (int s = 0; s < steps; ++s) {
        // Параметр t вдоль отрезка для интерполяции глубины
        float totalSteps = float(dx + dy);
        float t = (totalSteps > 0.f) ? float(s) / totalSteps : 0.f;
        float z = (1.f - t) * ca.z + t * cb.z - BIAS;
        // Рисуем пиксель с учётом толщины
        int half = thickness / 2;
        for (int dy2 = -half; dy2 <= half; ++dy2) {
            for (int dx2 = -half; dx2 <= half; ++dx2) {
                int px = x0 + dx2, py = y0 + dy2;
                if (px < 0 || px >= cam.width ||
                    py < 0 || py >= cam.height) continue;
                // Z-тест
                int zi = (int)(z * 1000000.f);
                int buf = zbuf[py * cam.width + px];
                // buf == INT_MAX → поверхности нет → ребро всегда видимо
                if (buf != 2147483647 && zi > buf) continue;
                mask[py * cam.width + px] = 255;
            }
        }
        // Шаг Брезенхема
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

// ── Impl ─────────────────────────────────────────────────────────────────

struct CudaRenderer::Impl {
    CameraIntrinsics K;

    // Device buffers — геометрия
    GpuTri* d_tris{nullptr};   int numTris{0};
    GpuSeg* d_segs{nullptr};   int numSegs{0};

    // Device buffers — рендер
    int*           d_zbuf{nullptr};
    unsigned char* d_mask{nullptr};

    // CPU результат
    std::vector<unsigned char> h_mask;

    explicit Impl(const CameraIntrinsics& k) : K(k) {
        allocFrameBuffers();
    }

    ~Impl() {
        freeGeometry();
        freeFrameBuffers();
    }

    void allocFrameBuffers() {
        int n = K.width * K.height;
        CUDA_CHECK(cudaMalloc(&d_zbuf, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_mask, n * sizeof(unsigned char)));
        h_mask.resize(n);
    }

    void freeFrameBuffers() {
        if (d_zbuf) { cudaFree(d_zbuf); d_zbuf = nullptr; }
        if (d_mask) { cudaFree(d_mask); d_mask = nullptr; }
    }

    void freeGeometry() {
        if (d_tris) { cudaFree(d_tris); d_tris = nullptr; numTris = 0; }
        if (d_segs) { cudaFree(d_segs); d_segs = nullptr; numSegs = 0; }
    }

    void uploadModel(const Model3D& model) {
        freeGeometry();

        // ── Треугольники ──────────────────────────────────────────────────
        std::vector<GpuTri> tris;
        tris.reserve(model.faces.size());
        for (const auto& f : model.faces) {
            GpuTri t;
            t.v0 = {float(f.v0.x()), float(f.v0.y()), float(f.v0.z())};
            t.v1 = {float(f.v1.x()), float(f.v1.y()), float(f.v1.z())};
            t.v2 = {float(f.v2.x()), float(f.v2.y()), float(f.v2.z())};
            tris.push_back(t);
        }
        numTris = int(tris.size());
        if (numTris > 0) {
            CUDA_CHECK(cudaMalloc(&d_tris, numTris * sizeof(GpuTri)));
            CUDA_CHECK(cudaMemcpy(d_tris, tris.data(),
                                  numTris * sizeof(GpuTri),
                                  cudaMemcpyHostToDevice));
        }

        // ── Отрезки рёбер ─────────────────────────────────────────────────
        std::vector<GpuSeg> segs;
        for (const auto& edge : model.edges) {
            for (size_t i = 0; i+1 < edge.pts.size(); ++i) {
                GpuSeg s;
                s.a = {float(edge.pts[i  ].x()),
                       float(edge.pts[i  ].y()),
                       float(edge.pts[i  ].z())};
                s.b = {float(edge.pts[i+1].x()),
                       float(edge.pts[i+1].y()),
                       float(edge.pts[i+1].z())};
                segs.push_back(s);
            }
        }
        numSegs = int(segs.size());
        if (numSegs > 0) {
            CUDA_CHECK(cudaMalloc(&d_segs, numSegs * sizeof(GpuSeg)));
            CUDA_CHECK(cudaMemcpy(d_segs, segs.data(),
                                  numSegs * sizeof(GpuSeg),
                                  cudaMemcpyHostToDevice));
        }

        std::cout << "[CudaRenderer] Uploaded: " << numTris << " tris, "
                  << numSegs << " edge segments\n";
    }

    // Собираем GpuCamera и GpuPose из текущих параметров
    GpuCamera makeCamera() const {
        GpuCamera c;
        c.fx = float(K.fx); c.fy = float(K.fy);
        c.cx = float(K.cx); c.cy = float(K.cy);
        c.width = K.width;  c.height = K.height;
        c.k1 = float(K.distCoeffs[0]); c.k2 = float(K.distCoeffs[1]);
        c.p1 = float(K.distCoeffs[2]); c.p2 = float(K.distCoeffs[3]);
        c.k3 = float(K.distCoeffs[4]);
        return c;
    }

    static GpuPose makePose(const SE3& pose) {
        GpuPose p;
        Eigen::Matrix3f R = pose.rotation().cast<float>();
        Eigen::Vector3f t = pose.translation().cast<float>();
        // Row-major
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                p.R[r*3+c] = R(r,c);
        p.t[0] = t.x(); p.t[1] = t.y(); p.t[2] = t.z();
        return p;
    }

    cv::Mat renderEdgeMask(const SE3& pose, int thickness) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int W = K.width, H = K.height, N = W*H;

        GpuCamera cam  = makeCamera();
        GpuPose   gp   = makePose(pose);

        // Очищаем буферы
        CUDA_CHECK(cudaMemset(d_zbuf, 0x7ff, N * sizeof(int)));   // INT_MAX
        CUDA_CHECK(cudaMemset(d_mask, 0,    N * sizeof(unsigned char)));

        constexpr int BLOCK = 256;
        auto t1 = std::chrono::high_resolution_clock::now();
        // ── Pass 1: треугольники → Z-буфер ────────────────────────────────
        if (numTris > 0) {
    int grid = (numTris + BLOCK-1) / BLOCK;
    kernelRasterizeFaces<<<grid, BLOCK>>>(
        d_tris, numTris, cam, gp, d_zbuf);
    cudaError_t e = cudaGetLastError();
    std::cout << "[CUDA] kernelRasterizeFaces: " << cudaGetErrorString(e) << "\n";
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    std::cout << "[CUDA] after sync: " << cudaGetErrorString(e) << "\n";
}
        auto t2 = std::chrono::high_resolution_clock::now();
        // ── Pass 2: рёбра + Z-тест → маска ────────────────────────────────
        if (numSegs > 0) {
            int grid = (numSegs + BLOCK-1) / BLOCK;
            kernelRasterizeEdges<<<grid, BLOCK>>>(
                d_segs, numSegs, cam, gp, d_zbuf, d_mask, thickness);
            CUDA_CHECK(cudaGetLastError());
        }

        // Синхронизация + readback
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t3 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask,
                              N * sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));
        
        auto t4 = std::chrono::high_resolution_clock::now();

        auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b-a).count();
    };
    std::cout << "[CUDA timing] setup=" << ms(t0,t1)
              << " faces=" << ms(t1,t2)
              << " edges=" << ms(t2,t3)
              << " readback=" << ms(t3,t4) << " ms\n";

        cv::imshow("RENDER", cv::Mat(H, W, CV_8UC1, h_mask.data()));
        return cv::Mat(H, W, CV_8UC1, h_mask.data()).clone();
    }
};

// ── Публичный интерфейс ───────────────────────────────────────────────────

CudaRenderer::CudaRenderer(const CameraIntrinsics& K) {
    int devCount = 0;
    cudaError_t err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess || devCount == 0) {
        std::cerr << "[CudaRenderer] No CUDA device found\n";
        return;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[CudaRenderer] GPU: " << prop.name
              << "  SM: " << prop.multiProcessorCount
              << "  Mem: " << prop.totalGlobalMem/1024/1024 << " MB\n";
    try {
        impl_ = std::make_unique<Impl>(K);
        valid_ = true;
    } catch (const std::exception& e) {
        std::cerr << "[CudaRenderer] Init failed: " << e.what() << "\n";
    }
}

CudaRenderer::~CudaRenderer() = default;

void CudaRenderer::uploadModel(const Model3D& model) {
    if (impl_) impl_->uploadModel(model);
}

void CudaRenderer::setCamera(const CameraIntrinsics& K) {
    if (!impl_) return;
    impl_->K = K;
    impl_->freeFrameBuffers();
    impl_->allocFrameBuffers();
}

cv::Mat CudaRenderer::renderEdgeMask(const SE3& pose, int thickness) {
    if (!impl_) return {};
    return impl_->renderEdgeMask(pose, thickness);
}

} // namespace pe
