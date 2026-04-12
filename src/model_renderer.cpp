#include "model_renderer.hpp"
#include "cuda_renderer.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <limits>
#include <cmath>

namespace pe {
    extern int firstframeINT;

ModelRenderer::ModelRenderer(const CameraIntrinsics& K, const Model3D& model)
    : K_(K), model_(model)
{
    cuda_ = std::make_unique<CudaRenderer>(K);
    if (cuda_->isValid()) {
        cuda_->uploadModel(model);
        std::cout << "[ModelRenderer] Using CUDA renderer\n";
    } else {
        cuda_.reset();
        std::cout << "[ModelRenderer] CUDA unavailable, using CPU fallback\n";
    }
}

ModelRenderer::~ModelRenderer() = default;

// ── CPU fallback: проекция ─────────────────────────────────────────────────

cv::Point2f ModelRenderer::projectPoint(const Vec3d& pc) const {
    double x = pc.x()/pc.z(), y = pc.y()/pc.z();
    const auto& d = K_.distCoeffs;
    double r2=x*x+y*y, r4=r2*r2, r6=r2*r4;
    double rad=1.+d[0]*r2+d[1]*r4+d[4]*r6;
    double xd=x*rad+2*d[2]*x*y+d[3]*(r2+2*x*x);
    double yd=y*rad+d[2]*(r2+2*y*y)+2*d[3]*x*y;
    return {float(K_.fx*xd+K_.cx), float(K_.fy*yd+K_.cy)};
}

static bool clipNear(Vec3d& A, Vec3d& B, double n) {
    bool a=A.z()>=n, b=B.z()>=n;
    if (!a&&!b) return false; if (a&&b) return true;
    Vec3d M=A+(n-A.z())/(B.z()-A.z())*(B-A);
    if (!a) A=M; else B=M; return true;
}

// ── render() ──────────────────────────────────────────────────────────────

void ModelRenderer::render(const SE3& pose, cv::Mat& out, int thickness) const {
    if (cuda_ && cuda_->isValid()) {
        out = cuda_->renderEdgeMask(pose, thickness);
        return;
    }
    // CPU fallback
    out = cv::Mat::zeros(K_.height, K_.width, CV_8UC1);
    for (const auto& poly : project(pose))
        for (size_t i=0; i+1<poly.size(); ++i)
            cv::line(out,
                cv::Point(int(poly[i].x),   int(poly[i].y)),
                cv::Point(int(poly[i+1].x), int(poly[i+1].y)),
                cv::Scalar(255), thickness, cv::LINE_AA);
}

// ── project() — TCD метрика, всегда CPU ───────────────────────────────────

std::vector<std::vector<cv::Point2f>>
ModelRenderer::project(const SE3& pose, float* visibleRatio) const {
    const Mat3d R=pose.rotation(); const Vec3d t=pose.translation();
    static cv::Mat zbuf;
    if (firstframeINT == 1)
        zbuf = buildZBufferCPU(pose);
    // set to 0 to build buffer once per frame
    firstframeINT = 1;
    bool hasZ = !zbuf.empty();

    std::vector<std::vector<cv::Point2f>> result;
    int total=0, vis=0;
    constexpr float BIAS=0.002f;

    for (const auto& edge : model_.edges) {
        if (edge.pts.size()<2) continue; ++total;
        std::vector<cv::Point2f> poly;
        for (size_t i=0; i+1<edge.pts.size(); ++i) {
            Vec3d A=R*edge.pts[i]+t, B=R*edge.pts[i+1]+t;
            if (!clipNear(A,B,near_)) continue;
            cv::Point2f p0=projectPoint(A), p1=projectPoint(B);
            cv::Point pi0(int(p0.x),int(p0.y)), pi1(int(p1.x),int(p1.y));
            if (!cv::clipLine(cv::Rect(0,0,K_.width,K_.height),pi0,pi1)) continue;
            if (hasZ) {
                int mx=std::max(0,std::min(K_.width-1, (pi0.x+pi1.x)/2));
                int my=std::max(0,std::min(K_.height-1,(pi0.y+pi1.y)/2));
                float ez=float(((A+B)*0.5).z());
                if (ez > zbuf.at<float>(my,mx)+BIAS) continue;
            }
            if (poly.empty()) poly.push_back(cv::Point2f(pi0));
            poly.push_back(cv::Point2f(pi1));
        }
        if (!poly.empty()) { result.push_back(std::move(poly)); ++vis; }
    }
    if (visibleRatio && total>0) *visibleRatio=float(vis)/total;
    return result;
}

// ── CPU Z-буфер (для TCD и fallback рендера) ──────────────────────────────

cv::Mat ModelRenderer::buildZBufferCPU(const SE3& pose) const {
    const Mat3d R=pose.rotation(); const Vec3d t=pose.translation();
    cv::Mat zbuf(K_.height, K_.width, CV_32F,
                 cv::Scalar(std::numeric_limits<float>::max()));
    for (const auto& f : model_.faces) {
        Vec3d c0=R*f.v0+t, c1=R*f.v1+t, c2=R*f.v2+t;
        if (c0.z()<near_&&c1.z()<near_&&c2.z()<near_) continue;
        if (c0.z()<near_) c0.z()=(double)near_;
        if (c1.z()<near_) c1.z()=(double)near_;
        if (c2.z()<near_) c2.z()=(double)near_;
        rasterizeTriangleCPU(
            projectPoint(c0),float(c0.z()),
            projectPoint(c1),float(c1.z()),
            projectPoint(c2),float(c2.z()), zbuf);
    }
    return zbuf;
}

void ModelRenderer::rasterizeTriangleCPU(
    const cv::Point2f& p0,float z0,
    const cv::Point2f& p1,float z1,
    const cv::Point2f& p2,float z2,
    cv::Mat& zbuf)
{
    int W=zbuf.cols, H=zbuf.rows;
    int minY=std::max(0,  (int)std::floor(std::min({p0.y,p1.y,p2.y})));
    int maxY=std::min(H-1,(int)std::ceil (std::max({p0.y,p1.y,p2.y})));
    for (int y=minY; y<=maxY; ++y) {
        float fy=y+.5f, xv[2], zv[2]; int cnt=0;
        struct{const cv::Point2f &a,&b;float za,zb;}es[3]={
            {p0,p1,z0,z1},{p1,p2,z1,z2},{p2,p0,z2,z0}};
        for (auto& e:es) {
            if (cnt>=2) break;
            if ((e.a.y<=fy&&e.b.y>fy)||(e.b.y<=fy&&e.a.y>fy)){
                float tt=(fy-e.a.y)/(e.b.y-e.a.y);
                xv[cnt]=e.a.x+tt*(e.b.x-e.a.x);
                zv[cnt]=e.za+tt*(e.zb-e.za); ++cnt;
            }
        }
        if (cnt<2) continue;
        if (xv[0]>xv[1]){std::swap(xv[0],xv[1]);std::swap(zv[0],zv[1]);}
        int x0i=std::max(0,(int)std::ceil(xv[0]));
        int x1i=std::min(W-1,(int)std::floor(xv[1]));
        float sp=xv[1]-xv[0];
        for (int x=x0i;x<=x1i;++x){
            float tt=(sp>0.f)?(x-xv[0])/sp:0.f;
            float z=zv[0]+tt*(zv[1]-zv[0]);
            float& cur=zbuf.at<float>(y,x);
            if (z<cur) cur=z;
        }
    }
}

bool ModelRenderer::isVisible(const Edge3D&, const SE3&) const { return true; }

} // namespace pe
