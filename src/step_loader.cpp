#include "step_loader.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>

#include <STEPControl_Reader.hxx>
#include <BRep_Tool.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopExp_Explorer.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <GCPnts_UniformDeflection.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <GeomLProp_SLProps.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <Bnd_Box.hxx>
#include <BRepBndLib.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>
#include <Poly_Triangulation.hxx>
#include <BRep_Tool.hxx>
#include <TopLoc_Location.hxx>

namespace pe {

namespace {

double edgeSharpnessAngle(const TopoDS_Edge& edge,
                          const TopTools_IndexedDataMapOfShapeListOfShape& edgeFaceMap)
{
    if (!edgeFaceMap.Contains(edge)) return 180.0;

    const TopTools_ListOfShape& faces = edgeFaceMap.FindFromKey(edge);
    if (faces.Size() < 2) return 180.0;

    auto it = faces.begin();
    const TopoDS_Face& f1 = TopoDS::Face(*it); ++it;
    const TopoDS_Face& f2 = TopoDS::Face(*it);

    auto getFaceNormal = [](const TopoDS_Face& f) -> gp_Vec {
        BRepAdaptor_Surface surf(f, Standard_True);
        double umid = (surf.FirstUParameter() + surf.LastUParameter()) * 0.5;
        double vmid = (surf.FirstVParameter() + surf.LastVParameter()) * 0.5;
        GeomLProp_SLProps props(surf.Surface().Surface(), umid, vmid, 1, 1e-6);
        if (!props.IsNormalDefined()) return gp_Vec(0, 0, 1);
        gp_Dir n = props.Normal();
        if (f.Orientation() == TopAbs_REVERSED) n.Reverse();
        return gp_Vec(n.X(), n.Y(), n.Z());
    };

    gp_Vec n1 = getFaceNormal(f1);
    gp_Vec n2 = getFaceNormal(f2);
    double mag = n1.Magnitude() * n2.Magnitude();
    if (mag < 1e-12) return 180.0;
    double cosA = std::max(-1.0, std::min(1.0, n1.Dot(n2) / mag));
    return std::acos(cosA) * 180.0 / M_PI;
}

std::vector<Vec3d> tessellateEdge(const TopoDS_Edge& edge, double deflection)
{
    std::vector<Vec3d> pts;
    BRepAdaptor_Curve curve(edge);
    if (curve.GetType() == GeomAbs_OtherCurve) return pts;

    GCPnts_UniformDeflection sampler(curve, deflection, Standard_True);
    if (!sampler.IsDone() || sampler.NbPoints() < 2) {
        gp_Pnt p0, p1;
        curve.D0(curve.FirstParameter(), p0);
        curve.D0(curve.LastParameter(),  p1);
        pts.push_back({p0.X(), p0.Y(), p0.Z()});
        pts.push_back({p1.X(), p1.Y(), p1.Z()});
        return pts;
    }
    for (int i = 1; i <= sampler.NbPoints(); ++i) {
        gp_Pnt p = sampler.Value(i);
        pts.push_back({p.X(), p.Y(), p.Z()});
    }
    return pts;
}

} // anonymous namespace

Model3D loadStep(const std::string& path, double deflection, double sharpAngleDeg)
{
    std::cout << "[StepLoader] Loading: " << path << "\n";

    STEPControl_Reader reader;
    if (reader.ReadFile(path.c_str()) != IFSelect_RetDone)
        throw std::runtime_error("Failed to read STEP file: " + path);

    reader.TransferRoots();
    TopoDS_Shape shape = reader.OneShape();
    if (shape.IsNull())
        throw std::runtime_error("STEP file produced empty shape");

    std::cout << "[StepLoader] Meshing...\n";
    BRepMesh_IncrementalMesh mesh(shape, deflection, Standard_False,
                                  0.5 * M_PI / 180.0, Standard_True);
    mesh.Perform();

    TopTools_IndexedDataMapOfShapeListOfShape edgeFaceMap;
    TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edgeFaceMap);

    Model3D model;
    int edgeCount = 0;
    constexpr double MM2M = 0.001;

    // ── Рёбра ────────────────────────────────────────────────────────────
    for (TopExp_Explorer exp(shape, TopAbs_EDGE); exp.More(); exp.Next()) {
        const TopoDS_Edge& edge = TopoDS::Edge(exp.Current());
        if (BRep_Tool::Degenerated(edge)) continue;

        auto pts = tessellateEdge(edge, deflection);
        if (pts.size() < 2) continue;

        // мм → м
        for (auto& p : pts) p *= MM2M;

        double angleDeg = edgeSharpnessAngle(edge, edgeFaceMap);
        bool sharp = (angleDeg > sharpAngleDeg);
        model.edges.push_back({std::move(pts), sharp});
        ++edgeCount;
    }

    // ── Треугольные грани для Z-буфера ───────────────────────────────────
    int faceCount = 0;
    for (TopExp_Explorer exp(shape, TopAbs_FACE); exp.More(); exp.Next()) {
        const TopoDS_Face& face = TopoDS::Face(exp.Current());
        TopLoc_Location loc;
        Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, loc);
        if (tri.IsNull()) continue;

        // Матрица преобразования грани в глобальные координаты
        const gp_Trsf& trsf = loc.IsIdentity() ? gp_Trsf() : loc.IsIdentity() ? gp_Trsf() : loc;
        bool reversed = (face.Orientation() == TopAbs_REVERSED);

        for (int i = 1; i <= tri->NbTriangles(); ++i) {
            int n1, n2, n3;
            tri->Triangle(i).Get(n1, n2, n3);
            if (reversed) std::swap(n2, n3);

            auto getVtx = [&](int idx) -> Vec3d {
                gp_Pnt p = tri->Node(idx).Transformed(trsf);
                return Vec3d(p.X(), p.Y(), p.Z()) * MM2M;
            };

            model.faces.push_back({getVtx(n1), getVtx(n2), getVtx(n3)});
            ++faceCount;
        }
    }

    // ── BBox ─────────────────────────────────────────────────────────────
    Bnd_Box bbox;
    BRepBndLib::Add(shape, bbox);
    double x0,y0,z0, x1,y1,z1;
    bbox.Get(x0,y0,z0, x1,y1,z1);
    model.bbox_min = Vec3d{x0,y0,z0} * MM2M;
    model.bbox_max = Vec3d{x1,y1,z1} * MM2M;
    model.centroid = (model.bbox_min + model.bbox_max) * 0.5;

    std::cout << "[StepLoader] Done. Edges: " << edgeCount
              << "  Faces: " << faceCount
              << "  BBox: [" << x0 << "," << y0 << "," << z0
              << "] - [" << x1 << "," << y1 << "," << z1 << "]\n";
    return model;
}

} // namespace pe



/*
namespace pe {

namespace {

// Возвращает угол между нормалями двух граней вдоль общего ребра (в градусах).
// Если ребро принадлежит только одной грани (граница) – возвращает 180.
double edgeSharpnessAngle(const TopoDS_Edge& edge,
                          const TopTools_IndexedDataMapOfShapeListOfShape& edgeFaceMap)
{
    if (!edgeFaceMap.Contains(edge)) return 180.0;

    const TopTools_ListOfShape& faces = edgeFaceMap.FindFromKey(edge);
    if (faces.Size() < 2) return 180.0;

    // Берём средину ребра для вычисления нормалей
    BRepAdaptor_Curve curve(edge);
    double mid = (curve.FirstParameter() + curve.LastParameter()) * 0.5;
    gp_Pnt pMid;
    gp_Vec tangent;
    curve.D1(mid, pMid, tangent);

    auto it = faces.begin();
    auto getN = [&](const TopoDS_Face& face) -> gp_Vec {
        BRepAdaptor_Surface surf(face, true);
        // Проецируем точку на поверхность (грубо: UV параметры)
        double u = (surf.FirstUParameter() + surf.LastUParameter()) * 0.5;
        double v = (surf.FirstVParameter() + surf.LastVParameter()) * 0.5;
        GeomLProp_SLProps props(surf.Surface().Surface(), u, v, 1, 1e-6);
        if (!props.IsNormalDefined()) return gp_Vec(0,0,1);
        gp_Dir n = props.Normal();
        if (face.Orientation() == TopAbs_REVERSED) n.Reverse();
        return gp_Vec(n.X(), n.Y(), n.Z());
    };

    TopoDS_Face f1 = TopoDS::Face(*it); ++it;
    TopoDS_Face f2 = TopoDS::Face(*it);

    gp_Vec n1 = getN(f1);
    gp_Vec n2 = getN(f2);

    double cosA = n1.Dot(n2) / (n1.Magnitude() * n2.Magnitude() + 1e-12);
    cosA = std::max(-1.0, std::min(1.0, cosA));
    double angle = std::acos(cosA) * 180.0 / M_PI;
    return angle;
}

// Тесселирует одно ребро в вектор 3D точек
std::vector<Vec3d> tessellateEdge(const TopoDS_Edge& edge, double deflection)
{
    std::vector<Vec3d> pts;
    BRepAdaptor_Curve curve(edge);
    if (curve.GetType() == GeomAbs_OtherCurve) return pts;

    GCPnts_UniformDeflection sampler(curve, deflection, Standard_True);
    if (!sampler.IsDone()) {
        // Фолбэк: просто конечные точки
        gp_Pnt p1, p2;
        curve.D0(curve.FirstParameter(), p1);
        curve.D0(curve.LastParameter(),  p2);
        pts.push_back({p1.X(), p1.Y(), p1.Z()});
        pts.push_back({p2.X(), p2.Y(), p2.Z()});
        return pts;
    }
    for (int i = 1; i <= sampler.NbPoints(); ++i) {
        gp_Pnt p = sampler.Value(i);
        pts.push_back({p.X(), p.Y(), p.Z()});
    }
    return pts;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────

Model3D loadStep(const std::string& path, double deflection, double sharpAngleDeg)
{
    std::cout << "[StepLoader] Loading: " << path << "\n";

    STEPControl_Reader reader;
    IFSelect_ReturnStatus status = reader.ReadFile(path.c_str());
    if (status != IFSelect_RetDone) {
        throw std::runtime_error("Failed to read STEP file: " + path);
    }

    reader.TransferRoots();
    TopoDS_Shape shape = reader.OneShape();
    if (shape.IsNull()) {
        throw std::runtime_error("STEP file produced empty shape");
    }

    // Инкрементальная мешировка (нужна для корректной работы BRepMesh)
    std::cout << "[StepLoader] Meshing...\n";
    BRepMesh_IncrementalMesh mesh(shape, deflection, Standard_False,
                                  0.5 * M_PI / 180.0, Standard_True);
    mesh.Perform();

    // Карта ребро → смежные грани (для классификации резкости)
    TopTools_IndexedDataMapOfShapeListOfShape edgeFaceMap;
    TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edgeFaceMap);

    Model3D model;
    int edgeCount = 0;

    for (TopExp_Explorer exp(shape, TopAbs_EDGE); exp.More(); exp.Next()) {
        const TopoDS_Edge& edge = TopoDS::Edge(exp.Current());

        // Пропускаем вырожденные рёбра
        if (BRep_Tool::Degenerated(edge)) continue;

        auto pts = tessellateEdge(edge, deflection);
        if (pts.size() < 2) continue;

        double angleDeg = edgeSharpnessAngle(edge, edgeFaceMap);
        bool sharp = (angleDeg > sharpAngleDeg);

        model.edges.push_back({std::move(pts), sharp});
        ++edgeCount;
    }

    // Ограничивающий параллелепипед
    Bnd_Box bbox;
    BRepBndLib::Add(shape, bbox);
    double x0,y0,z0, x1,y1,z1;
    bbox.Get(x0,y0,z0, x1,y1,z1);
    model.bbox_min = {x0, y0, z0};
    model.bbox_max = {x1, y1, z1};
    model.centroid = (model.bbox_min + model.bbox_max) * 0.5;

    std::cout << "[StepLoader] Done. Edges: " << edgeCount
              << "  BBox: ["
              << x0 << "," << y0 << "," << z0 << "] – ["
              << x1 << "," << y1 << "," << z1 << "]\n";
    return model;
}

} // namespace pe
*/