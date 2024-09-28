#pragma once

#include "../renderer_shared.h"

using namespace rtc9;
using namespace rtc9::shared;
using namespace rtc9::device;



struct HitGroupSBTRecordData {
    uint32_t geomInstSlot;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};

struct HitPointParameter {
    uint32_t primIndex;
    float b1, b2;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get() {
        HitPointParameter ret;
        float2 bc = optixGetTriangleBarycentrics();
        ret.b1 = bc.x;
        ret.b2 = bc.y;
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

struct InteractionPoint {
    Point3D position;
    struct SurfaceProperties {
        Normal3D geometricNormal;
        TexCoord2D texCoord;
        float hypAreaPDensity;
    } asSurf;
    ReferenceFrame shadingFrame;
    uint32_t inMedium : 1;

    CUDA_DEVICE_FUNCTION Vector3D toLocal(const Vector3D &v) const {
        return shadingFrame.toLocal(v);
    }
    CUDA_DEVICE_FUNCTION Vector3D fromLocal(const Vector3D &v) const {
        return shadingFrame.fromLocal(v);
    }

    CUDA_DEVICE_FUNCTION float calcDot(const Vector3D &dir) const {
        if (inMedium)
            return 1.0f;
        else
            return dot(dir, asSurf.geometricNormal);
    }
    CUDA_DEVICE_FUNCTION float calcAbsDot(const Vector3D &dir) const {
        if (inMedium)
            return 1.0f;
        else
            return absDot(dir, asSurf.geometricNormal);
    }

    CUDA_DEVICE_FUNCTION Point3D calcOffsetRayOrigin(bool evalFrontFace) const {
        if (inMedium)
            return position;
        else
            return offsetRayOrigin(position, (evalFrontFace ? 1 : -1) * asSurf.geometricNormal);
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint(
    const Instance &inst, const GeometryInstance &geomInst,
    uint32_t primIndex, float b1, float b2,
    InteractionPoint* interPt) {
    const Triangle &tri = geomInst.triangles[primIndex];
    const Vertex (&vs)[] = {
        geomInst.vertices[tri.indices[0]],
        geomInst.vertices[tri.indices[1]],
        geomInst.vertices[tri.indices[2]],
    };
    const Point3D ps[] = {
        inst.transform * vs[0].position,
        inst.transform * vs[1].position,
        inst.transform * vs[2].position,
    };

    float b0 = 1 - (b1 + b2);

    interPt->inMedium = false;
    interPt->position = b0 * ps[0] + b1 * ps[1] + b2 * ps[2];
    interPt->asSurf.geometricNormal = cross(ps[1] - ps[0], ps[2] - ps[0]);
    float area = 0.5f * interPt->asSurf.geometricNormal.length();
    interPt->asSurf.geometricNormal /= (2 * area);
    Normal3D localShadingNormal = b0 * vs[0].normal + b1 * vs[1].normal + b2 * vs[2].normal;
    Vector3D localShadingTangent = b0 * vs[0].tangent + b1 * vs[1].tangent + b2 * vs[2].tangent;
    interPt->shadingFrame = ReferenceFrame(
        normalize(inst.transform * localShadingNormal),
        normalize(inst.transform * localShadingTangent));
    interPt->asSurf.texCoord = b0 * vs[0].texCoord + b1 * vs[1].texCoord + b2 * vs[2].texCoord;

    const GeometryGroup &geomGroup = plp.s->geometryGroups[inst.geomGroupSlot];

    float lightProb = 1.0f;
    //if (plp.f->enableEnvironmentalLight)
    //    lightProb *= (1 - probToSampleEnvLight);
    float geomGroupImportance = geomGroup.lightGeomInstDist.integral();
    float instImportance = pow2(inst.uniformScale) * geomGroupImportance;
    float instProb = instImportance / plp.f->lightInstDist.integral();
    if (instProb == 0.0f) {
        interPt->asSurf.hypAreaPDensity = 0.0f;
        return;
    }
    lightProb *= instProb;
    float geomInstImportance = geomInst.emitterPrimDist.integral();
    float geomInstProb = geomInstImportance / geomGroupImportance;
    if (geomInstProb == 0.0f) {
        interPt->asSurf.hypAreaPDensity = 0.0f;
        return;
    }
    lightProb *= geomInstProb;
    lightProb *= geomInst.emitterPrimDist.evaluatePMF(primIndex);
    interPt->asSurf.hypAreaPDensity = lightProb / area;
}



template <bool useSolidAngleSampling>
CUDA_DEVICE_FUNCTION CUDA_INLINE void sampleLight(
    const Point3D &shadingPoint,
    bool sampleEnvLight, const WavelengthSamples &wls, float ul, float u0, float u1,
    shared::LightSample* lightSample, float* areaPDensity) {
    if (sampleEnvLight) {
        Vector3D dir;
        float dirPDensity;
        SampledSpectrum radiance = plp.f->envLight.sample(wls, u0, u1, &dir, &dirPDensity);

        lightSample->emittance = pi_v<float> * radiance;
        lightSample->position = static_cast<Point3D>(dir);
        lightSample->normal = -Normal3D(lightSample->position);
        lightSample->atInfinity = true;
        *areaPDensity = dirPDensity;
    }
    else {
        float lightProb = 1.0f;

        if (plp.f->lightInstDist.integral() == 0.0f) {
            *areaPDensity = 0.0f;
            return;
        }

        // JP: まずはインスタンスをサンプルする。
        // EN: First, sample an instance.
        float instProb;
        float uGeomInst;
        uint32_t instIndex = plp.f->lightInstDist.sample(ul, &instProb, &uGeomInst);
        lightProb *= instProb;
        const Instance &inst = plp.f->instances[instIndex];
        if (instProb == 0.0f) {
            *areaPDensity = 0.0f;
            return;
        }
        //Assert(inst.lightGeomInstDist.integral() > 0.0f,
        //       "Non-emissive inst %u, prob %g, u: %g(0x%08x).", instIndex, instProb, ul, *(uint32_t*)&ul);

        // JP: 次にサンプルしたインスタンスに属するジオメトリインスタンスをサンプルする。
        // EN: Next, sample a geometry instance which belongs to the sampled instance.
        float geomInstProb;
        float uPrim;
        const GeometryGroup &geomGroup = plp.s->geometryGroups[inst.geomGroupSlot];
        uint32_t geomInstIndexInGeomGroup =
            geomGroup.lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim);
        uint32_t geomInstIndex = geomGroup.geomInstSlots[geomInstIndexInGeomGroup];
        lightProb *= geomInstProb;
        const GeometryInstance &geomInst = plp.s->geometryInstances[geomInstIndex];
        if (geomInstProb == 0.0f) {
            *areaPDensity = 0.0f;
            return;
        }
        //Assert(geomInst.emitterPrimDist.integral() > 0.0f,
        //       "Non-emissive geom inst %u, prob %g, u: %g.", geomInstIndex, geomInstProb, uGeomInst);

        // JP: 最後に、サンプルしたジオメトリインスタンスに属するプリミティブをサンプルする。
        // EN: Finally, sample a primitive which belongs to the sampled geometry instance.
        float primProb;
        uint32_t primIndex = geomInst.emitterPrimDist.sample(uPrim, &primProb);
        lightProb *= primProb;

        //printf("%u-%u-%u: %g\n", instIndex, geomInstIndex, primIndex, lightProb);

        const SurfaceMaterial &mat = plp.s->surfaceMaterials[geomInst.surfMatSlot];

        const shared::Triangle &tri = geomInst.triangles[primIndex];
        const shared::Vertex (&v)[3] = {
            geomInst.vertices[tri.indices[0]],
            geomInst.vertices[tri.indices[1]],
            geomInst.vertices[tri.indices[2]]
        };
        const Point3D p[3] = {
            inst.transform * v[0].position,
            inst.transform * v[1].position,
            inst.transform * v[2].position,
        };

        Normal3D geomNormal = cross(p[1] - p[0], p[2] - p[0]);

        float t0, t1, t2;
        if constexpr (useSolidAngleSampling) {
            // Uniform sampling in solid angle subtended by the triangle for the shading point.
            float dist;
            Vector3D dir;
            float dirPDF;
            {
                const auto project = [](const Vector3D &vA, const Vector3D &vB) {
                    return normalize(vA - dot(vA, vB) * vB);
                };

                // TODO: ? compute in the local coordinates.
                Vector3D A = normalize(p[0] - shadingPoint);
                Vector3D B = normalize(p[1] - shadingPoint);
                Vector3D C = normalize(p[2] - shadingPoint);
                Vector3D cAB = normalize(cross(A, B));
                Vector3D cBC = normalize(cross(B, C));
                Vector3D cCA = normalize(cross(C, A));
                //float cos_a = dot(B, C);
                //float cos_b = dot(C, A);
                float cos_c = dot(A, B);
                float cosAlpha = -dot(cAB, cCA);
                float cosBeta = -dot(cBC, cAB);
                float cosGamma = -dot(cCA, cBC);
                float alpha = std::acos(cosAlpha);
                float sinAlpha = std::sqrt(1 - pow2(cosAlpha));
                float sphArea = alpha + std::acos(cosBeta) + std::acos(cosGamma) - Pi;

                float sphAreaHat = sphArea * u0;
                float s = std::sin(sphAreaHat - alpha);
                float t = std::cos(sphAreaHat - alpha);
                float uu = t - cosAlpha;
                float vv = s + sinAlpha * cos_c;
                float q = ((vv * t - uu * s) * cosAlpha - vv) / ((vv * s + uu * t) * sinAlpha);

                Vector3D cHat = q * A + std::sqrt(1 - pow2(q)) * project(C, A);
                float z = 1 - u1 * (1 - dot(cHat, B));
                Vector3D P = z * B + std::sqrt(1 - pow2(z)) * project(cHat, B);

                const auto restoreBarycentrics = [&geomNormal]
                (const Point3D &org, const Vector3D &dir,
                 const Point3D &pA, const Point3D &pB, const Point3D &pC,
                 float* dist, float* b1, float* b2) {
                     Vector3D eAB = pB - pA;
                     Vector3D eAC = pC - pA;
                     Vector3D pVec = cross(dir, eAC);
                     float recDet = 1.0f / dot(eAB, pVec);
                     Vector3D tVec = org - pA;
                     *b1 = dot(tVec, pVec) * recDet;
                     Vector3D qVec = cross(tVec, eAB);
                     *b2 = dot(dir, qVec) * recDet;
                     *dist = dot(eAC, qVec) * recDet;
                };
                dir = P;
                restoreBarycentrics(shadingPoint, dir, p[0], p[1], p[2], &dist, &t1, &t2);
                t0 = 1 - t1 - t2;
                dirPDF = 1 / sphArea;
            }

            geomNormal = normalize(geomNormal);
            float lpCos = -dot(dir, geomNormal);
            if (lpCos > 0 && isfinite(dirPDF))
                *areaPDensity = lightProb * (dirPDF * lpCos / pow2(dist));
            else
                *areaPDensity = 0.0f;
        }
        else {
            // Uniform sampling on unit triangle
            // A Low-Distortion Map Between Triangle and Square
            t0 = 0.5f * u0;
            t1 = 0.5f * u1;
            float offset = t1 - t0;
            if (offset > 0)
                t1 += offset;
            else
                t0 -= offset;
            t2 = 1 - (t0 + t1);

            float recArea = 2.0f / geomNormal.length();
            *areaPDensity = lightProb * recArea;
        }
        lightSample->position = t0 * p[0] + t1 * p[1] + t2 * p[2];
        lightSample->atInfinity = false;
        lightSample->normal = t0 * v[0].normal + t1 * v[1].normal + t2 * v[2].normal;
        lightSample->normal = normalize(inst.transform * lightSample->normal);

        lightSample->emittance = SampledSpectrum::Zero();
        if (mat.emittance) {
            TexCoord2D texCoord = t0 * v[0].texCoord + t1 * v[1].texCoord + t2 * v[2].texCoord;
            float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.u, texCoord.v, 0.0f);
            TripletSpectrum spEmittance = createTripletSpectrum(
                SpectrumType::LightSource, ColorSpace::Rec709_D65,
                texValue.x, texValue.y, texValue.z);
            lightSample->emittance = spEmittance.evaluate(wls);
        }
    }
}

CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum performNextEventEstimation(
    const InteractionPoint &interPt, const BSDF &bsdf, const BSDFQuery &bsdfQuery,
    const WavelengthSamples &wls, PCG32RNG &rng) {
    float uLight = rng.getFloat0cTo1o();
    bool selectEnvLight = false;
    float probToSampleCurLightType = 1.0f;
    if (plp.f->enableEnvironmentalLight) {
        if (plp.f->lightInstDist.integral() > 0.0f) {
            if (uLight < probToSampleEnvLight) {
                probToSampleCurLightType = probToSampleEnvLight;
                uLight /= probToSampleCurLightType;
                selectEnvLight = true;
            }
            else {
                probToSampleCurLightType = 1.0f - probToSampleEnvLight;
                uLight = (uLight - probToSampleEnvLight) / probToSampleCurLightType;
            }
        }
        else {
            selectEnvLight = true;
        }
    }

    LightSample lightSample;
    float areaPDensity;
    sampleLight<false>(
        interPt.position,
        selectEnvLight, wls, uLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &lightSample, &areaPDensity);
    if (areaPDensity == 0.0f)
        return SampledSpectrum::Zero();
    areaPDensity *= probToSampleCurLightType;

    Vector3D shadowRayDir = lightSample.atInfinity ?
        -Vector3D(lightSample.position) :
        (interPt.position - lightSample.position);
    float lpCos = dot(shadowRayDir, lightSample.normal);
    if (lpCos <= 0.0f)
        return SampledSpectrum::Zero();

    Point3D lightPoint;
    float dist2 = 1.0f;
    float traceLength = 1.0f;
    if (lightSample.atInfinity) {
        traceLength = 2 * plp.f->worldDimInfo.radius;
        lightPoint = interPt.position + traceLength * -shadowRayDir;
    }
    else {
        lightPoint = offsetRayOrigin(lightSample.position, lightSample.normal);
        shadowRayDir = interPt.position - lightPoint;
        dist2 = shadowRayDir.squaredLength();
        traceLength = std::sqrt(dist2);
        shadowRayDir /= traceLength;
        lpCos /= traceLength;
    }

    float visibility = 1.0f;
    VisibilityRaySignature::trace(
        plp.f->travHandle,
        lightPoint.toNativeType(), shadowRayDir.toNativeType(), 0.0f, traceLength * 0.9999f, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        PathTracingRayType::Visibility, shared::maxNumRayTypes, PathTracingRayType::Visibility,
        visibility);

    SampledSpectrum ret = SampledSpectrum::Zero();
    if (visibility > 0.0f) {
        if (plp.f->enableVolume)
            visibility *= std::expf(-plp.f->volumeDensity * std::sqrtf(dist2));

        Vector3D vInLocal = interPt.toLocal(-shadowRayDir);
        float bsdfPDensity = bsdf.evaluatePDF(bsdfQuery, vInLocal) * lpCos / dist2;
        if (!rtc9::isfinite(bsdfPDensity))
            bsdfPDensity = 0.0f;
        float lightPDensity = areaPDensity;
        float misWeight = pow2(lightPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));

        // Assume a diffuse emitter.
        SampledSpectrum Le = lightSample.emittance / pi_v<float>;
        SampledSpectrum fsValue = bsdf.evaluateF(bsdfQuery, vInLocal);
        float spCos = interPt.calcAbsDot(shadowRayDir);
        float G = lpCos * spCos / dist2;
        ret = fsValue * Le * (visibility * misWeight * G / areaPDensity);
    }

    return ret;
}