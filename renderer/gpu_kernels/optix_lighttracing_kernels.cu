#include "renderer_kernel_common.h"

static constexpr bool debugVisualizeBaseColor = false;



CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRaySignature::set(&visibility);
}



#define USE_ONLY_DIRECTIONAL_LIGHT 1

CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleLight(
    const WavelengthSamples &wls,
    const float ul, const float up0, const float up1, const float ud0, const float ud1,
    Point3D* const position, Normal3D* const normal, Vector3D* const direction,
    float* const areaPDensity, float* const dirPDensity)
{
    float lightProb = 1.0f;

    float uInst;
    uint32_t emitterType;
    if constexpr (USE_ONLY_DIRECTIONAL_LIGHT) {
        uInst = ul;
        emitterType = 1;
    }
    else {
        float emitterTypeProb;
        emitterType = sampleDiscrete(
            ul, &emitterTypeProb, &uInst, plp.f->lightInstDist.integral(), plp.f->dirLightInstDist.integral());
        lightProb *= emitterTypeProb;
    }
    const LightDistribution lightInstDist = emitterType == 0 ?
        plp.f->lightInstDist : plp.f->dirLightInstDist;

    // JP: まずはインスタンスをサンプルする。
    // EN: First, sample an instance.
    float instProb;
    float uGeomInst;
    const uint32_t instIdx = lightInstDist.sample(uInst, &instProb, &uGeomInst);
    const Instance &inst = plp.f->instances[instIdx];
    lightProb *= instProb;
    if (instProb == 0.0f) {
        *areaPDensity = 0.0f;
        return SampledSpectrum::Zero();
    }
    //Assert(inst.lightGeomInstDist.integral() > 0.0f,
    //       "Non-emissive inst %u, prob %g, u: %g(0x%08x).", instIdx, instProb, ul, *(uint32_t*)&ul);

    // JP: 次にサンプルしたインスタンスに属するジオメトリインスタンスをサンプルする。
    // EN: Next, sample a geometry instance which belongs to the sampled instance.
    float geomInstProb;
    float uPrim;
    const GeometryGroup &geomGroup = plp.s->geometryGroups[inst.geomGroupSlot];
    const LightDistribution lightGeomInstDist = emitterType == 0 ?
        geomGroup.lightGeomInstDist : geomGroup.dirLightGeomInstDist;
    const uint32_t geomInstIdxInGroup = lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim);
    const uint32_t geomInstIdx = geomGroup.geomInstSlots[geomInstIdxInGroup];
    const GeometryInstance &geomInst = plp.s->geometryInstances[geomInstIdx];
    lightProb *= geomInstProb;
    if (geomInstProb == 0.0f) {
        *areaPDensity = 0.0f;
        return SampledSpectrum::Zero();
    }
    //Assert(geomInst.emitterPrimDist.integral() > 0.0f,
    //       "Non-emissive geom inst %u, prob %g, u: %g.", geomInstIdx, geomInstProb, uGeomInst);

    // JP: 最後に、サンプルしたジオメトリインスタンスに属するプリミティブをサンプルする。
    // EN: Finally, sample a primitive which belongs to the sampled geometry instance.
    float primProb;
    const uint32_t primIdx = geomInst.emitterPrimDist.sample(uPrim, &primProb);
    lightProb *= primProb;

    //printf("%u-%u-%u: %g\n", instIdx, geomInstIdx, primIdx, lightProb);

    const SurfaceMaterial &mat = plp.s->surfaceMaterials[geomInst.surfMatSlot];

    const shared::Triangle &tri = geomInst.triangles[primIdx];
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

    const Normal3D geomNormal = cross(p[1] - p[0], p[2] - p[0]);

    float t0, t1, t2;
    {
        // Uniform sampling on unit triangle
        // A Low-Distortion Map Between Triangle and Square
        t0 = 0.5f * up0;
        t1 = 0.5f * up1;
        const float offset = t1 - t0;
        if (offset > 0)
            t1 += offset;
        else
            t0 -= offset;
        t2 = 1 - (t0 + t1);

        const float recArea = 2.0f / geomNormal.length();
        *areaPDensity = lightProb * recArea;
    }
    *position = t0 * p[0] + t1 * p[1] + t2 * p[2];
    *normal = t0 * v[0].normal + t1 * v[1].normal + t2 * v[2].normal;
    *normal = normalize(inst.transform * *normal);
    if (emitterType == 0) {
        const Vector3D dirLocal = cosineSampleHemisphere(ud0, ud1);
        const ReferenceFrame lightFrame(*normal);
        *direction = lightFrame.fromLocal(dirLocal);
        *dirPDensity = dirLocal.z / pi_v<float>;
    }
    else {
        *direction = *normal;
        *dirPDensity = 1.0f;
    }

    SampledSpectrum Le = SampledSpectrum::Zero();
    if (mat.emittance) {
        const TexCoord2D texCoord = t0 * v[0].texCoord + t1 * v[1].texCoord + t2 * v[2].texCoord;
        const float4 texValue = tex2DLod<float4>(mat.emittance, texCoord.u, texCoord.v, 0.0f);
        const TripletSpectrum spEmittance = createTripletSpectrum(
            SpectrumType::LightSource, ColorSpace::Rec709_D65,
            texValue.x, texValue.y, texValue.z);
        Le = spEmittance.evaluate(wls);
        if (emitterType == 0)
            Le /= pi_v<float>;
    }

    //printf("Le: " SPDFMT ", dir: " V3FMT "\n", spdprint(Le), v3print(*direction));

    return Le;
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void performNextEventEstimation(
    const SampledSpectrum &throughput,
    const InteractionPoint &interPt, const BSDF &bsdf, const BSDFQuery &bsdfQuery,
    const WavelengthSamples &wls, PCG32RNG &rng)
{
    const PerspectiveCamera &camera = plp.f->camera;
    float2 screenPos = camera.calcScreenPosition(interPt.position);
    if (screenPos.x < 0.0f || screenPos.x >= 1.0f ||
        screenPos.y < 0.0f || screenPos.y >= 1.0f)
        return;

    Vector3D eyeRayDir = interPt.position - camera.position;
    Normal3D camNormal = camera.orientation.toMatrix3x3() * Normal3D(0, 0, -1);
    float epCos = dot(eyeRayDir, camNormal);
    if (epCos <= 0.0f)
        return;

    float dist2 = eyeRayDir.squaredLength();
    float traceLength = std::sqrt(dist2);
    eyeRayDir /= traceLength;
    epCos /= traceLength;

    float visibility = 1.0f;
    VisibilityRaySignature::trace(
        plp.f->travHandle,
        camera.position.toNativeType(), eyeRayDir.toNativeType(), 0.0f, 0.9999f * traceLength, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        LightTracingRayType::Visibility, shared::maxNumRayTypes, LightTracingRayType::Visibility,
        visibility);

    if (visibility > 0.0f) {
        if (plp.f->enableVolume)
            visibility *= std::expf(-plp.f->volumeDensity * traceLength);

        Vector3D vOutLocal = interPt.toLocal(-eyeRayDir);
        SampledSpectrum We = SampledSpectrum::One();
        SampledSpectrum fsValue = bsdf.evaluateF(bsdfQuery, vOutLocal);
        float spCos = interPt.calcAbsDot(eyeRayDir);
        float G = epCos * spCos / dist2;
        SampledSpectrum contribution = throughput * fsValue * We * (visibility * G);
        //printf(
        //    "cont: " SPDFMT
        //    ", thr: " SPDFMT
        //    ", fs: " SPDFMT
        //    ", We: " SPDFMT
        //    ", dist: %g, spCos: %g, epCos: %g, vis: %g\n",
        //    spdprint(contribution), spdprint(throughput), spdprint(fsValue), spdprint(We),
        //    traceLength, spCos, epCos, visibility);

        uint2 pix(plp.s->imageSize.x * screenPos.x, plp.s->imageSize.y * screenPos.y);
        plp.s->ltTargetBuffer[pix].atomicAdd(wls, contribution);
    }
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void lightTrace_generic() {
    if constexpr (USE_ONLY_DIRECTIONAL_LIGHT) {
        if (plp.f->dirLightInstDist.integral() == 0.0f)
            return;
    }

    const uint32_t launchIndex = optixGetLaunchIndex().x;
    PCG32RNG rng = plp.s->ltRngBuffer[launchIndex];

    float wlPDensity;
    const auto wls = WavelengthSamples::createWithEqualOffsets(
        rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), &wlPDensity);

    Point3D rayOrigin;
    Vector3D rayDirection;
    Normal3D lpNormal;
    float lpAreaPDensity;
    float lDirPDensity;
    const SampledSpectrum Le = sampleLight(
        wls,
        rng.getFloat0cTo1o(),
        rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &rayOrigin, &lpNormal, &rayDirection, &lpAreaPDensity, &lDirPDensity);
    //printf(
    //    "org: (" V3FMT "), dir: (" V3FMT ")\n",
    //    v3print(rayOrigin), v3print(rayDirection));
    if (lpAreaPDensity == 0.0f || lDirPDensity == 0.0f) {
        plp.s->ltRngBuffer[launchIndex] = rng;
        return;
    }

    rayOrigin += 1e-3f * lpNormal;

    SampledSpectrum throughput =
        Le * (absDot(lpNormal, rayDirection) / (lDirPDensity * lpAreaPDensity * wlPDensity));
    //printf(
    //    "Le: (" SPDFMT "), wP: %g, areaP: %g, dirP: %g\n",
    //    spdprint(Le), wlPDensity, lpAreaPDensity, lDirPDensity);
    const float initImportance = throughput.importance(wls.selectedLambdaIndex());
    uint32_t pathLength = 0;
    while (true) {
        ++pathLength;

        uint32_t instSlot = 0xFFFFFFFF;
        uint32_t geomInstSlot = 0xFFFFFFFF;
        uint32_t primIndex = 0xFFFFFFFF;
        float b1 = 0.0f, b2 = 0.0f;
        float hitDist = 1e+10f;
        ClosestRaySignature::trace(
            plp.f->travHandle,
            rayOrigin.toNativeType(), rayDirection.toNativeType(), 0.0f, 1e+10f, 0.0f,
            0xFF, OPTIX_RAY_FLAG_NONE,
            LightTracingRayType::Closest, maxNumRayTypes, LightTracingRayType::Closest,
            instSlot, geomInstSlot, primIndex, b1, b2, hitDist);

        bool volEventHappens = false;
        if (plp.f->enableVolume) {
            const float fpDist = -std::log(1.0f - rng.getFloat0cTo1o()) / plp.f->volumeDensity;
            if (fpDist < hitDist) {
                volEventHappens = true;
                hitDist = fpDist;
            }
        }

        if (instSlot == 0xFFFFFFFF && !volEventHappens)
            break;

        InteractionPoint interPt;
        uint32_t surfMatSlot = 0xFFFFFFFF;
        if (volEventHappens) {
            interPt.position = rayOrigin + hitDist * rayDirection;
            interPt.shadingFrame = ReferenceFrame(-rayDirection);
            interPt.inMedium = true;
        }
        else {
            const Instance &inst = plp.f->instances[instSlot];
            const GeometryInstance &geomInst = plp.s->geometryInstances[geomInstSlot];
            computeSurfacePoint(inst, geomInst, primIndex, b1, b2, &interPt);

            surfMatSlot = geomInst.surfMatSlot;

            if (geomInst.normal) {
                const Normal3D modLocalNormal = geomInst.readModifiedNormal(
                    geomInst.normal, geomInst.normalDimInfo, interPt.asSurf.texCoord);
                applyBumpMapping(modLocalNormal, &interPt.shadingFrame);
            }
        }

        const Vector3D vIn(-rayDirection);
        const Vector3D vInLocal = interPt.toLocal(vIn);

        // Russian roulette
        const float continueProb =
            std::fmin(throughput.importance(wls.selectedLambdaIndex()) / initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            break;

        BSDF bsdf;
        BSDFQuery bsdfQuery;
        if (volEventHappens) {
            bsdf.setup(plp.f->scatteringAlbedo);
            bsdfQuery = BSDFQuery();
        }
        else {
            const SurfaceMaterial &surfMat = plp.s->surfaceMaterials[surfMatSlot];
            bsdf.setup(surfMat, interPt.asSurf.texCoord, wls);
            bsdfQuery = BSDFQuery(vInLocal, interPt.toLocal(interPt.asSurf.geometricNormal), wls);
        }

        throughput /= continueProb;
        performNextEventEstimation(throughput, interPt, bsdf, bsdfQuery, wls, rng);

        const BSDFSample bsdfSample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult bsdfResult;
        const SampledSpectrum fsValue = bsdf.sampleF(bsdfQuery, bsdfSample, &bsdfResult);
        if (bsdfResult.dirPDensity == 0.0f)
            break; // sampling failed.
        rayDirection = interPt.fromLocal(bsdfResult.dirLocal);
        const float dotSGN = interPt.calcDot(rayDirection);
        const SampledSpectrum localThroughput = fsValue * (std::fabs(dotSGN) / bsdfResult.dirPDensity);
        throughput *= localThroughput;
        Assert(
            localThroughput.allNonNegativeFinite(),
            "tp: (" SPDFMT "), dotSGN: %g, dirP: %g",
            spdprint(localThroughput), dotSGN, bsdfResult.dirPDensity);

        rayOrigin = interPt.calcOffsetRayOrigin(dotSGN > 0.0f);
    }

    plp.s->ltRngBuffer[launchIndex] = rng;
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(lightTrace)() {
    lightTrace_generic();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(lightTrace)() {
    uint32_t instSlot = optixGetInstanceId();
    auto sbtr = HitGroupSBTRecordData::get();
    auto hp = HitPointParameter::get();
    float dist = optixGetRayTmax();

    ClosestRaySignature::set(&instSlot, &sbtr.geomInstSlot, &hp.primIndex, &hp.b1, &hp.b2, &dist);
}
