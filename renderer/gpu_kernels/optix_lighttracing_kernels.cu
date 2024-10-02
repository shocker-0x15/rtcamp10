#include "renderer_kernel_common.h"

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRaySignature::set(&visibility);
}



#define USE_ONLY_DIRECTIONAL_LIGHT 1

CUDA_DEVICE_FUNCTION CUDA_INLINE void performNextEventEstimation(
    const SampledSpectrum &throughput,
    const InteractionPoint &interPt,
    const BSDF<TransportMode::Importance, BSDFGrade::Unidirectional> &bsdf,
    const BSDFQuery &bsdfQuery, const WavelengthSamples &wls, PCG32RNG &rng)
{
    const PerspectiveCamera &camera = plp.f->camera;
    SampledSpectrum We;
    float2 screenPos;
    float dirPDens;
    float epCos;
    float dist2;
    if (!camera.calcScreenPosition(interPt.position, &We, &screenPos, &dirPDens, &epCos, &dist2))
        return;

    //printf(
    //    "tp: (" SPDFMT "), screen: (%g, %g)\n",
    //    spdprint(throughput), screenPos.x, screenPos.y);

    float visibility = 1.0f;
    const float traceLength = std::sqrt(dist2);
    const Vector3D eyeRayDir = (interPt.position - camera.position) / traceLength;
    VisibilityRaySignature::trace(
        plp.f->travHandle,
        camera.position, eyeRayDir, 0.0f, 0.9999f * traceLength, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        LightTracingRayType::Visibility, shared::maxNumRayTypes, LightTracingRayType::Visibility,
        visibility);

    if (visibility > 0.0f) {
        if (plp.f->enableVolume)
            visibility *= std::expf(-plp.f->volumeDensity * traceLength);

        const Vector3D vOutLocal = interPt.toLocal(-eyeRayDir);
        const SampledSpectrum fsValue = bsdf.evaluateF(bsdfQuery, vOutLocal);
        const float spCos = interPt.calcAbsDot(eyeRayDir);
        const float G = epCos * spCos / dist2;
        const SampledSpectrum contribution = throughput * fsValue * We * (visibility * G);
        //printf(
        //    "cont: " SPDFMT
        //    ", thr: " SPDFMT
        //    ", fs: " SPDFMT
        //    ", We: " SPDFMT
        //    ", dist: %g, spCos: %g, epCos: %g, vis: %g\n",
        //    spdprint(contribution), spdprint(throughput), spdprint(fsValue), spdprint(We),
        //    traceLength, spCos, epCos, visibility);

        const uint2 pix(plp.s->imageSize.x * screenPos.x, plp.s->imageSize.y * screenPos.y);
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
    const SampledSpectrum Le = sampleLight<USE_ONLY_DIRECTIONAL_LIGHT>(
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

    SampledSpectrum throughput = Le *
        (absDot(lpNormal, rayDirection) / (lDirPDensity * lpAreaPDensity * wlPDensity * plp.s->numLightPaths));
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
            interPt.shadingFrame = ReferenceFrame(rayDirection);
            interPt.inMedium = true;
        }
        else {
            const Instance &inst = plp.f->instances[instSlot];
            const GeometryInstance &geomInst = plp.s->geometryInstances[geomInstSlot];
            computeSurfacePoint<false>(inst, geomInst, primIndex, b1, b2, &interPt);

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

        constexpr TransportMode transportMode = TransportMode::Importance;
        BSDF<transportMode, BSDFGrade::Unidirectional> bsdf;
        BSDFQuery bsdfQuery;
        if (volEventHappens) {
            bsdf.setup(plp.f->scatteringAlbedo);
            bsdfQuery = BSDFQuery();
        }
        else {
            const SurfaceMaterial &surfMat = plp.s->surfaceMaterials[surfMatSlot];
            bsdf.setup(surfMat, interPt.asSurf.texCoord, wls);
            bsdfQuery = BSDFQuery(
                vInLocal, interPt.toLocal(interPt.asSurf.geometricNormal),
                transportMode, wls);
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
