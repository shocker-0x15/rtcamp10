#include "renderer_kernel_common.h"

static constexpr bool debugVisualizeBaseColor = false;



CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRaySignature::set(&visibility);
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_generic() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex());
    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    const PerspectiveCamera &camera = plp.f->camera;

    SampledSpectrum We0;
    SampledSpectrum We1;
    Point3D rayOrigin;
    Vector3D rayDirection;
    float areaPDens;
    float dirPDens;
    float cosX0;
    const float2 screenPos(
        (launchIndex.x + rng.getFloat0cTo1o()) / plp.s->imageSize.x,
        (launchIndex.y + rng.getFloat0cTo1o()) / plp.s->imageSize.y);
    camera.sampleRay(
        screenPos,
        &rayOrigin, &We0, &areaPDens,
        &rayDirection, &We1, &dirPDens, &cosX0);

    float wlPDensity;
    auto wls = WavelengthSamples::createWithEqualOffsets(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), &wlPDensity);

    auto contribution = SampledSpectrum::Zero();
    const float imageSizeCorrFactor = plp.s->imageSize.x * plp.s->imageSize.y;
    SampledSpectrum throughput = (We0 * We1) *
        (cosX0 / (areaPDens * dirPDens * wlPDensity * imageSizeCorrFactor));
    float initImportance = throughput.importance(wls.selectedLambdaIndex());
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
            PathTracingRayType::Closest, maxNumRayTypes, PathTracingRayType::Closest,
            instSlot, geomInstSlot, primIndex, b1, b2, hitDist);

        bool volEventHappens = false;
        if (plp.f->enableVolume) {
            float fpDist = -std::log(1.0f - rng.getFloat0cTo1o()) / plp.f->volumeDensity;
            if (fpDist < hitDist) {
                volEventHappens = true;
                hitDist = fpDist;
            }
        }

        // JP: 何にもヒットしなかった、環境光源にレイが飛んだ場合。
        // EN: 
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
            computeSurfacePoint<true>(inst, geomInst, primIndex, b1, b2, &interPt);

            surfMatSlot = geomInst.surfMatSlot;

            if (geomInst.normal) {
                Normal3D modLocalNormal = geomInst.readModifiedNormal(
                    geomInst.normal, geomInst.normalDimInfo, interPt.asSurf.texCoord);
                applyBumpMapping(modLocalNormal, &interPt.shadingFrame);
            }
        }

        Vector3D vOut(-rayDirection);
        Vector3D vOutLocal = interPt.toLocal(vOut);

        // Implicit light sampling
        if (!volEventHappens) {
            const SurfaceMaterial &surfMat = plp.s->surfaceMaterials[surfMatSlot];
            if (vOutLocal.z > 0 && surfMat.emittance &&
                static_cast<EmitterType>(surfMat.emitterType) != EmitterType::Directional) {
                float4 texValue = tex2DLod<float4>(
                    surfMat.emittance, interPt.asSurf.texCoord.u, interPt.asSurf.texCoord.v, 0.0f);
                SampledSpectrum emittance = createTripletSpectrum(
                    SpectrumType::LightSource, ColorSpace::Rec709_D65,
                    texValue.x, texValue.y, texValue.z).evaluate(wls);
                float misWeight = 1.0f;
                if (pathLength > 1) {
                    float dist2 = squaredDistance(rayOrigin, interPt.position);
                    float lightPDensity = interPt.asSurf.hypAreaPDensity * dist2 / vOutLocal.z;
                    float bsdfPDensity = dirPDens;
                    misWeight = pow2(bsdfPDensity) / (pow2(bsdfPDensity) + pow2(lightPDensity));
                }
                // Assume a diffuse emitter.
                contribution += throughput * emittance * (misWeight / pi_v<float>);
            }
        }

        // Russian roulette
        float continueProb = std::fmin(throughput.importance(wls.selectedLambdaIndex()) / initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            break;

        constexpr TransportMode transportMode = TransportMode::Radiance;
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
                vOutLocal, interPt.toLocal(interPt.asSurf.geometricNormal),
                transportMode, wls);
        }

        throughput /= continueProb;
        if constexpr (debugVisualizeBaseColor) {
            contribution += throughput * bsdf.evaluateDHReflectanceEstimate(bsdfQuery);
            break;
        }
        contribution += throughput * performNextEventEstimation(interPt, bsdf, bsdfQuery, wls, rng);

        BSDFSample bsdfSample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult bsdfResult;
        SampledSpectrum fsValue = bsdf.sampleF(bsdfQuery, bsdfSample, &bsdfResult);
        if (bsdfResult.dirPDensity == 0.0f)
            break; // sampling failed.
        rayDirection = interPt.fromLocal(bsdfResult.dirLocal);
        float dotSGN = interPt.calcDot(rayDirection);
        SampledSpectrum localThroughput = fsValue * (std::fabs(dotSGN) / bsdfResult.dirPDensity);
        throughput *= localThroughput;
        Assert(
            localThroughput.allNonNegativeFinite(),
            "tp: (" SPDFMT "), dotSGN: %g, dirP: %g",
            spdprint(localThroughput), dotSGN, bsdfResult.dirPDensity);

        rayOrigin = interPt.calcOffsetRayOrigin(dotSGN > 0.0f);
        dirPDens = bsdfResult.dirPDensity;
    }

    plp.s->rngBuffer.write(launchIndex, rng);

    if (!contribution.allNonNegativeFinite()) {
        printf("Store Cont.: %4u, %4u: (" SPDFMT ")\n",
                launchIndex.x, launchIndex.y, spdprint(contribution));
        return;
    }

    if (plp.f->numAccumFrames == 0)
        plp.s->accumBuffer[launchIndex].reset();
    plp.s->accumBuffer[launchIndex].add(wls, contribution);
    plp.s->ltTargetBuffer[launchIndex] = DiscretizedSpectrum::Zero();
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTrace)() {
    pathTrace_generic();
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(pathTrace)() {
    uint32_t instSlot = optixGetInstanceId();
    auto sbtr = HitGroupSBTRecordData::get();
    auto hp = HitPointParameter::get();
    float dist = optixGetRayTmax();

    ClosestRaySignature::set(&instSlot, &sbtr.geomInstSlot, &hp.primIndex, &hp.b1, &hp.b2, &dist);
}
