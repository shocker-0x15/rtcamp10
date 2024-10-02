#include "renderer_kernel_common.h"

static constexpr bool includeRrProbability = true;
static constexpr uint32_t debugPathLength = 0;

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRaySignature::set(&visibility);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(getHitInfo)() {
    const uint32_t instSlot = optixGetInstanceId();
    const auto sbtr = HitGroupSBTRecordData::get();
    const auto hp = HitPointParameter::get();
    const float dist = optixGetRayTmax();

    ClosestRaySignature::set(&instSlot, &sbtr.geomInstSlot, &hp.primIndex, &hp.b1, &hp.b2, &dist);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void storeLightVertex(
    const SurfacePointIdentifier &surfPtId, const Point3D &positionInMedium, const SampledSpectrum &flux,
    const Vector3D &dirInLocal, const Vector3D &dirInInMedium, float backwardConversionFactor,
    float probDensity, float prevProbDensity,
    float secondPrevPartialDenomMisWeight, float secondPrevProbRatioToFirst,
    bool deltaSampled, bool prevDeltaSampled, bool wlSelected, bool isInMedium, uint32_t pathLength)
{
    LightPathVertex lightVertex = {};
    if (isInMedium) {
        lightVertex.positionInMedium = positionInMedium;
        lightVertex.dirInInMedium = dirInInMedium;
    }
    else {
        lightVertex.instSlot = surfPtId.instSlot;
        lightVertex.geomInstSlot = surfPtId.geomInstSlot;
        lightVertex.primIndex = surfPtId.primIndex;
        lightVertex.bcB = surfPtId.bcB;
        lightVertex.bcC = surfPtId.bcC;
        lightVertex.dirInLocal = dirInLocal;
    }
    lightVertex.probDensity = probDensity;
    lightVertex.prevProbDensity = prevProbDensity;
    lightVertex.secondPrevPartialDenomMisWeight = secondPrevPartialDenomMisWeight;
    lightVertex.secondPrevProbRatioToFirst = secondPrevProbRatioToFirst;
    lightVertex.backwardConversionFactor = backwardConversionFactor;
    lightVertex.flux = flux;
    lightVertex.deltaSampled = deltaSampled;
    lightVertex.prevDeltaSampled = prevDeltaSampled;
    lightVertex.wlSelected = wlSelected;
    lightVertex.pathLength = pathLength;
    lightVertex.isInMedium = isInMedium;
    const uint32_t cacheIdx = atomicAdd(&plp.s->lvcBptPassInfo->numLightVertices, 1u);
    plp.s->lightVertexCache[cacheIdx] = lightVertex;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE SurfacePointIdentifier sampleLight(
    const float ul, const float up0, const float up1,
    InteractionPoint* const surfPt)
{
    SurfacePointIdentifier surfPtId = {};
    float lightProb = 1.0f;

    float emitterTypeProb;
    float uInst;
    const uint32_t emitterType = sampleDiscrete(
        ul, &emitterTypeProb, &uInst,
        plp.f->lightInstDist.integral(), plp.f->dirLightInstDist.integral());
    lightProb *= emitterTypeProb;
    const LightDistribution &lightInstDist = emitterType == 0 ?
        plp.f->lightInstDist : plp.f->dirLightInstDist;

    float instProb;
    float uGeomInst;
    surfPtId.instSlot = lightInstDist.sample(uInst, &instProb, &uGeomInst);
    const Instance &inst = plp.f->instances[surfPtId.instSlot];
    lightProb *= instProb;
    if (instProb == 0.0f) {
        surfPtId.instSlot = 0xFFFF'FFFF;
        return surfPtId;
    }

    float geomInstProb;
    float uPrim;
    const GeometryGroup &geomGroup = plp.s->geometryGroups[inst.geomGroupSlot];
    const LightDistribution &lightGeomInstDist = emitterType == 0 ?
        geomGroup.lightGeomInstDist : geomGroup.dirLightGeomInstDist;
    const uint32_t geomInstIdxInGroup = lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim);
    surfPtId.geomInstSlot = geomGroup.geomInstSlots[geomInstIdxInGroup];
    const GeometryInstance &geomInst = plp.s->geometryInstances[surfPtId.geomInstSlot];
    lightProb *= geomInstProb;
    if (geomInstProb == 0.0f) {
        surfPtId.instSlot = 0xFFFF'FFFF;
        return surfPtId;
    }

    float primProb;
    surfPtId.primIndex = geomInst.emitterPrimDist.sample(uPrim, &primProb);
    lightProb *= primProb;

    const shared::Triangle &tri = geomInst.triangles[surfPtId.primIndex];
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

    // Uniform sampling on unit triangle
    // A Low-Distortion Map Between Triangle and Square
    float bcB = 0.5f * up0;
    float bcC = 0.5f * up1;
    const float offset = bcC - bcB;
    if (offset > 0)
        bcC += offset;
    else
        bcB -= offset;

    surfPtId.bcB = bcB;
    surfPtId.bcC = bcC;

    const float recQuadArea = 1.0f / geomNormal.length();
    surfPt->asSurf.hypAreaPDensity = lightProb * (2.0f * recQuadArea);
    geomNormal *= recQuadArea;

    const float bcA = 1.0f - (bcB + bcC);

    surfPt->position = bcA * p[0] + bcB * p[1] + bcC * p[2];
    const Normal3D sn = normalize(
        inst.transform *
        (bcA * v[0].normal + bcB * v[1].normal + bcC * v[2].normal));
    surfPt->shadingFrame = ReferenceFrame(sn);
    surfPt->inMedium = false;
    surfPt->asSurf.geometricNormal = geomNormal;
    surfPt->asSurf.texCoord = bcA * v[0].texCoord + bcB * v[1].texCoord + bcC * v[2].texCoord;

    return surfPtId;
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(generateLightVertices)() {
    const uint32_t launchIndex = optixGetLaunchIndex().x;

    WavelengthSamples wls = plp.s->lvcBptPassInfo->wls;
    PCG32RNG rng = plp.s->ltRngBuffer[launchIndex];

    InteractionPoint lightPt;
    const SurfacePointIdentifier lightPtId = sampleLight(
        rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &lightPt);
    if (lightPtId.instSlot == 0xFFFF'FFFF) {
        plp.s->ltRngBuffer[launchIndex] = rng;
        return;
    }

    SampledSpectrum throughput;
    Point3D rayOrg;
    Vector3D rayDir;
    float dirPDens;
    bool deltaSampled = false;
    float cosTerm = 0.0f;
    float prevAreaPDens = 1.0f;
    bool prevDeltaSampled = false;
    float secondPrevAreaPDens;
    bool secondPrevDeltaSampled;
    float secondPrevPartialDenomMisWeight = 0.0f;
    float secondPrevProbRatioToFirst = 1.0f;
    {
        const GeometryInstance &geomInst = plp.s->geometryInstances[lightPtId.geomInstSlot];

        const TexCoord2D &tc = lightPt.asSurf.texCoord;
        const SurfaceMaterial &mat = plp.s->surfaceMaterials[geomInst.surfMatSlot];
        const float4 texValue = tex2DLod<float4>(mat.emittance, tc.u, tc.v, 0.0f);
        const SampledSpectrum Le0 = createTripletSpectrum(
            SpectrumType::LightSource, ColorSpace::Rec709_D65,
            texValue.x, texValue.y, texValue.z).evaluate(wls);

        const float areaPDens = plp.s->numLightPaths * lightPt.asSurf.hypAreaPDensity;
        throughput = Le0 / areaPDens;

        storeLightVertex(
            lightPtId, Point3D(), throughput,
            Vector3D(0, 0, 1), Vector3D(), 0,
            areaPDens, prevAreaPDens,
            secondPrevPartialDenomMisWeight, secondPrevProbRatioToFirst,
            deltaSampled, prevDeltaSampled, false, false, 0);

        secondPrevAreaPDens = prevAreaPDens;
        secondPrevDeltaSampled = prevDeltaSampled;
        prevAreaPDens = areaPDens;
        prevDeltaSampled = deltaSampled;

        rayOrg = lightPt.calcOffsetRayOrigin(true);

        SampledSpectrum Le1;
        if (mat.emitterType == uint32_t(EmitterType::Diffuse)) {
            Le1 = SampledSpectrum::One() / pi_v<float>;
            const Vector3D vOutLocal = cosineSampleHemisphere(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
            rayDir = lightPt.fromLocal(vOutLocal);
            dirPDens = vOutLocal.z / pi_v<float>;
            deltaSampled = false;
        }
        else {
            Le1 = SampledSpectrum::One();
            rayDir = lightPt.shadingFrame.normal;
            dirPDens = 1.0f;
            deltaSampled = true;
        }

        if (dirPDens == 0.0f) {
            plp.s->ltRngBuffer[launchIndex] = rng;
            return;
        }

        cosTerm = lightPt.calcDot(rayDir);
        throughput *= Le1 * (cosTerm / dirPDens);
    }

    float secondPrevRevAreaPDens = 1.0f;
    uint32_t pathLength = 0;
    while (true) {
        ++pathLength;

        SurfacePointIdentifier surfPtId = {};
        surfPtId.instSlot = 0xFFFF'FFFF;
        float hitDist = 1e+10f;
        ClosestRaySignature::trace(
            plp.f->travHandle,
            rayOrg, rayDir, 0.0f, 1e+10f, 0.0f,
            0xFF, OPTIX_RAY_FLAG_NONE,
            LvcBptRayType::Closest, maxNumRayTypes, LvcBptRayType::Closest,
            surfPtId.instSlot, surfPtId.geomInstSlot, surfPtId.primIndex,
            surfPtId.bcB, surfPtId.bcC, hitDist);

        bool volEventHappens = false;
        if (plp.f->enableVolume) {
            const float fpDist = -std::log(1.0f - rng.getFloat0cTo1o()) / plp.f->volumeDensity;
            if (fpDist < hitDist) {
                volEventHappens = true;
                hitDist = fpDist;
            }
        }

        if (surfPtId.instSlot == 0xFFFF'FFFF && !volEventHappens)
            break;

        InteractionPoint interPt;
        uint32_t surfMatSlot = 0xFFFF'FFFF;
        if (volEventHappens) {
            interPt.position = rayOrg + hitDist * rayDir;
            interPt.shadingFrame = ReferenceFrame(rayDir);
            interPt.inMedium = true;
        }
        else {
            const Instance &inst = plp.f->instances[surfPtId.instSlot];
            const GeometryInstance &geomInst = plp.s->geometryInstances[surfPtId.geomInstSlot];
            computeSurfacePoint<false>(inst, geomInst, surfPtId.primIndex, surfPtId.bcB, surfPtId.bcC, &interPt);

            surfMatSlot = geomInst.surfMatSlot;

            if (geomInst.normal) {
                const Normal3D modLocalNormal = geomInst.readModifiedNormal(
                    geomInst.normal, geomInst.normalDimInfo, interPt.asSurf.texCoord);
                applyBumpMapping(modLocalNormal, &interPt.shadingFrame);
            }
        }

        const Vector3D dirIn(-rayDir);
        const Vector3D dirInLocal = interPt.toLocal(dirIn);

        // JP: 現在のパスセグメントより2つ前のセグメントにおける部分的なMISウェイトが確定する。
        // EN: The partial MIS weight at the segment two segments before the current path segment can be determined.
        const bool secondLastSegIsValidConnection =
            (!prevDeltaSampled && !secondPrevDeltaSampled) &&
            pathLength > 2; // separately accumulate the ratio for the strategy with zero light vertices.
        const float probRatio = secondPrevRevAreaPDens / secondPrevAreaPDens;
        secondPrevPartialDenomMisWeight =
            pow2(probRatio) *
            (secondPrevPartialDenomMisWeight + (secondLastSegIsValidConnection ? 1 : 0));
        secondPrevProbRatioToFirst *= probRatio;

        const float lastDist2 = squaredDistance(rayOrg, interPt.position);
        //if (rwPayload->originIsInfinity)
        //    lastDist2 = 1;
        const float areaPDens = dirPDens * interPt.calcAbsDot(dirInLocal) / lastDist2;

        storeLightVertex(
            surfPtId, interPt.position, throughput,
            dirInLocal, dirIn, cosTerm / lastDist2,
            areaPDens, prevAreaPDens,
            secondPrevPartialDenomMisWeight, secondPrevProbRatioToFirst,
            deltaSampled, prevDeltaSampled,
            wls.singleIsSelected(), interPt.inMedium, pathLength);

        secondPrevAreaPDens = prevAreaPDens;
        secondPrevDeltaSampled = prevDeltaSampled;
        prevAreaPDens = areaPDens;
        prevDeltaSampled = deltaSampled;

        constexpr TransportMode transportMode = TransportMode::Importance;
        BSDF<transportMode, BSDFGrade::Bidirectional> bsdf;
        BSDFQuery bsdfQuery;
        if (volEventHappens) {
            bsdf.setup(plp.f->scatteringAlbedo);
            bsdfQuery = BSDFQuery();
        }
        else {
            const SurfaceMaterial &surfMat = plp.s->surfaceMaterials[surfMatSlot];
            bsdf.setup(surfMat, interPt.asSurf.texCoord, wls);
            bsdfQuery = BSDFQuery(
                dirInLocal, interPt.toLocal(interPt.asSurf.geometricNormal),
                transportMode, wls);
        }

        const BSDFSample bsdfSample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult bsdfResult;
        BSDFQueryReverseResult bsdfRevResult;
        const SampledSpectrum fsValue = bsdf.sampleF(bsdfQuery, bsdfSample, &bsdfResult, &bsdfRevResult);
        if (bsdfResult.dirPDensity == 0.0f)
            break; // sampling failed.
        rayDir = interPt.fromLocal(bsdfResult.dirLocal);
        secondPrevRevAreaPDens = bsdfRevResult.dirPDensity * cosTerm / lastDist2;

        const float dotSGN = interPt.calcAbsDot(rayDir);
        cosTerm = std::fabs(dotSGN);
        const SampledSpectrum localThroughput = fsValue * (cosTerm / bsdfResult.dirPDensity);

        // Russian roulette
        const float continueProb = std::fmin(localThroughput.importance(wls.selectedLambdaIndex()), 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            break;

        throughput *= localThroughput / continueProb;
        Assert(
            localThroughput.allNonNegativeFinite(),
            "tp: (" SPDFMT "), dotSGN: %g, dirP: %g",
            spdprint(localThroughput), dotSGN, bsdfResult.dirPDensity);

        rayOrg = interPt.calcOffsetRayOrigin(dotSGN > 0.0f);
        dirPDens = bsdfResult.dirPDensity;
        deltaSampled = bsdfResult.sampledType.isDelta();
        if constexpr (includeRrProbability) {
            dirPDens *= continueProb;
            const SampledSpectrum revLocalThroughput =
                bsdfRevResult.fsValue * interPt.calcAbsDot(dirIn) / bsdfRevResult.dirPDensity;
            const float revContinueProb =
                std::fmin(revLocalThroughput.importance(wls.selectedLambdaIndex()), 1.0f);
            secondPrevRevAreaPDens *= revContinueProb;
        }
    }

    plp.s->ltRngBuffer[launchIndex] = rng;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void decodeInteractionPoint(
    const LightPathVertex &lightVertex, InteractionPoint* const interPt)
{
    *interPt = {};
    interPt->inMedium = lightVertex.isInMedium;

    if (lightVertex.isInMedium) {
        interPt->position = lightVertex.positionInMedium;
        interPt->shadingFrame = ReferenceFrame(lightVertex.dirInInMedium);
    }
    else {
        const Instance &inst = plp.f->instances[lightVertex.instSlot];
        const GeometryInstance &geomInst = plp.s->geometryInstances[lightVertex.geomInstSlot];

        const Triangle &triangle = geomInst.triangles[lightVertex.primIndex];
        const Vertex &vA = geomInst.vertices[triangle.indices[0]];
        const Vertex &vB = geomInst.vertices[triangle.indices[1]];
        const Vertex &vC = geomInst.vertices[triangle.indices[2]];

        const StaticTransform &transform = inst.transform;

        const Vector3D e1 = transform * (vB.position - vA.position);
        const Vector3D e2 = transform * (vC.position - vA.position);
        Normal3D geometricNormal = cross(e1, e2);
        const float area = geometricNormal.length() / 2; // TODO: スケーリングの考慮。
        geometricNormal /= 2 * area;

        const float bcB = lightVertex.bcB;
        const float bcC = lightVertex.bcC;
        const float bcA = 1.0f - (bcB + bcC);

        Point3D position = bcA * vA.position + bcB * vB.position + bcC * vC.position;
        Normal3D shadingNormal = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        Vector3D tangent = bcA * vA.tangent + bcB * vB.tangent + bcC * vC.tangent;
        const TexCoord2D texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

        position = transform * position;
        shadingNormal = normalize(transform * shadingNormal);
        tangent = transform * tangent;
        if (!shadingNormal.allFinite() || !tangent.allFinite()) {
            Vector3D bitangent;
            shadingNormal = geometricNormal;
            shadingNormal.makeCoordinateSystem(&tangent, &bitangent);
        }

        // JP: 法線と接線が直交することを保証する。
        //     直交性の消失は重心座標補間によっておこる？
        // EN: guarantee the orthogonality between the normal and tangent.
        //     Orthogonality break might be caused by barycentric interpolation?
        const float dotNT = dot(shadingNormal, tangent);
        tangent = normalize(tangent - dotNT * Vector3D(shadingNormal));

        interPt->position = position;
        interPt->asSurf.geometricNormal = geometricNormal;
        interPt->asSurf.texCoord = texCoord;
        interPt->shadingFrame = ReferenceFrame(shadingNormal, tangent);
    }
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool testBinaryVisibility(
    const InteractionPoint &eInterPt, const InteractionPoint &lInterPt,
    Vector3D* const conRayDir, float* const conDist2)
{
    //Assert(eInterPt.atInfinity == false, "Shading point must be in finite region.");

    *conRayDir = lInterPt.position - eInterPt.position;
    *conDist2 = conRayDir->squaredLength();
    const float conDist = std::sqrt(*conDist2);
    *conRayDir /= conDist;

    const Point3D ePoint = eInterPt.calcOffsetRayOrigin(eInterPt.calcDot(*conRayDir));
    const Point3D lPoint = lInterPt.calcOffsetRayOrigin(!lInterPt.calcDot(*conRayDir));

    float tmax = FLT_MAX;
    if (/*!lInterPt.atInfinity*/true) // TODO: アドホックな調整ではなくoffsetRayOriginと一貫性のある形に。
        tmax = distance(lPoint, ePoint) * 0.9999f;

    float visibility = 1.0f;
    VisibilityRaySignature::trace(
        plp.f->travHandle,
        ePoint, *conRayDir, 0.0f, tmax, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        LvcBptRayType::Visibility, maxNumRayTypes, LvcBptRayType::Visibility,
        visibility);

    return visibility > 0;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void connectFromLens(
    const SampledSpectrum &throughput,
    const InteractionPoint &eInterPt, const WavelengthSamples &wls, const float uLVtx)
{
    const uint32_t lVtxIdx = mapPrimarySampleToDiscrete(uLVtx, plp.s->lvcBptPassInfo->numLightVertices);
    const LightPathVertex &lVtx = plp.s->lightVertexCache[lVtxIdx];
    const float lVtxProb = 1.0f / plp.s->lvcBptPassInfo->numLightVertices;

    InteractionPoint lInterPt;
    decodeInteractionPoint(lVtx, &lInterPt);

    Vector3D conRayDir;
    float conDist2;
    const bool binVisiblity = testBinaryVisibility(eInterPt, lInterPt, &conRayDir, &conDist2);
    if (!binVisiblity || !(debugPathLength == 0 || (lVtx.pathLength + 1) == debugPathLength))
        return;

    const float recConDist2 = 1.0f / conDist2;

    constexpr TransportMode lTransMode = TransportMode::Importance;
    BSDF<lTransMode, BSDFGrade::Bidirectional> lBsdf;
    if (lInterPt.inMedium) {
        lBsdf.setup(plp.f->scatteringAlbedo);
    }
    else {
        const uint32_t lMatSlot = plp.s->geometryInstances[lVtx.geomInstSlot].surfMatSlot;
        const SurfaceMaterial &lMat = plp.s->surfaceMaterials[lMatSlot];
        lBsdf.setup(
            lMat, lInterPt.asSurf.texCoord, wls,
            lVtx.pathLength == 0 ? BSDFBuildFlags::FromEDF : BSDFBuildFlags::None);
    }

    if (!lBsdf.hasNonDelta())
        return;

    const Vector3D lDirInLocal = lVtx.dirInLocal;
    const Normal3D lGeomNormal = lInterPt.toLocal(lInterPt.asSurf.geometricNormal);
    const BSDFQuery lBsdfQuery(lDirInLocal, lGeomNormal, lTransMode, wls);

    const Vector3D lConRayDirLocal = lInterPt.toLocal(-conRayDir);
    const Vector3D conRayDirLocalE = eInterPt.toLocal(conRayDir);

    const float lCosTerm = lInterPt.calcAbsDot(conRayDir);
    const float eCosTerm = eInterPt.calcAbsDot(conRayDir);
    float fracVis = 1.0f;
    if (plp.f->enableVolume)
        fracVis *= std::expf(-plp.f->volumeDensity * std::sqrt(conDist2));
    const float G = lCosTerm * eCosTerm * fracVis * recConDist2;
    float scalarConTerm = G / lVtxProb;
    if (lVtx.wlSelected)
        scalarConTerm *= SampledSpectrum::NumComponents();

    // on the light vertex
    SampledSpectrum lBackwardFs;
    const SampledSpectrum lForwardFs = lBsdf.evaluateF(lBsdfQuery, lConRayDirLocal, &lBackwardFs);
    float lBackwardDirPDens;
    // JP: Implicit Lens Sampling戦略は考えない。
    // EN: Don't consider the implicit lens sampling strategy.
    /*float forwardDirDensityL = */lBsdf.evaluatePDF(lBsdfQuery, lConRayDirLocal, &lBackwardDirPDens);
    //float forwardAreaDensityL = forwardDirDensityL * eCosTerm * recSquaredConDist;
    if constexpr (includeRrProbability) {
        const SampledSpectrum localThroughput = lBackwardFs *
            (std::fabs(dot(lBsdfQuery.dirLocal, lGeomNormal)) / lBackwardDirPDens);
        const float rrProb = std::fmin(localThroughput.importance(wls.selectedLambdaIndex()), 1.0f);
        lBackwardDirPDens *= rrProb;
    }
    float lBackwardAreaPDens = lBackwardDirPDens * lVtx.backwardConversionFactor;
    if (lVtx.prevDeltaSampled)
        lBackwardAreaPDens = 0;

    // on the eye vertex
    //SampledSpectrum eBackwardFs;
    //const SampledSpectrum eForwardFs = idf.evaluateDirectionalImportance(idfQuery, conRayDirLocalE);
    //float eForwardDirPDens = idf.evaluatePDF(idfQuery, conRayDirLocalE);
    SampledSpectrum eForwardFs;
    float eForwardDirPDens;
    float2 screenPos;
    bool onScreen;
    {
        const PerspectiveCamera &camera = plp.f->camera;
        float cosTerm;
        float dist2;
        onScreen = camera.calcScreenPosition(
            lInterPt.position, &eForwardFs, &screenPos, &eForwardDirPDens, &cosTerm, &dist2);
    }    
    const float eForwardAreaPDens = eForwardDirPDens * lCosTerm * recConDist2;
    //if (lInterPt.isPoint) // Delta function in the positional density (e.g. point light)
    //    eForwardAreaPDens = 0;

    // extend eye subpath, shorten light subpath.
    float lPartialDenomMisWeight;
    {
        float probRatioToFirst = lVtx.secondPrevProbRatioToFirst;

        const bool l2ndLastSegIsValidConnection =
            (!lVtx.deltaSampled && !lVtx.prevDeltaSampled) &&
            lVtx.pathLength > 1; // separately accumulate the ratio for the strategy with zero light vertices.
        const float lastTo2ndLastProbRatio = lBackwardAreaPDens / lVtx.prevProbDensity;
        lPartialDenomMisWeight =
            pow2(lastTo2ndLastProbRatio) *
            (lVtx.secondPrevPartialDenomMisWeight + (l2ndLastSegIsValidConnection ? 1 : 0));
        if (lVtx.pathLength > 0)
            probRatioToFirst *= lastTo2ndLastProbRatio;

        const bool l1stLastSegIsValidConnection =
            (/*lBsdf.hasNonDelta() &&*/ !lVtx.deltaSampled) &&
            lVtx.pathLength > 0; // separately accumulate the ratio for the strategy with zero light vertices.
        const float curTo1stLastProbRatio = eForwardAreaPDens / lVtx.probDensity;
        lPartialDenomMisWeight =
            pow2(curTo1stLastProbRatio) *
            (lPartialDenomMisWeight + (l1stLastSegIsValidConnection ? 1 : 0));
        probRatioToFirst *= curTo1stLastProbRatio;

        // JP: Implicit Light Sampling戦略にはLight Vertex Cacheからのランダムな選択確率は含まれない。
        // EN: Implicit light sampling strategy doesn't contain a probability to
        //     randomly select from the light vertex cache.
        lPartialDenomMisWeight += pow2(probRatioToFirst / lVtxProb);
    }

    const SampledSpectrum conTerm = lForwardFs * scalarConTerm * eForwardFs;
    const SampledSpectrum unweightedContribution = lVtx.flux * conTerm * throughput;

    const float recMisWeight = 1.0f + lPartialDenomMisWeight;
    const float misWeight = debugPathLength == 0 ? 1.0f / recMisWeight : 1.0f;
    const SampledSpectrum contribution = misWeight * unweightedContribution;
    const float2 pixel = make_float2(
        screenPos.x * plp.s->imageSize.x, screenPos.y * plp.s->imageSize.y);
    if (contribution.allFinite() && onScreen) {
        const uint2 pix(plp.s->imageSize.x * screenPos.x, plp.s->imageSize.y * screenPos.y);
        plp.s->ltTargetBuffer[pix].atomicAdd(wls, contribution);
    }
    else {
        printf(
            "Pass %u, (%u, %u - %u, %u), k%u (s%ut1): Not a finite value.\n",
            plp.f->numAccumFrames, optixGetLaunchIndex().x, optixGetLaunchIndex().y,
            static_cast<int32_t>(pixel.x), static_cast<int32_t>(pixel.y),
            lVtx.pathLength + 1, lVtx.pathLength + 1);
    }
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(eyePaths)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex());

    WavelengthSamples wls = plp.s->lvcBptPassInfo->wls;
    PCG32RNG rng = plp.s->rngBuffer.read(launchIndex);

    const PerspectiveCamera &camera = plp.f->camera;

    SampledSpectrum throughput;
    Point3D rayOrg;
    bool deltaSampled;
    {
        const Vector3D lensNormal = normalize(camera.orientation.toMatrix3x3() * Vector3D(0, 0, -1));
        const Vector3D lensTangent = normalize(camera.orientation.toMatrix3x3() * Vector3D(-1, 0, 0));

        InteractionPoint eInterPt = {};
        eInterPt.position = camera.position;
        eInterPt.inMedium = false;
        eInterPt.asSurf.geometricNormal = lensNormal;
        eInterPt.shadingFrame = ReferenceFrame(lensNormal, lensTangent);

        const SampledSpectrum We0 = SampledSpectrum::One();
        deltaSampled = true;
        const float areaPDens = 1.0f;

        const float imageSizeCorrFactor = plp.s->imageSize.x * plp.s->imageSize.y;
        throughput = We0 / (areaPDens * plp.s->lvcBptPassInfo->wlPDens * imageSizeCorrFactor);

        connectFromLens(throughput, eInterPt, wls, rng.getFloat0cTo1o());

        rayOrg = eInterPt.position;
    }

    plp.s->rngBuffer.write(launchIndex, rng);
}