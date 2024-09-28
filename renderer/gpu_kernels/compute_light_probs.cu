#define PURE_CUDA
#include "../../common/common_renderer_types.h"
#include "../renderer_shared.h" // only for debug for now

using namespace rtc10;
using namespace rtc10::shared;
using namespace rtc10::device;



CUDA_DEVICE_KERNEL void initializeWorldDimInfo(
    WorldDimInfo* worldDimInfo) {
    BoundingBox3DAsOrderedInt aabbAsInt = BoundingBox3D();
    *reinterpret_cast<BoundingBox3DAsOrderedInt*>(&worldDimInfo->aabb) = aabbAsInt;
}



CUDA_DEVICE_FUNCTION CUDA_INLINE float computeTriangleImportance(
    GeometryInstance* geomInst, uint32_t triIndex,
    const SurfaceMaterial* materialBuffer) {
    const SurfaceMaterial &mat = materialBuffer[geomInst->surfMatSlot];
    const Triangle &tri = geomInst->triangles[triIndex];
    const Vertex (&v)[3] = {
        geomInst->vertices[tri.indices[0]],
        geomInst->vertices[tri.indices[1]],
        geomInst->vertices[tri.indices[2]]
    };

    Normal3D normal = cross(v[1].position - v[0].position, v[2].position - v[0].position);
    float area = 0.5f * normal.length();

    const auto sample = []
    (CUtexObject texture, const TexCoord2D &texCoord) {
        float4 texValue = tex2DLod<float4>(texture, texCoord.u, texCoord.v, 0);
        return RGBSpectrum(texValue.x, texValue.y, texValue.z);
    };

    // TODO: もっと正確な、少なくとも保守的な推定の実装。テクスチャー空間中の面積に応じてMIPレベルを選択する？
    RGBSpectrum emittanceEstimate = RGBSpectrum::Zero();
    emittanceEstimate += sample(mat.emittance, v[0].texCoord);
    emittanceEstimate += sample(mat.emittance, v[1].texCoord);
    emittanceEstimate += sample(mat.emittance, v[2].texCoord);
    emittanceEstimate /= 3;

    float importance = emittanceEstimate.luminance() * area;
    Assert(rtc10::isfinite(importance), "imp: %g, area: %g", importance, area);
    //printf("area: %g, %p, %g\n", area, mat.emittance, importance);
    return importance;
}

CUDA_DEVICE_KERNEL void computeTriangleProbBuffer(
    GeometryInstance* geomInst, uint32_t numTriangles,
    const SurfaceMaterial* materialBuffer) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex == 0)
        geomInst->emitterPrimDist.setNumValues(numTriangles);
    if (linearIndex < numTriangles) {
        float importance = computeTriangleImportance(geomInst, linearIndex, materialBuffer);
        geomInst->emitterPrimDist.setWeightAt(linearIndex, importance);
    }
}



CUDA_DEVICE_KERNEL void computeGeomInstProbBuffer(
    GeometryGroup* geomGroup, uint32_t numGeomInsts,
    const SurfaceMaterial* surfMatBuffer,
    const GeometryInstance* geometryInstanceBuffer) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex == 0) {
        geomGroup->lightGeomInstDist.setNumValues(numGeomInsts);
        geomGroup->dirLightGeomInstDist.setNumValues(numGeomInsts);
    }
    if (linearIndex < numGeomInsts) {
        uint32_t slot = geomGroup->geomInstSlots[linearIndex];
        const GeometryInstance &geomInst = geometryInstanceBuffer[slot];
        const SurfaceMaterial &surfMat = surfMatBuffer[geomInst.surfMatSlot];
        const auto emitterType = static_cast<EmitterType>(surfMat.emitterType);
        float importance = geomInst.emitterPrimDist.integral();

        geomGroup->lightGeomInstDist.setWeightAt(
            linearIndex, emitterType == EmitterType::Diffuse ? importance : 0.0f);
        geomGroup->dirLightGeomInstDist.setWeightAt(
            linearIndex, emitterType == EmitterType::Directional ? importance : 0.0f);
    }
}



CUDA_DEVICE_FUNCTION CUDA_INLINE float computeInstImportance(
    const Instance &inst, const GeometryGroup* geomGroupBuffer,
    BoundingBox3D* instAabb) {
    Vector3D scale;
    inst.transform.mat.decompose(&scale, nullptr, nullptr);
    float uniformScale = scale.x;
    const GeometryGroup &geomGroup = geomGroupBuffer[inst.geomGroupSlot];
    float importance = pow2(uniformScale) * geomGroup.lightGeomInstDist.integral();
    *instAabb = inst.transform * geomGroup.aabb;
    //printf("scale: %g, groupSlot: %u, imp: %g\n", uniformScale, inst.geomGroupSlot, importance);
    return importance;
}

CUDA_DEVICE_KERNEL void computeInstProbBuffer(
    WorldDimInfo* worldDimInfo,
    DiscreteDistribution1D* lightInstDist, DiscreteDistribution1D* dirLightInstDist,
    uint32_t numInsts,
    const GeometryGroup* geomGroupBuffer, const Instance* instanceBuffer) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (linearIndex == 0) {
        lightInstDist->setNumValues(numInsts);
        dirLightInstDist->setNumValues(numInsts);
    }
    if (linearIndex < numInsts) {
        const Instance &inst = instanceBuffer[linearIndex];

        Vector3D scale;
        inst.transform.mat.decompose(&scale, nullptr, nullptr);
        float uniformScale = scale.x;
        const GeometryGroup &geomGroup = geomGroupBuffer[inst.geomGroupSlot];
        float difImportance = pow2(uniformScale) * geomGroup.lightGeomInstDist.integral();
        float dirImportance = pow2(uniformScale) * geomGroup.dirLightGeomInstDist.integral();
        BoundingBox3D instAabb = inst.transform * geomGroup.aabb;

        lightInstDist->setWeightAt(linearIndex, difImportance);
        dirLightInstDist->setWeightAt(linearIndex, dirImportance);

        CUDA_SHARED_MEM uint32_t sm_aabbMem[sizeof(BoundingBox3DAsOrderedInt) / sizeof(uint32_t)];
        auto &sm_aabb = *reinterpret_cast<BoundingBox3DAsOrderedInt*>(sm_aabbMem);
        if (threadIdx.x == 0)
            sm_aabb = BoundingBox3D();
        __syncthreads();
        atomicMinMax_block(&sm_aabb, instAabb);
        if (threadIdx.x == 0)
            atomicMinMax(
                reinterpret_cast<BoundingBox3DAsOrderedInt*>(&worldDimInfo->aabb),
                sm_aabb);
    }
}



CUDA_DEVICE_KERNEL void finalizeDiscreteDistribution1D(
    DiscreteDistribution1D* lightInstDist) {
    if (threadIdx.x != 0)
        return;

    lightInstDist->finalize();
}



CUDA_DEVICE_KERNEL void finalizeWorldDimInfo(
    WorldDimInfo* worldDimInfo, DiscreteDistribution1D* lightInstDist, DiscreteDistribution1D* dirLightInstDist) {
    if (threadIdx.x != 0)
        return;

    lightInstDist->finalize();
    dirLightInstDist->finalize();

    auto aabbAsInt = *reinterpret_cast<BoundingBox3DAsOrderedInt*>(&worldDimInfo->aabb);
    BoundingBox3D aabb = static_cast<BoundingBox3D>(aabbAsInt);

    worldDimInfo->aabb = aabb;
    worldDimInfo->center = aabb.calcCentroid();
    Vector3D d = aabb.maxP - worldDimInfo->center;
    worldDimInfo->radius = d.length();
}



CUDA_DEVICE_KERNEL void computeFirstMipOfEnvIBLImportanceMap(
    optixu::NativeBlockBuffer2D<float4> envMap, uint2 imageSize,
    HierarchicalImportanceMap* impMap, optixu::NativeBlockBuffer2D<float> dstMip) {
    uint2 pix = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                           blockDim.y * blockIdx.y + threadIdx.y);
    uint32_t mapDimX = nextPowerOf2(imageSize.x);
    uint32_t mapDimY = max(nextPowerOf2(imageSize.y), mapDimX >> 1);
    if (pix.x >= mapDimX || pix.y >= mapDimY)
        return;
    if (pix == make_uint2(0, 0))
        impMap->setDimensions(make_uint2(mapDimX, mapDimY));
    float importance = 0.0f;
    if (pix.x < imageSize.x && pix.y < imageSize.y) {
        float4 texValue = envMap.read(pix);
        RGBSpectrum radiance(texValue.x, texValue.y, texValue.z);
        float correction = pi_v<float> / 2 * std::sin(pi_v<float> * (pix.y + 0.5f) / imageSize.y);
        importance = radiance.luminance() * correction;
    }
    dstMip.write(pix, importance);
}

CUDA_DEVICE_KERNEL void computeMipOfImportanceMap(
    HierarchicalImportanceMap* impMap, uint32_t dstMipLevel,
    optixu::NativeBlockBuffer2D<float> srcMip,
    optixu::NativeBlockBuffer2D<float> dstMip) {
    uint32_t numMipLevels = impMap->calcNumMipLevels();
    if (dstMipLevel >= numMipLevels)
        return;

    uint2 srcDims = impMap->getDimensions() >> (dstMipLevel - 1);
    uint2 dstDims = max(impMap->getDimensions() >> dstMipLevel, make_uint2(1, 1));
    uint2 globalIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    uint2 ul = 2 * globalIndex;
    uint2 ur = ul + make_uint2(1, 0);
    uint2 ll = ul + make_uint2(0, 1);
    uint2 lr = ll + make_uint2(1, 0);
    float sum = 0.0f;
    sum += (ul.x < srcDims.x && ul.y < srcDims.y) ? srcMip.read(ul) : 0.0f;
    sum += (ur.x < srcDims.x && ur.y < srcDims.y) ? srcMip.read(ur) : 0.0f;
    sum += (ll.x < srcDims.x && ll.y < srcDims.y) ? srcMip.read(ll) : 0.0f;
    sum += (lr.x < srcDims.x && lr.y < srcDims.y) ? srcMip.read(lr) : 0.0f;
    if (globalIndex.x < dstDims.x && globalIndex.y < dstDims.y) {
        sum *= 0.25f;
        dstMip.write(globalIndex, sum);
        if (dstMipLevel == numMipLevels - 1) {
            float integral = sum * (1 << (2 * (numMipLevels - 1)));
            impMap->setIntegral(integral);
        }
    }
}

CUDA_DEVICE_KERNEL void testImportanceMap(
    uint2 imageSize,
    const HierarchicalImportanceMap* impMap,
    PCG32RNG* rngs,
    uint32_t* histogram) {
    uint32_t linearIndex = blockDim.x * blockIdx.x + threadIdx.x;
    PCG32RNG &rng = rngs[linearIndex];
    float u0 = rng.getFloat0cTo1o();
    float u1 = rng.getFloat0cTo1o();
    float prob;
    uint2 pix = impMap->sample(u0, u1, &u0, &u1, &prob);
    if (linearIndex < 512) {
        float prob1 = impMap->evaluate(pix);
        printf("sampled %g vs evaluated %g\n", prob, prob1);
    }
    atomicAdd(&histogram[imageSize.x * pix.y + pix.x], 1u);
}
