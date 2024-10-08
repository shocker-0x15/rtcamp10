#pragma once

#include "../common/common_renderer_types.h"

namespace rtc10::shared {

static constexpr float probToSampleEnvLight = 0.25f;



struct PathTracingRayType {
    enum Value {
        Closest,
        Visibility,
        NumTypes
    } value;

    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr PathTracingRayType(Value v = Closest) : value(v) {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE operator uint32_t() const {
        return static_cast<uint32_t>(value);
    }
};

struct LightTracingRayType {
    enum Value {
        Closest,
        Visibility,
        NumTypes
    } value;

    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr LightTracingRayType(Value v = Closest) : value(v) {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE operator uint32_t() const {
        return static_cast<uint32_t>(value);
    }
};

struct LvcBptRayType {
    enum Value {
        Closest,
        Visibility,
        NumTypes
    } value;

    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr LvcBptRayType(Value v = Closest) : value(v) {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE operator uint32_t() const {
        return static_cast<uint32_t>(value);
    }
};

constexpr uint32_t maxNumRayTypes = 2;



struct LightSample {
    SampledSpectrum emittance;
    Point3D position;
    Normal3D normal;
    uint32_t atInfinity : 1;
};



struct SurfacePointIdentifier {
    uint32_t instSlot;
    uint32_t geomInstSlot;
    uint32_t primIndex;
    float bcB;
    float bcC;
};



// TODO: キャッシュラインサイズの考慮。
struct LightPathVertex {
    union {
        struct {
            uint32_t instSlot;
            uint32_t geomInstSlot;
            uint32_t primIndex;
            float bcB, bcC;
            Vector3D dirInLocal;
        };
        struct {
            Point3D positionInMedium;
            Vector3D dirInInMedium;
        };
    };
    float probDensity;
    float prevProbDensity;
    float secondPrevPartialDenomMisWeight; // minus prob ratio to the strategy of implicit light sampling
    float secondPrevProbRatioToFirst; // prob ratio of implicit light sampling
    float backwardConversionFactor;
    SampledSpectrum flux;
    uint32_t wlSelected : 1;
    uint32_t deltaSampled : 1;
    uint32_t prevDeltaSampled : 1;
    uint32_t pathLength : 16;
    uint32_t isInMedium : 1;
    uint32_t isInObject : 1;

    CUDA_DEVICE_FUNCTION CUDA_INLINE LightPathVertex() {}
};



static constexpr uint32_t invalidNodeIndex = 0xFFFFFFFF;

// Reference: Efficient Incoherent Ray Traversal on GPUs Through Wide BVHs
// Similar to the data layout in this paper, but not the same.
template <uint32_t arity>
struct CompressedInternalNode_T {
    union ChildMeta {
        struct {
            uint8_t leafOffset;
        };
        uint8_t asUInt;

        CUDA_COMMON_FUNCTION CUDA_INLINE ChildMeta() : asUInt(0) {}

        CUDA_COMMON_FUNCTION CUDA_INLINE void setLeafOffset(uint32_t _leafOffset) {
            leafOffset = _leafOffset;
        }
        CUDA_COMMON_FUNCTION CUDA_INLINE uint8_t getLeafOffset() const {
            return leafOffset;
        }
    };

    Point3D quantBoxOrigin;
    uint8_t quantBoxExpScaleX;
    uint8_t quantBoxExpScaleY;
    uint8_t quantBoxExpScaleZ;
    uint8_t internalMask;
    uint32_t intNodeChildBaseIndex;
    uint32_t leafBaseIndex;
    ChildMeta childMetas[arity];
    uint8_t childQMinXs[arity];
    uint8_t childQMinYs[arity];
    uint8_t childQMinZs[arity];
    uint8_t childQMaxXs[arity];
    uint8_t childQMaxYs[arity];
    uint8_t childQMaxZs[arity];
    static constexpr uint32_t __paddingSize =
        arity == 8 ? 0 :
        arity == 4 ? 12 :
        arity == 2 ? 10 :
        1;
    uint8_t __padding[__paddingSize];

    CUDA_COMMON_FUNCTION CUDA_INLINE Vector3D __decodeQuantBoxScale() const {
        const Vector3D d(
            stc::bit_cast<float>(quantBoxExpScaleX << 23),
            stc::bit_cast<float>(quantBoxExpScaleY << 23),
            stc::bit_cast<float>(quantBoxExpScaleZ << 23));
        return d;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE AABB __decodeAabb(
        const uint8_t qMinX, const uint8_t qMinY, const uint8_t qMinZ,
        const uint8_t qMaxX, const uint8_t qMaxY, const uint8_t qMaxZ) const {
        const Vector3D d = __decodeQuantBoxScale();
        const Vector3D qMinPf = Vector3D(qMinX, qMinY, qMinZ);
        const Vector3D qMaxPf = Vector3D(qMaxX, qMaxY, qMaxZ);
        const AABB ret(
            quantBoxOrigin + qMinPf * d,
            quantBoxOrigin + qMaxPf * d);
        return ret;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE void setQuantizationAabb(const AABB &box) {
        quantBoxOrigin = box.minP;
        const Vector3D d = (box.maxP - box.minP) / 255;
        const auto calcExpScale = []
        (float s) {
            Assert(s >= 0, "s should not be negative.");
#if defined(__CUDA_ARCH__)
            const uint32_t us = __float_as_uint(s);
#else
            const uint32_t us = std::bit_cast<uint32_t>(s);
#endif
            return (us >> 23) + ((us & 0x7F'FFFF) ? 1 : 0);
        };
        quantBoxExpScaleX = calcExpScale(d.x);
        quantBoxExpScaleY = calcExpScale(d.y);
        quantBoxExpScaleZ = calcExpScale(d.z);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE AABB getQuantizationAabb() const {
        const Vector3D d = __decodeQuantBoxScale();
        const AABB ret(
            quantBoxOrigin,
            quantBoxOrigin + 255 * d);
        return ret;
    }

    /*CUDA_COMMON_FUNCTION CUDA_INLINE */void setChildAabb(uint32_t slot, const AABB &box) {
        const Vector3D recD = safeDivide(Vector3D(1.0f), __decodeQuantBoxScale());
        const auto qMinPf = static_cast<Point3D>((box.minP - quantBoxOrigin) * recD);
        const auto qMaxPf = static_cast<Point3D>((box.maxP - quantBoxOrigin) * recD);
        //const uint3 qMinP = make_uint3(qMinPf.toNative());
        //const uint3 qMaxP = min(make_uint3(qMaxPf.toNative()) + 1, 255);
        const uint3 qMinP = make_uint3(qMinPf.x, qMinPf.y, qMinPf.z);
        const uint3 qMaxP = make_uint3(
            std::min<uint32_t>(qMaxPf.x + 1, 255),
            std::min<uint32_t>(qMaxPf.y + 1, 255),
            std::min<uint32_t>(qMaxPf.z + 1, 255));
        childQMinXs[slot] = static_cast<uint8_t>(qMinP.x);
        childQMinYs[slot] = static_cast<uint8_t>(qMinP.y);
        childQMinZs[slot] = static_cast<uint8_t>(qMinP.z);
        childQMaxXs[slot] = static_cast<uint8_t>(qMaxP.x);
        childQMaxYs[slot] = static_cast<uint8_t>(qMaxP.y);
        childQMaxZs[slot] = static_cast<uint8_t>(qMaxP.z);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE void setInvalidChildBox(uint32_t slot) {
        childQMinXs[slot] = 255;
        childQMinYs[slot] = 255;
        childQMinZs[slot] = 255;
        childQMaxXs[slot] = 0;
        childQMaxYs[slot] = 0;
        childQMaxZs[slot] = 0;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE void setChildMeta(uint32_t slot, const ChildMeta &meta) {
        childMetas[slot] = meta;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE bool getChildIsValid(uint32_t slot) const {
        return childQMinXs[slot] != 255 || childQMaxXs[slot] != 0;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE AABB getChildAabb(uint32_t slot) const {
        return __decodeAabb(
            childQMinXs[slot], childQMinYs[slot], childQMinZs[slot],
            childQMaxXs[slot], childQMaxYs[slot], childQMaxZs[slot]);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE bool getChildIsLeaf(uint32_t slot) const {
        return ((internalMask >> slot) & 0b1) == 0;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t getInternalChildNumber(uint32_t slot) const {
        Assert(!getChildIsLeaf(slot), "only valid for an internal node child.");
        const uint32_t lowerMask = (1 << slot) - 1;
        return popcnt(internalMask & lowerMask);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t getLeafChildNumber(uint32_t slot) const {
        Assert(getChildIsLeaf(slot), "only valid for an leaf node child.");
        const uint32_t lowerMask = (1 << slot) - 1;
        return popcnt(~internalMask & lowerMask);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t getLeafOffset(uint32_t slot) const {
        Assert(getChildIsLeaf(slot), "Child offset is only valid for a leaf child.");
        return childMetas[slot].leafOffset;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE AABB getAabb() const {
        uint8_t qMinX = 255, qMinY = 255, qMinZ = 255;
        uint8_t qMaxX = 0, qMaxY = 0, qMaxZ = 0;
        for (uint32_t slot = 0; slot < arity; ++slot) {
            if (!getChildIsValid(slot))
                break;
            qMinX = stc::min(childQMinXs[slot], qMinX);
            qMinY = stc::min(childQMinYs[slot], qMinY);
            qMinZ = stc::min(childQMinZs[slot], qMinZ);
            qMaxX = stc::max(childQMaxXs[slot], qMaxX);
            qMaxY = stc::max(childQMaxYs[slot], qMaxY);
            qMaxZ = stc::max(childQMaxZs[slot], qMaxZ);
        }
        return __decodeAabb(
            qMinX, qMinY, qMinZ,
            qMaxX, qMaxY, qMaxZ);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE bool hasInternalChild(uint32_t index) const {
        if (intNodeChildBaseIndex == UINT32_MAX)
            return false;
        return index >= intNodeChildBaseIndex &&
            index < (intNodeChildBaseIndex + popcnt(internalMask));
    }
};
static_assert(sizeof(CompressedInternalNode_T<2>) == 48, "Unexpected sizeof(CompressedInternalNode_T<2>)");
static_assert(sizeof(CompressedInternalNode_T<4>) == 64, "Unexpected sizeof(CompressedInternalNode_T<4>)");
static_assert(sizeof(CompressedInternalNode_T<8>) == 80, "Unexpected sizeof(CompressedInternalNode_T<8>)");

template <uint32_t arity>
struct UncompressedInternalNode_T {
    union ChildMeta {
        struct {
            uint8_t leafOffset;
        };
        uint8_t asUInt;

        CUDA_COMMON_FUNCTION CUDA_INLINE ChildMeta() : asUInt(0) {}

        CUDA_COMMON_FUNCTION CUDA_INLINE void setLeafOffset(uint32_t _leafOffset) {
            leafOffset = _leafOffset;
        }
        CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t getLeafOffset() const {
            return leafOffset;
        }
    };

    uint32_t intNodeChildBaseIndex;
    uint32_t leafBaseIndex;
    AABB childAabbs[arity];
    ChildMeta childMetas[arity];
    uint8_t internalMask;
    static constexpr uint32_t __paddingSize =
        arity == 8 ? 15 :
        arity == 4 ? 3 :
        arity == 2 ? 5 :
        1;
    uint8_t __padding[__paddingSize];

    CUDA_COMMON_FUNCTION CUDA_INLINE void setQuantizationAabb(const AABB &box) {}
    CUDA_COMMON_FUNCTION CUDA_INLINE AABB getQuantizationAabb() const {
        AABB ret;
        for (uint32_t slot = 0; slot < arity; ++slot) {
            if (!getChildIsValid(slot))
                break;
            ret.unify(childAabbs[slot]);
        }
        return ret;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE void setChildAabb(uint32_t slot, const AABB &box) {
        childAabbs[slot] = box;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE void setInvalidChildBox(uint32_t slot) {
        childAabbs[slot] = AABB();
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE void setChildMeta(uint32_t slot, const ChildMeta &meta) {
        childMetas[slot] = meta;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE bool getChildIsValid(uint32_t slot) const {
        return childAabbs[slot].isValid();
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE AABB getChildAabb(uint32_t slot) const {
        return childAabbs[slot];
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE bool getChildIsLeaf(uint32_t slot) const {
        return ((internalMask >> slot) & 0b1) == 0;
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t getInternalChildNumber(uint32_t slot) const {
        Assert(!getChildIsLeaf(slot), "only valid for an internal node child.");
        const uint32_t lowerMask = (1 << slot) - 1;
        return popcnt(internalMask & lowerMask);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t getLeafChildNumber(uint32_t slot) const {
        Assert(getChildIsLeaf(slot), "only valid for an leaf node child.");
        const uint32_t lowerMask = (1 << slot) - 1;
        return popcnt(~internalMask & lowerMask);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t getLeafOffset(uint32_t slot) const {
        Assert(getChildIsLeaf(slot), "Child offset is only valid for a leaf child.");
        return childMetas[slot].leafOffset;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE AABB getAabb() const {
        return getQuantizationAabb();
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE bool hasInternalChild(uint32_t index) const {
        if (intNodeChildBaseIndex == UINT32_MAX)
            return false;
        return index >= intNodeChildBaseIndex &&
            index < (intNodeChildBaseIndex + popcnt(internalMask));
    }
};
static_assert(sizeof(UncompressedInternalNode_T<2>) == 64, "Unexpected sizeof(UncompressedInternalNode_T<2>)");
static_assert(sizeof(UncompressedInternalNode_T<4>) == 112, "Unexpected sizeof(UncompressedInternalNode_T<4>)");
static_assert(sizeof(UncompressedInternalNode_T<8>) == 224, "Unexpected sizeof(UncompressedInternalNode_T<8>)");

template <uint32_t arity>
using InternalNode_T = CompressedInternalNode_T<arity>;

struct PrimitiveReference {
    uint32_t storageIndex : 31;
    uint32_t isLeafEnd : 1;
};

struct TriangleStorage {
    Point3D pA;
    Point3D pB;
    Point3D pC;
    uint32_t geomIndex;
    uint32_t primIndex;
    uint32_t __padding;
};
static_assert(sizeof(TriangleStorage) == 48, "Unexpected sizeof(TriangleStorage)");

union ParentPointer {
    struct {
        uint32_t index : 29;
        uint32_t slot : 3;
    };
    uint32_t asUInt;
    ParentPointer() {}
    ParentPointer(const uint32_t v) : asUInt(v) {}
    ParentPointer(const uint32_t _index, const uint32_t _slot) : index(_index), slot(_slot) {}
};

template <uint32_t arity>
struct GeometryBVH_T {
    ROBuffer<InternalNode_T<arity>> intNodes;
    ROBuffer<TriangleStorage> triStorages;
    ROBuffer<PrimitiveReference> primRefs;
    ROBuffer<ParentPointer> parentPointers;
};

struct InstanceReference {
    Matrix3x3 rotToObj;
    Vector3D transToObj;
    Matrix3x3 rotFromObj;
    Vector3D transFromObj;
    uintptr_t bvhAddress;
    uint32_t nodeIndex;
    uint32_t instanceIndex;
    uint32_t userData;
    uint32_t __padding[3];
};

template <uint32_t arity>
struct InstanceBVH_T {
    ROBuffer<InternalNode_T<arity>> intNodes;
    ROBuffer<InstanceReference> instRefs;
    ROBuffer<ParentPointer> parentPointers;
};

struct HitObject {
    float dist;
    uint32_t instIndex;
    uint32_t instUserData;
    uint32_t geomIndex;
    uint32_t primIndex;
    float bcA;
    float bcB;
    float bcC;

    CUDA_COMMON_FUNCTION CUDA_INLINE bool isHit() const {
        return primIndex != UINT32_MAX;
    }
};



struct LvcBptPassInfo {
    WavelengthSamples wls;
    float wlPDens;
    uint32_t numLightVertices;
};

struct StaticPipelineLaunchParameters {
    DiscretizedSpectrumAlwaysSpectral::CMF DiscretizedSpectrum_xbar;
    DiscretizedSpectrumAlwaysSpectral::CMF DiscretizedSpectrum_ybar;
    DiscretizedSpectrumAlwaysSpectral::CMF DiscretizedSpectrum_zbar;
    float DiscretizedSpectrum_integralCMF;
#if RTC10_SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
    const UpsampledSpectrum::spectrum_grid_cell_t* UpsampledSpectrum_spectrum_grid;
    const UpsampledSpectrum::spectrum_data_point_t* UpsampledSpectrum_spectrum_data_points;
#elif RTC10_SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
    const float* UpsampledSpectrum_maxBrightnesses;
    const UpsampledSpectrum::PolynomialCoefficients* UpsampledSpectrum_coefficients_sRGB_D65;
    const UpsampledSpectrum::PolynomialCoefficients* UpsampledSpectrum_coefficients_sRGB_E;
#endif

    ROBuffer<BSDFProcedureSet> bsdfProcedureSets;
    ROBuffer<SurfaceMaterial> surfaceMaterials;
    ROBuffer<GeometryInstance> geometryInstances;
    ROBuffer<GeometryGroup> geometryGroups;

    int2 imageSize;
    optixu::NativeBlockBuffer2D<PCG32RNG> rngBuffer;
    RWBuffer<PCG32RNG> ltRngBuffer;
    optixu::BlockBuffer2D<DiscretizedSpectrum, 0> ltTargetBuffer;
    optixu::BlockBuffer2D<SpectrumStorage, 0> accumBuffer;

    // LVC-BPT
    RWBuffer<LightPathVertex> lightVertexCache;
    LvcBptPassInfo* lvcBptPassInfo;
    uint32_t numLightPaths;
};

struct PerFramePipelineLaunchParameters {
    ROBuffer<Instance> instances;

    EnvironmentalLight envLight;

    WorldDimInfo worldDimInfo;
    LightDistribution lightInstDist;

    optixu::NativeBlockBuffer2D<RGBSpectrum> outputBuffer;

    int2 mousePosition;
    float volumeDensity;
    float scatteringAlbedo;

    OptixTraversableHandle travHandle;
    PerspectiveCamera camera;
    uint32_t numAccumFrames;
    uint32_t enableEnvironmentalLight : 1;
    uint32_t enableVolume : 1;
    uint32_t enableDebugPrint : 1;
};

struct PipelineLaunchParameters {
    StaticPipelineLaunchParameters* s;
    PerFramePipelineLaunchParameters* f;
};



using ClosestRaySignature = optixu::PayloadSignature<
    uint32_t, uint32_t, uint32_t, float, float, float>;
using VisibilityRaySignature = optixu::PayloadSignature<float>;

} // namespace rtc10::shared



#if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM rtc10::shared::PipelineLaunchParameters plp;
#else
RT_PIPELINE_LAUNCH_PARAMETERS rtc10::shared::PipelineLaunchParameters plp;
#endif

namespace rtc10::device {

CUDA_DEVICE_FUNCTION CUDA_INLINE int2 getMousePosition() {
    return plp.f->mousePosition;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool getDebugPrintEnabled() {
    return plp.f->enableDebugPrint;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool isDebugPixel(const int2 launchIndex) {
    const int2 mousePos = getMousePosition();
    return launchIndex.x == mousePos.x && launchIndex.y == mousePos.y && getDebugPrintEnabled();
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool isDebugPixel(const uint2 launchIndex) {
    return isDebugPixel(make_int2(launchIndex.x, launchIndex.y));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE const shared::BSDFProcedureSet &getBSDFProcedureSet(uint32_t slot) {
    return plp.s->bsdfProcedureSets[slot];
}

} // namespace rtc10::device

#endif // #if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)



#if defined(__CUDA_ARCH__)
#include "../common/spectrum_types.cpp"
#endif
