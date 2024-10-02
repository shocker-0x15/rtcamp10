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

    CUDA_DEVICE_FUNCTION CUDA_INLINE LightPathVertex() {}
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
    LightDistribution dirLightInstDist;

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

CUDA_DEVICE_FUNCTION CUDA_INLINE const shared::BSDFProcedureSet &getBSDFProcedureSet(uint32_t slot) {
    return plp.s->bsdfProcedureSets[slot];
}

} // namespace rtc10::device

#endif // #if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)



#if defined(__CUDA_ARCH__)
#include "../common/spectrum_types.cpp"
#endif
