#pragma once

#include "basic_types.h"
#include "rgb_spectrum_types.h"
#include "spectrum_types.h"
#include "../common/utils/optixu_on_cudau.h"

// JP: Callable Programや関数ポインターによる動的な関数呼び出しを
//     無くした場合の性能を見たい場合にこのマクロを有効化する。
// EN: Enable this switch when you want to see performance
//     without dynamic function calls by callable programs or function pointers.
static constexpr bool useGenericBSDF = false;
//#define HARD_CODED_BSDF DichromaticBRDF
//#define HARD_CODED_BSDF SimplePBR_BRDF
#define HARD_CODED_BSDF LambertBRDF

#define RTC10_USE_SPECTRAL_RENDERING 1

namespace rtc10 {
#if RTC10_USE_SPECTRAL_RENDERING
    using WavelengthSamples = WavelengthSamplesTemplate<float, NumSpectralSamples>;
    using SampledSpectrum = SampledSpectrumTemplate<float, NumSpectralSamples>;
    using DiscretizedSpectrum = DiscretizedSpectrumTemplate<float, NumStrataForStorage>;
    using SpectrumStorage = SpectrumStorageTemplate<float, NumStrataForStorage>;
    using TripletSpectrum = UpsampledSpectrum;
#   define SPDFMT "%g, %g, %g, %g"
#   define spdprint(sp) (sp)[0], (sp)[1], (sp)[2], (sp)[3]
#else
    using WavelengthSamples = RGBWavelengthSamplesTemplate<float>;
    using SampledSpectrum = RGBSpectrumTemplate<float>;
    using DiscretizedSpectrum = RGBSpectrumTemplate<float>;
    using SpectrumStorage = RGBStorageTemplate<float>;
    using TripletSpectrum = RGBSpectrum;
#   define SPDFMT "%g, %g, %g"
#   define spdprint(sp) (sp).r, (sp).g, (sp).b
#endif

    using DiscretizedSpectrumAlwaysSpectral = DiscretizedSpectrumTemplate<float, NumStrataForStorage>;

    CUDA_DEVICE_FUNCTION CUDA_INLINE TripletSpectrum createTripletSpectrum(
        SpectrumType spectrumType, ColorSpace colorSpace, float e0, float e1, float e2) {
#if RTC10_USE_SPECTRAL_RENDERING
        return UpsampledSpectrum(spectrumType, colorSpace, e0, e1, e2);
#else
        if constexpr (renderingRGB == ColorSpace::Rec709_D65) {
            if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma ||
                colorSpace == ColorSpace::Rec709_D65)
                return RGBSpectrum(e0, e1, e2);
        }

        float XYZ[3];

        switch (colorSpace) {
        case ColorSpace::Rec709_D65_sRGBGamma: {
            e0 = sRGB_degamma(e0);
            e1 = sRGB_degamma(e1);
            e2 = sRGB_degamma(e2);
            // pass to Rec709 (D65)
            [[fallthrough]];
        }
        case ColorSpace::Rec709_D65: {
            float RGB[3] = { e0, e1, e2 };
            switch (spectrumType) {
            case SpectrumType::Reflectance:
            case SpectrumType::IndexOfRefraction:
            case SpectrumType::NA:
                transformTristimulus(mat_Rec709_E_to_XYZ, RGB, XYZ);
                break;
            case SpectrumType::LightSource:
                transformTristimulus(mat_Rec709_D65_to_XYZ, RGB, XYZ);
                break;
            default:
                Assert_ShouldNotBeCalled();
                break;
            }
            break;
        }
        case ColorSpace::XYZ: {
            XYZ[0] = e0;
            XYZ[1] = e1;
            XYZ[2] = e2;
            break;
        }
        case ColorSpace::xyY: {
            Assert(e0 >= 0.0f && e1 >= 0.0f && e0 <= 1.0f && e1 <= 1.0f && e2 >= 0.0f,
                   "xy should be in [0, 1], Y should not be negative.");
            if (e1 == 0) {
                XYZ[0] = XYZ[1] = XYZ[2] = 0;
                break;
            }
            float z = 1 - (e0 + e1);
            float b = e2 / e1;
            XYZ[0] = e0 * b;
            XYZ[1] = e2;
            XYZ[2] = z * b;
            break;
        }
        default:
            Assert_NotImplemented();
            break;
        }

        // JP: ここの変換で内部的に使っている色空間がRec.709の場合は、
        //     Rec.709のRGB入力に対しては入力値に戻しているだけになるが、ここでRec.2020などを使うこともできる。
        float RGB[3];
        transformToRenderingRGB(spectrumType, XYZ, RGB);
        return max(RGBSpectrum(RGB[0], RGB[1], RGB[2]), RGBSpectrum::Zero());
#endif
    }
}

#define DEBUG_MOUSE_POS_CONDITION \
    (device::getMousePosition() == make_int2(optixGetLaunchIndex()) && device::getDebugPrintEnabled())

#define PROCESS_DYNAMIC_FUNCTIONS \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromNormalMap), \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromNormalMap2ch), \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromHeightMap), \
    PROCESS_DYNAMIC_FUNCTION(setupLambertBRDF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_getSurfaceParameters), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluateDHReflectanceEstimate), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_matches), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_sampleF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_sampleFWithRev), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluateF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluateFWithRev), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluatePDF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluatePDFWithRev), \
    PROCESS_DYNAMIC_FUNCTION(setupSpecularBSDF), \
    PROCESS_DYNAMIC_FUNCTION(SpecularBSDF_getSurfaceParameters), \
    PROCESS_DYNAMIC_FUNCTION(SpecularBSDF_evaluateDHReflectanceEstimate), \
    PROCESS_DYNAMIC_FUNCTION(SpecularBSDF_matches), \
    PROCESS_DYNAMIC_FUNCTION(SpecularBSDF_sampleF), \
    PROCESS_DYNAMIC_FUNCTION(SpecularBSDF_sampleFWithRev), \
    PROCESS_DYNAMIC_FUNCTION(SpecularBSDF_evaluateF), \
    PROCESS_DYNAMIC_FUNCTION(SpecularBSDF_evaluateFWithRev), \
    PROCESS_DYNAMIC_FUNCTION(SpecularBSDF_evaluatePDF), \
    PROCESS_DYNAMIC_FUNCTION(SpecularBSDF_evaluatePDFWithRev), \
    PROCESS_DYNAMIC_FUNCTION(setupDichromaticBRDF), \
    PROCESS_DYNAMIC_FUNCTION(setupSimplePBR_BRDF), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_getSurfaceParameters), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_evaluateDHReflectanceEstimate), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_matches), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_sampleF), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_sampleFWithRev), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_evaluateF), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_evaluateFWithRev), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_evaluatePDF), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_evaluatePDFWithRev), \
    PROCESS_DYNAMIC_FUNCTION(ImageBasedEnvironmentalLight_sample), \
    PROCESS_DYNAMIC_FUNCTION(ImageBasedEnvironmentalLight_evaluate),

enum CallableProgram {
#define PROCESS_DYNAMIC_FUNCTION(Func) CallableProgram_ ## Func
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
    NumCallablePrograms
};

// JP: OptiXに関してホストコードが必要とするCallable Programsのエントリーポイント名。
// EN: Entry point names of callable programs required by the host code regarding OptiX.
constexpr const char* callableProgramEntryPoints[] = {
#define PROCESS_DYNAMIC_FUNCTION(Func) RT_DC_NAME_STR(#Func)
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
};

// JP: Pure CUDAに関してホストコードが必要とする関数ポインターのシンボル名。
// EN: Symbol names of function pointers required by the host code regarding pure CUDA.
#define CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR(name) "ptr_" #name
constexpr const char* callableProgramPointerNames[] = {
#define PROCESS_DYNAMIC_FUNCTION(Func) CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR(Func)
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
};
#undef CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR

#undef PROCESS_DYNAMIC_FUNCTIONS

// JP: Callable Programのインデックスから対応する関数ポインターへのマップ。
// EN: Map from callable programs to corresponding function pointers.
#if (defined(__CUDA_ARCH__) && defined(PURE_CUDA)) || defined(RTC10_Platform_CodeCompletion)
CUDA_CONSTANT_MEM void* c_callableToPointerMap[NumCallablePrograms];
#endif

#if defined(PURE_CUDA)
#   define CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(name) \
        extern "C" CUDA_DEVICE_MEM auto ptr_ ## name = RT_DC_NAME(name)
#else
#   define CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(name)
#endif



namespace rtc10::device {

CUDA_DEVICE_FUNCTION int2 getMousePosition();
CUDA_DEVICE_FUNCTION bool getDebugPrintEnabled();

}



namespace rtc10::shared {

static constexpr bool enableBufferOobCheck = true;
template <typename T>
using ROBuffer = cudau::ROBufferTemplate<T, enableBufferOobCheck>;
template <typename T>
using RWBuffer = cudau::RWBufferTemplate<T, enableBufferOobCheck>;

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint32_t mapPrimarySampleToDiscrete(
    float u01, uint32_t numValues, float* uRemapped = nullptr) {
    uint32_t idx = min(static_cast<uint32_t>(u01 * numValues), numValues - 1);
    if (uRemapped)
        *uRemapped = u01 * numValues - idx;
    return idx;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t sampleDiscrete(
    const float u01, float* const prob, float* const remapped, const float wA, const float wB) {
    const float wSum = wA + wB;
    const float u = wSum * u01;
    const float recWSum = 1.0f / wSum;
    if (u < wA) {
        *prob = wA * recWSum;
        *remapped = (u - 0.0f) / wA;
        return 0;
    }
    else {
        *prob = wB * recWSum;
        *remapped = (u - wA) / wB;
        return 1;
    }
}



class PCG32RNG {
    uint64_t state;

public:
    CUDA_COMMON_FUNCTION PCG32RNG() {}

    CUDA_COMMON_FUNCTION constexpr void setState(uint64_t _state) {
        state = _state;
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ uint32_t operator()() {
        uint64_t oldstate = state;
        // Advance internal state
        state = oldstate * 6364136223846793005ULL + 1;
        // Calculate output function (XSH RR), uses old state for max ILP
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ float getFloat0cTo1o() {
        uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
        return *reinterpret_cast<float*>(&fractionBits) - 1.0f;
    }
};



template <typename RealType>
struct AliasTableEntry {
    uint32_t secondIndex;
    RealType probToPickFirst;

    CUDA_COMMON_FUNCTION AliasTableEntry() {}
    CUDA_COMMON_FUNCTION constexpr AliasTableEntry(uint32_t _secondIndex, RealType _probToPickFirst) :
        secondIndex(_secondIndex), probToPickFirst(_probToPickFirst) {}
};

template <typename RealType>
struct AliasValueMap {
    RealType scaleForFirst;
    RealType scaleForSecond;
    RealType offsetForSecond;
};



template <typename RealType, bool useAliasTable>
class DiscreteDistribution1DTemplate;

template <typename RealType>
class DiscreteDistribution1DTemplate<RealType, false> {
    RealType* m_weights;
    RealType* m_CDF;
    RealType m_integral;
    uint32_t m_numValues;

public:
    DiscreteDistribution1DTemplate(
        RealType* weights, RealType* CDF, RealType integral, uint32_t numValues) :
        m_weights(weights), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}

    CUDA_COMMON_FUNCTION DiscreteDistribution1DTemplate() {}

    CUDA_COMMON_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped = nullptr) const {
        Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
        u *= m_integral;
        int idx = 0;
        for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
            if (idx + d >= m_numValues)
                continue;
            if (m_CDF[idx + d] <= u)
                idx += d;
        }
        Assert(idx < m_numValues, "Invalid Index!: %u >= %u, u: %g, integ: %g",
               idx, m_numValues, u, m_integral);
        if (remapped) {
            RealType lCDF = m_CDF[idx];
            RealType rCDF = m_integral;
            if (idx < m_numValues - 1)
                rCDF = m_CDF[idx + 1];
            *remapped = (u - lCDF) / (rCDF - lCDF);
            Assert(isfinite(*remapped), "Remapped value is not a finite value %g.",
                   *remapped);
    }
        *prob = m_weights[idx] / m_integral;
        return idx;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePMF(uint32_t idx) const {
        if (!m_weights || m_integral == 0.0f)
            return 0.0f;
        Assert(idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
        return m_weights[idx] / m_integral;
    }

    CUDA_COMMON_FUNCTION RealType integral() const { return m_integral; }

    CUDA_COMMON_FUNCTION uint32_t numValues() const { return m_numValues; }

    CUDA_COMMON_FUNCTION void setNumValues(uint32_t numValues) {
        m_numValues = numValues;
    }

    CUDA_COMMON_FUNCTION const RealType* weights() const {
        return m_weights;
    }

    CUDA_COMMON_FUNCTION const RealType* cdfs() const {
        return m_CDF;
    }

#if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION void setWeightAt(uint32_t index, RealType value) {
        m_weights[index] = value;
    }

    CUDA_DEVICE_FUNCTION void finalize() {
        uint32_t lastIndex = m_numValues - 1;
        m_integral = m_CDF[lastIndex] + m_weights[lastIndex];
        //printf("%g\n", m_integral);
    }
#endif
};

template <typename RealType>
class DiscreteDistribution1DTemplate<RealType, true> {
    RealType* m_weights;
    const AliasTableEntry<RealType>* m_aliasTable;
    const AliasValueMap<RealType>* m_valueMaps;
    RealType m_integral;
    uint32_t m_numValues;

public:
    DiscreteDistribution1DTemplate(
        RealType* weights, AliasTableEntry<RealType>* aliasTable, AliasValueMap<RealType>* valueMaps,
        RealType integral, uint32_t numValues) :
        m_weights(weights), m_aliasTable(aliasTable), m_valueMaps(valueMaps),
        m_integral(integral), m_numValues(numValues) {}

    CUDA_COMMON_FUNCTION DiscreteDistribution1DTemplate() {}

    CUDA_COMMON_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped = nullptr) const {
        Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
        uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
        const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
        const AliasValueMap<RealType> &valueMap = m_valueMaps[idx];
        if (u < entry.probToPickFirst) {
            if (remapped)
                *remapped = valueMap.scaleForFirst * u;
        }
        else {
            idx = entry.secondIndex;
            if (remapped)
                *remapped = valueMap.scaleForSecond * u + valueMap.offsetForSecond;
        }
        *prob = m_weights[idx] / m_integral;
        return idx;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePMF(uint32_t idx) const {
        if (!m_weights || m_integral == 0.0f)
            return 0.0f;
        Assert(idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
        return m_weights[idx] / m_integral;
    }

    CUDA_COMMON_FUNCTION RealType integral() const { return m_integral; }

    CUDA_COMMON_FUNCTION uint32_t numValues() const { return m_numValues; }

    CUDA_COMMON_FUNCTION uint32_t setNumValues(uint32_t numValues) {
        m_numValues = numValues;
    }

#if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION uint32_t setWeightAt(uint32_t index, RealType value) {
        m_weights[index] = value;
    }
#endif
};



template <typename RealType, bool useAliasTable>
class RegularConstantContinuousDistribution1DTemplate;

template <typename RealType>
class RegularConstantContinuousDistribution1DTemplate<RealType, false> {
    const RealType* m_weights;
    const RealType* m_CDF;
    RealType m_integral;
    uint32_t m_numValues;

public:
    RegularConstantContinuousDistribution1DTemplate(
        const RealType* weights, const RealType* CDF, RealType integral, uint32_t numValues) :
        m_weights(weights), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}

    CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution1DTemplate() {}

    CUDA_COMMON_FUNCTION RealType sample(RealType u, RealType* probDensity) const {
        Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
        u *= m_integral;
        int idx = 0;
        for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
            if (idx + d >= m_numValues)
                continue;
            if (m_CDF[idx + d] <= u)
                idx += d;
        }
        Assert(idx < m_numValues, "Invalid Index!: %u >= %u, u: %g, integ: %g",
               idx, m_numValues, u, m_integral);
        RealType lCDF = m_CDF[idx];
        RealType rCDF = m_integral;
        if (idx < m_numValues - 1)
            rCDF = m_CDF[idx + 1];
        RealType t = (u - lCDF) / (rCDF - lCDF);
        *probDensity = m_weights[idx] / m_integral;
        return (idx + t) / m_numValues;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType smp) const {
        Assert(smp >= 0 && smp < 1.0, "\"smp\": %g is out of range [0, 1).", smp);
        int32_t idx = min(m_numValues - 1, static_cast<uint32_t>(smp * m_numValues));
        return m_weights[idx] / m_integral;
    }

    CUDA_COMMON_FUNCTION RealType integral() const {
        return m_integral;
    }

    CUDA_COMMON_FUNCTION uint32_t numValues() const {
        return m_numValues;
    }
};

template <typename RealType>
class RegularConstantContinuousDistribution1DTemplate<RealType, true> {
    const RealType* m_PDF;
    const AliasTableEntry<RealType>* m_aliasTable;
    const AliasValueMap<RealType>* m_valueMaps;
    RealType m_integral;
    uint32_t m_numValues;

public:
    RegularConstantContinuousDistribution1DTemplate(
        const RealType* PDF, const AliasTableEntry<RealType>* aliasTable, const AliasValueMap<RealType>* valueMaps,
        RealType integral, uint32_t numValues) :
        m_PDF(PDF), m_aliasTable(aliasTable), m_valueMaps(valueMaps),
        m_integral(integral), m_numValues(numValues) {}

    CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution1DTemplate() {}

    CUDA_COMMON_FUNCTION RealType sample(RealType u, RealType* probDensity) const {
        Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
        uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
        const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
        const AliasValueMap<RealType> &valueMap = m_valueMaps[idx];
        RealType t;
        if (u < entry.probToPickFirst) {
            t = valueMap.scaleForFirst * u;
        }
        else {
            idx = entry.secondIndex;
            t = valueMap.scaleForSecond * u + valueMap.offsetForSecond;
        }
        *probDensity = m_PDF[idx];
        return (idx + t) / m_numValues;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType smp) const {
        Assert(smp >= 0 && smp < 1.0, "\"smp\": %g is out of range [0, 1).", smp);
        int32_t idx = min(m_numValues - 1, static_cast<uint32_t>(smp * m_numValues));
        return m_PDF[idx];
    }

    CUDA_COMMON_FUNCTION RealType integral() const {
        return m_integral;
    }

    CUDA_COMMON_FUNCTION uint32_t numValues() const {
        return m_numValues;
    }
};



template <typename RealType, bool useAliasTable>
class RegularConstantContinuousDistribution2DTemplate {
    using RegularConstantContinuousDistribution1D =
        RegularConstantContinuousDistribution1DTemplate<RealType, useAliasTable>;

    const RegularConstantContinuousDistribution1D* m_1DDists;
    RegularConstantContinuousDistribution1D m_top1DDist;

public:
    RegularConstantContinuousDistribution2DTemplate(
        const RegularConstantContinuousDistribution1D* _1DDists,
        const RegularConstantContinuousDistribution1D &top1DDist) :
        m_1DDists(_1DDists), m_top1DDist(top1DDist) {}

    CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution2DTemplate() {}

    CUDA_COMMON_FUNCTION void sample(
        RealType u0, RealType u1, RealType* d0, RealType* d1, RealType* probDensity) const {
        RealType topPDF;
        *d1 = m_top1DDist.sample(u1, &topPDF);
        uint32_t idx1D = mapPrimarySampleToDiscrete(*d1, m_top1DDist.numValues());
        *d0 = m_1DDists[idx1D].sample(u0, probDensity);
        *probDensity *= topPDF;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType d0, RealType d1) const {
        uint32_t idx1D = mapPrimarySampleToDiscrete(d1, m_top1DDist.numValues());
        return m_top1DDist.evaluatePDF(d1) * m_1DDists[idx1D].evaluatePDF(d0);
    }
};



CUDA_COMMON_FUNCTION CUDA_INLINE uint2 computeProbabilityTextureDimentions(uint32_t maxNumElems) {
#if !defined(__CUDA_ARCH__)
    using std::max;
#endif
    uint2 dims = make_uint2(max(nextPowerOf2(maxNumElems), 2u), 1u);
    while ((dims.x != dims.y) && (dims.x != 2 * dims.y)) {
        dims.x /= 2;
        dims.y *= 2;
    }
    return dims;
}

class HierarchicalImportanceMap {
    CUtexObject m_cuTexObj;
    unsigned int m_maxDimX : 16;
    unsigned int m_maxDimY : 16;
    unsigned int m_dimX : 16;
    unsigned int m_dimY : 16;
    float m_integral;

public:
    CUDA_COMMON_FUNCTION void setTexObject(CUtexObject texObj, uint2 maxDims) {
        m_cuTexObj = texObj;
        m_maxDimX = maxDims.x;
        m_maxDimY = maxDims.y;
    }

    CUDA_COMMON_FUNCTION void setDimensions(const uint2 &dims) {
        m_dimX = dims.x;
        m_dimY = dims.y;
    }

    CUDA_COMMON_FUNCTION uint2 getDimensions() const {
        return make_uint2(m_dimX, m_dimY);
    }

    CUDA_COMMON_FUNCTION float integral() const {
        if (m_cuTexObj == 0)
            return 0.0f;
        return m_integral;
    }

    CUDA_COMMON_FUNCTION uint32_t calcNumMipLevels() const {
        return nextPowOf2Exponent(m_dimX) + 1;
    }
    CUDA_COMMON_FUNCTION uint32_t calcMaxNumMipLevels() const {
        return nextPowOf2Exponent(m_maxDimX) + 1;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static uint2 compute2DFrom1D(uint2 dim, uint32_t index1D) {
        return make_uint2(index1D % dim.x, index1D / dim.y);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static uint32_t compute1DFrom2D(uint2 dim, const uint2 &index2D) {
        return index2D.y * dim.x + index2D.x;
    }

#if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION uint2 sample(
        float u0, float u1, float* remappedU0, float* remappedU1, float* prob) const {
        Assert(u0 >= 0 && u1 < 1, "\"u0\": %g must be in range [0, 1).", u0);
        Assert(u1 >= 0 && u1 < 1, "\"u1\": %g must be in range [0, 1).", u1);
        uint2 index2D = make_uint2(0, 0);
        uint32_t numMipLevels = calcNumMipLevels();
        *prob = 1;
        float2 recCurActualDims;
        {
            uint2 curActualDims = make_uint2(2, m_maxDimX > m_maxDimY ? 1 : 2);
            curActualDims <<= calcMaxNumMipLevels() - numMipLevels;
            recCurActualDims = make_float2(1.0f / curActualDims.x, 1.0f / curActualDims.y);
        }
        uint2 curDims = make_uint2(2, m_dimX > m_dimY ? 1 : 2);
        for (uint32_t mipLevel = numMipLevels - 2; mipLevel != UINT32_MAX; --mipLevel) {
            index2D = 2 * index2D;
            float2 tc = make_float2(index2D.x + 0.5f, index2D.y + 0.5f);
            float2 ll = tc + make_float2(0, 1);
            float2 lr = tc + make_float2(1, 1);
            float2 ur = tc + make_float2(1, 0);
            float2 ul = tc + make_float2(0, 0);
            float2 nll = ll * recCurActualDims;
            float2 nlr = lr * recCurActualDims;
            float2 nur = ur * recCurActualDims;
            float2 nul = ul * recCurActualDims;
            float vll = ll.y < curDims.y ?
                tex2DLod<float>(m_cuTexObj, nll.x, nll.y, mipLevel) : 0.0f;
            float vlr = (lr.x < curDims.x && lr.y < curDims.y) ?
                tex2DLod<float>(m_cuTexObj, nlr.x, nlr.y, mipLevel) : 0.0f;
            float vur = ur.x < curDims.x ?
                tex2DLod<float>(m_cuTexObj, nur.x, nur.y, mipLevel) : 0.0f;
            float vul = tex2DLod<float>(m_cuTexObj, nul.x, nul.y, mipLevel);

            float leftProb = vll + vul;
            float rightProb = vlr + vur;
            u0 *= (leftProb + rightProb);
            *prob *= 1.0f / (leftProb + rightProb);
            float upperProb;
            float lowerProb;
            if (u0 < leftProb) {
                u0 = (u0 - 0) / leftProb;
                upperProb = vul;
                lowerProb = vll;
            }
            else {
                index2D.x += 1;
                u0 = (u0 - leftProb) / rightProb;
                upperProb = vur;
                lowerProb = vlr;
            }

            u1 *= (upperProb + lowerProb);
            if (u1 < upperProb) {
                *prob *= upperProb;
                u1 = (u1 - 0) / upperProb;
            }
            else {
                index2D.y += 1;
                *prob *= lowerProb;
                u1 = (u1 - upperProb) / lowerProb;
            }

            recCurActualDims /= 2.0f;
            curDims *= 2;
        }
        *remappedU0 = u0;
        *remappedU1 = u1;

        return make_uint2(index2D.x, index2D.y);
    }

    CUDA_DEVICE_FUNCTION float evaluate(const uint2 &p) const {
        float2 tc = make_float2(p.x + 0.5f, p.y + 0.5f) / make_float2(m_maxDimX, m_maxDimY);
        float v = tex2DLod<float>(m_cuTexObj, tc.x, tc.y, 0.0f);
        float prob = v / m_integral;
        return prob;

        //uint32_t numMipLevels = calcNumMipLevels();
        //float2 recCurActualDims;
        //{
        //    uint2 curActualDims = make_uint2(m_maxDimX, m_maxDimY);
        //    recCurActualDims = make_float2(1.0f / curActualDims.x, 1.0f / curActualDims.y);
        //}
        //uint2 curDims = make_uint2(m_dimX, m_dimY);
        //float probDensity = 1.0f / (m_dimX * m_dimY);
        //uint2 index2D = make_uint2(p.x, p.y);
        //for (uint32_t mipLevel = 0; mipLevel < numMipLevels - 1; ++mipLevel) {
        //    uint2 baseIndex2D = index2D / 2 * 2;
        //    float2 tc = make_float2(baseIndex2D.x + 0.5f, baseIndex2D.y + 0.5f);
        //    float2 ll = tc + make_float2(0, 1);
        //    float2 lr = tc + make_float2(1, 1);
        //    float2 ur = tc + make_float2(1, 0);
        //    float2 ul = tc + make_float2(0, 0);
        //    float2 nll = ll * recCurActualDims;
        //    float2 nlr = lr * recCurActualDims;
        //    float2 nur = ur * recCurActualDims;
        //    float2 nul = ul * recCurActualDims;
        //    float vll = ll.y < curDims.y ?
        //        tex2DLod<float>(m_cuTexObj, nll.x, nll.y, mipLevel) : 0.0f;
        //    float vlr = (lr.x < curDims.x && lr.y < curDims.y) ?
        //        tex2DLod<float>(m_cuTexObj, nlr.x, nlr.y, mipLevel) : 0.0f;
        //    float vur = ur.x < curDims.x ?
        //        tex2DLod<float>(m_cuTexObj, nur.x, nur.y, mipLevel) : 0.0f;
        //    float vul = tex2DLod<float>(m_cuTexObj, nul.x, nul.y, mipLevel);

        //    probDensity /= (vll + vul + vlr + vur);
        //    if (index2D.x == baseIndex2D.x) {
        //        if (index2D.y == baseIndex2D.y)
        //            probDensity *= vul;
        //        else
        //            probDensity *= vll;
        //    }
        //    else {
        //        if (index2D.y == baseIndex2D.y)
        //            probDensity *= vur;
        //        else
        //            probDensity *= vlr;
        //    }

        //    index2D /= 2;
        //    recCurActualDims *= 2.0f;
        //    curDims /= 2;
        //}

        //return probDensity;
    }

    CUDA_DEVICE_FUNCTION void setIntegral(float v) {
        m_integral = v;
    }
#endif
};



using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float, false>;
using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float, false>;
using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float, false>;
using LightDistribution = DiscreteDistribution1D;



template <typename FuncType>
class DynamicFunction;

template <typename ReturnType, typename... ArgTypes>
class DynamicFunction<ReturnType(ArgTypes...)> {
    using Signature = ReturnType(*)(ArgTypes...);
    optixu::DirectCallableProgramID<ReturnType(ArgTypes...)> m_callableHandle;

public:
    CUDA_COMMON_FUNCTION DynamicFunction() {}
    CUDA_COMMON_FUNCTION constexpr DynamicFunction(uint32_t sbtIndex) : m_callableHandle(sbtIndex) {}

    CUDA_COMMON_FUNCTION constexpr explicit operator uint32_t() const {
        return static_cast<uint32_t>(m_callableHandle);
    }

#if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
#   if defined(PURE_CUDA)
        void* ptr = c_callableToPointerMap[static_cast<uint32_t>(m_callableHandle)];
        auto func = reinterpret_cast<Signature>(ptr);
        return func(args...);
#   else
        return m_callableHandle(args...);
#   endif
    }
#endif
};



struct TexDimInfo {
    uint32_t dimX : 14;
    uint32_t dimY : 14;
    uint32_t isNonPowerOfTwo : 1;
    uint32_t isBCTexture : 1;
    uint32_t isLeftHanded : 1; // for normal map
};



using ReadModifiedNormal = DynamicFunction<
    Normal3D(CUtexObject texture, TexDimInfo dimInfo, TexCoord2D texCoord)>;



struct DirectionType {
    enum InternalEnum : uint32_t {
        IE_LowFreq = 1 << 0,
        IE_HighFreq = 1 << 1,
        IE_Delta0D = 1 << 2,
        IE_Delta1D = 1 << 3,
        IE_NonDelta = IE_LowFreq | IE_HighFreq,
        IE_Delta = IE_Delta0D | IE_Delta1D,
        IE_AllFreq = IE_NonDelta | IE_Delta,

        IE_Reflection = 1 << 4,
        IE_Transmission = 1 << 5,
        IE_Emission = IE_Reflection,
        IE_Acquisition = IE_Reflection,
        IE_WholeSphere = IE_Reflection | IE_Transmission,

        IE_All = IE_AllFreq | IE_WholeSphere,

        IE_Dispersive = 1 << 6,

        IE_LowFreqReflection = IE_LowFreq | IE_Reflection,
        IE_LowFreqTransmission = IE_LowFreq | IE_Transmission,
        IE_LowFreqScattering = IE_LowFreqReflection | IE_LowFreqTransmission,
        IE_HighFreqReflection = IE_HighFreq | IE_Reflection,
        IE_HighFreqTransmission = IE_HighFreq | IE_Transmission,
        IE_HighFreqScattering = IE_HighFreqReflection | IE_HighFreqTransmission,
        IE_Delta0DReflection = IE_Delta0D | IE_Reflection,
        IE_Delta0DTransmission = IE_Delta0D | IE_Transmission,
        IE_Delta0DScattering = IE_Delta0DReflection | IE_Delta0DTransmission,
    };

    InternalEnum value;

    CUDA_DEVICE_FUNCTION CUDA_INLINE DirectionType() : value(InternalEnum(0)) { }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr DirectionType(InternalEnum v) : value(v) { }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr DirectionType &operator&=(const DirectionType &r) {
        value = static_cast<InternalEnum>(value & r.value);
        return *this;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr DirectionType &operator|=(const DirectionType &r) {
        value = static_cast<InternalEnum>(value | r.value);
        return *this;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr DirectionType flip() const {
        return static_cast<InternalEnum>(value ^ IE_WholeSphere);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool matches(DirectionType t) const {
        uint32_t res = value & t.value;
        return (res & IE_WholeSphere) && (res & IE_AllFreq);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool hasNonDelta() const {
        return value & IE_NonDelta;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool hasDelta() const {
        return value & IE_Delta;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool isDelta() const {
        return (value & IE_Delta) && !(value & IE_NonDelta);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool isReflection() const {
        return (value & IE_Reflection) && !(value & IE_Transmission);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool isTransmission() const {
        return !(value & IE_Reflection) && (value & IE_Transmission);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool isDispersive() const {
        return value & IE_Dispersive;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType LowFreq() {
        return IE_LowFreq;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType HighFreq() {
        return IE_HighFreq;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Delta0D() {
        return IE_Delta0D;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Delta1D() {
        return IE_Delta1D;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType NonDelta() {
        return IE_NonDelta;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Delta() {
        return IE_Delta;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType AllFreq() {
        return IE_AllFreq;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Reflection() {
        return IE_Reflection;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Transmission() {
        return IE_Transmission;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Emission() {
        return IE_Emission;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Acquisition() {
        return IE_Acquisition;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType WholeSphere() {
        return IE_WholeSphere;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType All() {
        return IE_All;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Dispersive() {
        return IE_Dispersive;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType LowFreqReflection() {
        return IE_LowFreqReflection;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType LowFreqTransmission() {
        return IE_LowFreqTransmission;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType LowFreqScattering() {
        return IE_LowFreqScattering;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType HighFreqReflection() {
        return IE_HighFreqReflection;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType HighFreqTransmission() {
        return IE_HighFreqTransmission;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType HighFreqScattering() {
        return IE_HighFreqScattering;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Delta0DReflection() {
        return IE_Delta0DReflection;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Delta0DTransmission() {
        return IE_Delta0DTransmission;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr DirectionType Delta0DScattering() {
        return IE_Delta0DScattering;
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool operator==(
    const DirectionType &l, const DirectionType &r) {
    return l.value == r.value;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool operator!=(
    const DirectionType &l, const DirectionType &r) {
    return l.value != r.value;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr DirectionType operator&(
    const DirectionType &l, const DirectionType &r) {
    DirectionType ret = l;
    ret &= r;
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr DirectionType operator|(
    const DirectionType &l, const DirectionType &r) {
    DirectionType ret = l;
    ret |= r;
    return ret;
}



enum class TransportMode {
    Radiance = 0,
    Importance,
};



struct SurfaceMaterial;

struct BSDFBuildFlags {
    enum Value {
        None = 0,
        FromEDF = 1 << 0,
    } value;

    CUDA_COMMON_FUNCTION constexpr BSDFBuildFlags(Value v = None) : value(v) {}

    CUDA_COMMON_FUNCTION operator uint32_t() const {
        return static_cast<uint32_t>(value);
    }
};

struct BSDFQuery {
    Vector3D dirLocal;
    Normal3D geometricNormalLocal;
    uint32_t transportMode : 2;
    uint32_t wlHint : 1;

    CUDA_COMMON_FUNCTION BSDFQuery() {}
    CUDA_COMMON_FUNCTION BSDFQuery(
        const Vector3D &dirL, const Normal3D &gNormL,
        TransportMode transMode, const WavelengthSamples &wls) :
        dirLocal(dirL), geometricNormalLocal(gNormL),
        transportMode(static_cast<uint32_t>(transMode)),
        wlHint(wls.selectedLambdaIndex()) {}
};

struct BSDFSample {
    float uDir[2];

    CUDA_COMMON_FUNCTION BSDFSample() {}
    CUDA_COMMON_FUNCTION BSDFSample(float uDir0, float uDir1) :
        uDir{ uDir0, uDir1 } {}
};

struct BSDFQueryResult {
    Vector3D dirLocal;
    float dirPDensity;
    DirectionType sampledType;

    CUDA_COMMON_FUNCTION BSDFQueryResult() {}
};

struct BSDFQueryReverseResult {
    SampledSpectrum fsValue;
    float dirPDensity;
};

using SetupBSDFBody = DynamicFunction<void(
    const SurfaceMaterial &matData, TexCoord2D texCoord,
    uint32_t* bodyData, BSDFBuildFlags flags)>;

using BSDFGetSurfaceParameters = DynamicFunction<void(
    const uint32_t* data,
    SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness)>;
using BSDFEvaluateDHReflectanceEstimate = DynamicFunction<
    SampledSpectrum(const uint32_t* data, const BSDFQuery &query)>;

using BSDFMatches = DynamicFunction<bool(
    const uint32_t* data, DirectionType dirType)>;
using BSDFSampleF = DynamicFunction<SampledSpectrum(
    const uint32_t* data, const BSDFQuery &query, const BSDFSample &sample,
    BSDFQueryResult* result)>;
using BSDFSampleFWithRev = DynamicFunction<SampledSpectrum(
    const uint32_t* data, const BSDFQuery &query, const BSDFSample &sample,
    BSDFQueryResult* result, BSDFQueryReverseResult* revResult)>;
using BSDFEvaluateF = DynamicFunction<SampledSpectrum(
    const uint32_t* data, const BSDFQuery &query, const Vector3D &vSampled)>;
using BSDFEvaluateFWithRev = DynamicFunction<SampledSpectrum(
    const uint32_t* data, const BSDFQuery &query, const Vector3D &vSampled,
    SampledSpectrum* revValue)>;
using BSDFEvaluatePDF = DynamicFunction<float(
    const uint32_t* data, const BSDFQuery &query, const Vector3D &vSampled)>;
using BSDFEvaluatePDFWithRev = DynamicFunction<float(
    const uint32_t* data, const BSDFQuery &query, const Vector3D &vSampled,
    float* revValue)>;

struct BSDFProcedureSet {
    SetupBSDFBody setupBSDFBody;
    BSDFGetSurfaceParameters getSurfaceParameters;
    BSDFEvaluateDHReflectanceEstimate evaluateDHReflectanceEstimate;
    BSDFMatches matches;
    BSDFSampleF sampleF;
    BSDFSampleFWithRev sampleFWithRev;
    BSDFEvaluateF evaluateF;
    BSDFEvaluateFWithRev evaluateFWithRev;
    BSDFEvaluatePDF evaluatePDF;
    BSDFEvaluatePDFWithRev evaluatePDFWithRev;
};



struct EDFQuery {
    CUDA_COMMON_FUNCTION EDFQuery() {}
};

struct EDFSample {
    float uDir[2];

    CUDA_COMMON_FUNCTION EDFSample() {}
    CUDA_COMMON_FUNCTION EDFSample(float uDir0, float uDir1) :
        uDir{ uDir0, uDir1 } {}
};

struct EDFQueryResult {
    Vector3D dirLocal;
    float dirPDensity;
    DirectionType sampledType;

    CUDA_COMMON_FUNCTION EDFQueryResult() {}
};



struct LambertianSurfaceMaterial {
    CUtexObject reflectance;
    TexDimInfo reflectanceDimInfo;
};

struct SpecularScatteringSurfaceMaterial {
    float iorExt;
    float abbeNumExt;
    float iorInt;
    float abbeNumInt;
};

struct DichromaticSurfaceMaterial {
    CUtexObject diffuse;
    CUtexObject specular;
    CUtexObject smoothness;
    TexDimInfo diffuseDimInfo;
    TexDimInfo specularDimInfo;
    TexDimInfo smoothnessDimInfo;
};

struct SimplePBRSurfaceMaterial {
    CUtexObject baseColor_opacity;
    CUtexObject occlusion_roughness_metallic;
    TexDimInfo baseColor_opacity_dimInfo;
    TexDimInfo occlusion_roughness_metallic_dimInfo;
};



enum class EmitterType {
    Diffuse = 0,
    Directional,
};

struct SurfaceMaterial {
    uint32_t body[12];
    CUtexObject emittance;
    uint32_t bsdfProcSetSlot;
    SetupBSDFBody setupBSDFBody; // shortcut
    uint32_t emitterType : 2;
};



using EnvLightSample = DynamicFunction<
    SampledSpectrum(const uint32_t* data, const WavelengthSamples &wls,
                    float uDir0, float uDir1, float* phi, float* theta, float* dirPDensity)>;
using EnvLightEvaluate = DynamicFunction<
    SampledSpectrum(const uint32_t* data, const WavelengthSamples &wls,
                    float phi, float theta, float* hypDirPDensity)>;



struct ImageBasedEnvironmentalLight {
    CUtexObject texObj;
    HierarchicalImportanceMap importanceMap;
    uint32_t imageWidth : 16;
    uint32_t imageHeight : 16;

#if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION SampledSpectrum sample(
        const WavelengthSamples &wls,
        float uDir0, float uDir1, float* phi, float* theta, float* dirPDensity) const;
    CUDA_DEVICE_FUNCTION SampledSpectrum evaluate(
        const WavelengthSamples &wls,
        float phi, float theta, float* hypDirPDensity) const;
#endif
};



struct EnvironmentalLight {
    uint32_t body[sizeof(ImageBasedEnvironmentalLight) / sizeof(uint32_t)];

    EnvLightSample envLightSample;
    EnvLightEvaluate envLightEvaluate;
    float powerCoeff;
    float rotation;

#if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION SampledSpectrum sample(
        const WavelengthSamples &wls,
        float uDir0, float uDir1, Vector3D* dir, float* dirPDensity) const {
        float phi, theta;
        SampledSpectrum radiance = powerCoeff * envLightSample(body, wls, uDir0, uDir1, &phi, &theta, dirPDensity);
        Assert(radiance.allNonNegativeFinite() && isfinite(*dirPDensity),
               "ImageBasedEnvironmentalLight::sample: u: (%g, %g), phi: %g, theta: %g, dirPD: %g",
               uDir0, uDir1, phi, theta, *dirPDensity);
        float posPhi = phi - rotation;
        posPhi = posPhi - floorf(posPhi / (2 * pi_v<float>)) * 2 * pi_v<float>;
        phi = posPhi;
        *dir = Vector3D::fromPolarYUp(phi, theta);
        return radiance;
    }

    CUDA_DEVICE_FUNCTION SampledSpectrum evaluate(
        const WavelengthSamples &wls,
        const Vector3D dir, float* hypDirPDensity) const {
        float posPhi, theta;
        dir.toPolarYUp(&theta, &posPhi);
        float phi = posPhi + rotation;
        phi = phi - floorf(phi / (2 * pi_v<float>)) * 2 * pi_v<float>;
        SampledSpectrum radiance = powerCoeff * envLightEvaluate(body, wls, phi, theta, hypDirPDensity);
        Assert(radiance.allNonNegativeFinite() && isfinite(*hypDirPDensity),
               "ImageBasedEnvironmentalLight::evaluate: phi: %g, theta: %g, hypDirPD: %g",
               phi, theta, *hypDirPDensity);
        return radiance;
    }
#endif
};



struct PerspectiveCamera {
    float aspect;
    float fovY;
    Point3D position;
    Quaternion orientation;

    CUDA_COMMON_FUNCTION CUDA_INLINE void sampleRay(
        const float2 &screenPos,
        Point3D* const rayOrg, SampledSpectrum* const We0,
        float* const areaPDens,
        Vector3D* const rayDir, SampledSpectrum* const We1,
        float* const dirPDens, float* const cosTerm) const
    {
        const float vh = 2 * std::tan(fovY * 0.5f);
        const float vw = aspect * vh;
        const float sensorArea = vw * vh; // normalized

        *rayOrg = position;
        *We0 = SampledSpectrum::One();
        *areaPDens = 1.0f;

        const Vector3D localRayDir = normalize(Vector3D(
            vw * (-0.5f + screenPos.x), vh * (0.5f - screenPos.y), -1));
        *rayDir = orientation.toMatrix3x3() * localRayDir;
        *We1 = SampledSpectrum::One();
        *cosTerm = -localRayDir.z;
        *dirPDens = 1 / (pow3(*cosTerm) * sensorArea);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE bool calcScreenPosition(
        const Point3D &posInWorld,
        SampledSpectrum* const We, float2* const screenPos,
        float* const dirPDens, float* const cosTerm, float* const dist2) const
    {
        const Vector3D rayDir = posInWorld - position;
        const Matrix3x3 invOri = conjugate(orientation).toMatrix3x3();
        Vector3D localRayDir = invOri * rayDir;
        if (localRayDir.z >= 0.0f) {
            *We = SampledSpectrum::Zero();
            *dirPDens = 0.0f;
            return false;
        }
        *dist2 = rayDir.squaredLength();
        localRayDir /= std::sqrt(*dist2);

        const float vh = 2 * std::tan(fovY / 2);
        const float vw = aspect * vh;
        const float sensorArea = vw * vh; // normalized

        *cosTerm = -localRayDir.z;
        const float2 posAtZ1 = make_float2(localRayDir.x / *cosTerm, localRayDir.y / *cosTerm);
        *screenPos = make_float2(0.5f + posAtZ1.x / vw, 0.5f - posAtZ1.y / vh);
        if (screenPos->x < 0 || screenPos->x >= 1.0f ||
            screenPos->y < 0 || screenPos->y >= 1.0f)
        {
            *We = SampledSpectrum::Zero();
            *dirPDens = 0.0f;
            return false;
        }
        *We = SampledSpectrum::One(); // We0 * We1
        *dirPDens = 1 / (pow3(*cosTerm) * sensorArea);

        return true;
    }
};



struct Vertex {
    Point3D position;
    Normal3D normal;
    Vector3D tangent;
    TexCoord2D texCoord;
};

struct Triangle {
    uint32_t indices[3];
};

struct GeometryInstance {
    ROBuffer<Vertex> vertices;
    ROBuffer<Triangle> triangles;
    CUtexObject normal;
    TexDimInfo normalDimInfo;
    ReadModifiedNormal readModifiedNormal;
    BoundingBox3D aabb;
    LightDistribution emitterPrimDist;
    uint32_t surfMatSlot;
};

struct StaticTransform {
    Matrix4x4 mat;
    Matrix4x4 invMat;

    CUDA_COMMON_FUNCTION StaticTransform() {}
    CUDA_COMMON_FUNCTION /*constexpr*/ StaticTransform(const Matrix4x4 &_mat) :
        mat(_mat), invMat(invert(_mat)) {}
    CUDA_COMMON_FUNCTION constexpr StaticTransform(
        const Matrix4x4 &_mat, const Matrix4x4 _invMat) :
        mat(_mat), invMat(_invMat) {}

    CUDA_COMMON_FUNCTION constexpr Vector3D operator*(const Vector3D &v) const {
        return mat * v;
    }
    CUDA_COMMON_FUNCTION constexpr Vector4D operator*(const Vector4D &v) const {
        return mat * v;
    }
    CUDA_COMMON_FUNCTION constexpr Point3D operator*(const Point3D &p) const {
        return mat * p;
    }
    CUDA_COMMON_FUNCTION constexpr Normal3D operator*(const Normal3D &n) const {
        // The length of the normal is changed if the transform has scaling, so it requires normalization.
        return Normal3D(invMat.m00 * n.x + invMat.m10 * n.y + invMat.m20 * n.z,
                        invMat.m01 * n.x + invMat.m11 * n.y + invMat.m21 * n.z,
                        invMat.m02 * n.x + invMat.m12 * n.y + invMat.m22 * n.z);
    }
    CUDA_COMMON_FUNCTION constexpr BoundingBox3D operator*(const BoundingBox3D &b) const { return mat * b; }


    CUDA_COMMON_FUNCTION constexpr Vector3D mulInv(const Vector3D& v) const {
        return invMat * v;
    }
    CUDA_COMMON_FUNCTION constexpr Vector4D mulInv(const Vector4D& v) const {
        return invMat * v;
    }
    CUDA_COMMON_FUNCTION constexpr Point3D mulInv(const Point3D& p) const {
        return invMat * p;
    }
    CUDA_COMMON_FUNCTION constexpr Normal3D mulInv(const Normal3D& n) const {
        // The length of the normal is changed if the transform has scaling, so it requires normalization.
        return Normal3D(mat.m00 * n.x + mat.m10 * n.y + mat.m20 * n.z,
                        mat.m01 * n.x + mat.m11 * n.y + mat.m21 * n.z,
                        mat.m02 * n.x + mat.m12 * n.y + mat.m22 * n.z);
    }
    CUDA_COMMON_FUNCTION constexpr BoundingBox3D mulInv(const BoundingBox3D &b) const {
        return invMat * b;
    }
};

struct GeometryGroup {
    ROBuffer<uint32_t> geomInstSlots;
    LightDistribution lightGeomInstDist;
    LightDistribution dirLightGeomInstDist;
    BoundingBox3D aabb;
};

struct Instance {
    StaticTransform transform;
    StaticTransform prevTransform;
    float uniformScale;
    uint32_t geomGroupSlot;
};



struct WorldDimInfo {
    BoundingBox3D aabb;
    Point3D center;
    float radius;
};



struct Point3DAsOrderedInt {
    int32_t x, y, z;

    CUDA_COMMON_FUNCTION Point3DAsOrderedInt() : x(0), y(0), z(0) {
    }
    CUDA_COMMON_FUNCTION Point3DAsOrderedInt(const Point3D &v) :
        x(floatToOrderedInt(v.x)), y(floatToOrderedInt(v.y)), z(floatToOrderedInt(v.z)) {
    }

    CUDA_COMMON_FUNCTION Point3DAsOrderedInt& operator=(const Point3DAsOrderedInt &v) {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION Point3DAsOrderedInt& operator=(const volatile Point3DAsOrderedInt &v) {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile Point3DAsOrderedInt& operator=(const Point3DAsOrderedInt &v) volatile {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile Point3DAsOrderedInt& operator=(const volatile Point3DAsOrderedInt &v) volatile {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    CUDA_COMMON_FUNCTION explicit operator Point3D() const {
        return Point3D(orderedIntToFloat(x), orderedIntToFloat(y), orderedIntToFloat(z));
    }
    CUDA_COMMON_FUNCTION explicit operator Point3D() const volatile {
        return Point3D(orderedIntToFloat(x), orderedIntToFloat(y), orderedIntToFloat(z));
    }
};

struct BoundingBox3DAsOrderedInt {
    Point3DAsOrderedInt minP;
    Point3DAsOrderedInt maxP;

    CUDA_COMMON_FUNCTION BoundingBox3DAsOrderedInt(const BoundingBox3D &aabb) :
        minP(aabb.minP), maxP(aabb.maxP) {}

    CUDA_COMMON_FUNCTION BoundingBox3DAsOrderedInt& operator=(const BoundingBox3DAsOrderedInt &v) {
        minP = v.minP;
        maxP = v.maxP;
        return *this;
    }
    CUDA_COMMON_FUNCTION BoundingBox3DAsOrderedInt& operator=(const volatile BoundingBox3DAsOrderedInt &v) {
        minP = v.minP;
        maxP = v.maxP;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile BoundingBox3DAsOrderedInt& operator=(const BoundingBox3DAsOrderedInt &v) volatile {
        minP = v.minP;
        maxP = v.maxP;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile BoundingBox3DAsOrderedInt& operator=(const volatile BoundingBox3DAsOrderedInt &v) volatile {
        minP = v.minP;
        maxP = v.maxP;
        return *this;
    }

    CUDA_COMMON_FUNCTION explicit operator BoundingBox3D() const {
        return BoundingBox3D(static_cast<Point3D>(minP), static_cast<Point3D>(maxP));
    }
    CUDA_COMMON_FUNCTION explicit operator BoundingBox3D() const volatile {
        return BoundingBox3D(static_cast<Point3D>(minP), static_cast<Point3D>(maxP));
    }
};

struct RGBSpectrumAsOrderedInt {
    int32_t r, g, b;

    CUDA_COMMON_FUNCTION RGBSpectrumAsOrderedInt() : r(0), g(0), b(0) {
    }
    CUDA_COMMON_FUNCTION RGBSpectrumAsOrderedInt(const RGBSpectrum &v) :
        r(floatToOrderedInt(v.r)), g(floatToOrderedInt(v.g)), b(floatToOrderedInt(v.b)) {
    }

    CUDA_COMMON_FUNCTION RGBSpectrumAsOrderedInt& operator=(const RGBSpectrumAsOrderedInt &v) {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION RGBSpectrumAsOrderedInt& operator=(const volatile RGBSpectrumAsOrderedInt &v) {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile RGBSpectrumAsOrderedInt& operator=(const RGBSpectrumAsOrderedInt &v) volatile {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION volatile RGBSpectrumAsOrderedInt& operator=(const volatile RGBSpectrumAsOrderedInt &v) volatile {
        r = v.r;
        g = v.g;
        b = v.b;
        return *this;
    }

    CUDA_COMMON_FUNCTION explicit operator RGBSpectrum() const {
        return RGBSpectrum(orderedIntToFloat(r), orderedIntToFloat(g), orderedIntToFloat(b));
    }
    CUDA_COMMON_FUNCTION explicit operator RGBSpectrum() const volatile {
        return RGBSpectrum(orderedIntToFloat(r), orderedIntToFloat(g), orderedIntToFloat(b));
    }
};

} // namespace rtc10::shared



#if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)

namespace rtc10::device {

static constexpr float RayEpsilon = 1e-4;

CUDA_DEVICE_FUNCTION const shared::BSDFProcedureSet &getBSDFProcedureSet(uint32_t slot);



CUDA_DEVICE_FUNCTION CUDA_INLINE void concentricSampleDisk(float u0, float u1, float* dx, float* dy) {
    float r, theta;
    float sx = 2 * u0 - 1;
    float sy = 2 * u1 - 1;

    if (sx == 0 && sy == 0) {
        *dx = 0;
        *dy = 0;
        return;
    }
    if (sx >= -sy) { // region 1 or 2
        if (sx > sy) { // region 1
            r = sx;
            theta = sy / sx;
        }
        else { // region 2
            r = sy;
            theta = 2 - sx / sy;
        }
    }
    else { // region 3 or 4
        if (sx > sy) {/// region 4
            r = -sy;
            theta = 6 + sx / sy;
        }
        else {// region 3
            r = -sx;
            theta = 4 + sy / sx;
        }
    }
    theta *= pi_v<float> / 4;
    *dx = r * cos(theta);
    *dy = r * sin(theta);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D cosineSampleHemisphere(float u0, float u1) {
    float x, y;
    concentricSampleDisk(u0, u1, &x, &y);
    return Vector3D(x, y, std::sqrt(std::fmax(0.0f, 1.0f - x * x - y * y)));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D uniformSampleSphere(float u0, float u1) {
    float phi = 2 * pi_v<float> * u1;
    float theta = std::acos(1 - 2 * u0);
    return Vector3D::fromPolarZUp(phi, theta);
}



// JP: 自己交叉回避のためにレイの原点にオフセットを付加する。
// EN: Add an offset to a ray origin to avoid self-intersection.
CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr Point3D offsetRayOriginNaive(
    const Point3D &p, const Normal3D &geometricNormal) {
    return p + RayEpsilon * geometricNormal;
}

// Reference:
// Chapter 6. A Fast and Robust Method for Avoiding Self-Intersection, Ray Tracing Gems, 2019
CUDA_DEVICE_FUNCTION CUDA_INLINE /*constexpr*/ Point3D offsetRayOrigin(
    const Point3D &p, const Normal3D &geometricNormal) {
    constexpr float kOrigin = 1.0f / 32.0f;
    constexpr float kFloatScale = 1.0f / 65536.0f;
    constexpr float kIntScale = 256.0f;

    int32_t offsetInInt[] = {
        static_cast<int32_t>(kIntScale * geometricNormal.x),
        static_cast<int32_t>(kIntScale * geometricNormal.y),
        static_cast<int32_t>(kIntScale * geometricNormal.z)
    };

    // JP: 数学的な衝突点の座標と、実際の座標の誤差は原点からの距離に比例する。
    //     intとしてオフセットを加えることでスケール非依存に適切なオフセットを加えることができる。
    // EN: The error of the actual coorinates of the intersection point to the mathematical one is proportional to the distance to the origin.
    //     Applying the offset as int makes applying appropriate scale invariant amount of offset possible.
    Point3D newP1(
        __int_as_float(__float_as_int(p.x) + (p.x < 0 ? -1 : 1) * offsetInInt[0]),
        __int_as_float(__float_as_int(p.y) + (p.y < 0 ? -1 : 1) * offsetInInt[1]),
        __int_as_float(__float_as_int(p.z) + (p.z < 0 ? -1 : 1) * offsetInInt[2]));

    // JP: 原点に近い場所では、原点からの距離に依存せず一定の誤差が残るため別処理が必要。
    // EN: A constant amount of error remains near the origin independent of the distance to the origin so we need handle it separately.
    Point3D newP2 = p + kFloatScale * geometricNormal;

    return Point3D(std::fabs(p.x) < kOrigin ? newP2.x : newP1.x,
                   std::fabs(p.y) < kOrigin ? newP2.y : newP1.y,
                   std::fabs(p.z) < kOrigin ? newP2.z : newP1.z);
}



struct ReferenceFrame {
    Vector3D tangent;
    Vector3D bitangent;
    Normal3D normal;

    CUDA_DEVICE_FUNCTION CUDA_INLINE ReferenceFrame() {}
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr ReferenceFrame(
        const Vector3D &_tangent, const Vector3D &_bitangent, const Normal3D &_normal) :
        tangent(_tangent), bitangent(_bitangent), normal(_normal) {}
    CUDA_DEVICE_FUNCTION CUDA_INLINE /*constexpr*/ ReferenceFrame(const Normal3D &_normal) :
        normal(_normal) {
        normal.makeCoordinateSystem(&tangent, &bitangent);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE /*constexpr*/ ReferenceFrame(
        const Normal3D &_normal, const Vector3D &_tangent) :
        tangent(_tangent), normal(_normal)
    {
        bitangent = cross(normal, tangent);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr Vector3D toLocal(const Vector3D &v) const {
        return Vector3D(dot(tangent, v), dot(bitangent, v), dot(normal, v));
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr Vector3D fromLocal(const Vector3D &v) const {
        return Vector3D(dot(Vector3D(tangent.x, bitangent.x, normal.x), v),
                        dot(Vector3D(tangent.y, bitangent.y, normal.y), v),
                        dot(Vector3D(tangent.z, bitangent.z, normal.z), v));
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE void applyBumpMapping(
    const Normal3D &modNormalInTF, ReferenceFrame* frameToModify) {
    // JP: 法線から回転軸と回転角(、Quaternion)を求めて対応する接平面ベクトルを求める。
    // EN: calculate a rotating axis and an angle (and quaternion) from the normal then calculate corresponding tangential vectors.
    float projLength = std::sqrt(modNormalInTF.x * modNormalInTF.x + modNormalInTF.y * modNormalInTF.y);
    if (projLength < 1e-3f)
        return;
    float tiltAngle = std::atan(projLength / modNormalInTF.z);
    float qSin, qCos;
    sincos(tiltAngle / 2, &qSin, &qCos);
    float qX = (-modNormalInTF.y / projLength) * qSin;
    float qY = (modNormalInTF.x / projLength) * qSin;
    float qW = qCos;
    Vector3D modTangentInTF(1 - 2 * qY * qY, 2 * qX * qY, -2 * qY * qW);
    Vector3D modBitangentInTF(2 * qX * qY, 1 - 2 * qX * qX, 2 * qX * qW);

    Matrix3x3 matTFtoW(frameToModify->tangent, frameToModify->bitangent, frameToModify->normal);
    ReferenceFrame bumpShadingFrame(matTFtoW * modTangentInTF,
                                    matTFtoW * modBitangentInTF,
                                    matTFtoW * modNormalInTF);

    *frameToModify = bumpShadingFrame;
}

RT_CALLABLE_PROGRAM Normal3D RT_DC_NAME(readModifiedNormalFromNormalMap)
(CUtexObject texture, shared::TexDimInfo dimInfo, TexCoord2D texCoord) {
    float4 texValue = tex2DLod<float4>(texture, texCoord.u, texCoord.v, 0.0f);
    Normal3D modLocalNormal(texValue.x, texValue.y, texValue.z);
    modLocalNormal = 2.0f * modLocalNormal - Normal3D(1.0f);
    if (dimInfo.isLeftHanded)
        modLocalNormal.y *= -1; // DirectX convention
    return modLocalNormal;
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(readModifiedNormalFromNormalMap);

RT_CALLABLE_PROGRAM Normal3D RT_DC_NAME(readModifiedNormalFromNormalMap2ch)
(CUtexObject texture, shared::TexDimInfo dimInfo, TexCoord2D texCoord) {
    float2 texValue = tex2DLod<float2>(texture, texCoord.u, texCoord.v, 0.0f);
    float x = 2.0f * texValue.x - 1.0f;
    float y = 2.0f * texValue.y - 1.0f;
    float z = std::sqrt(1.0f - pow2(x) - pow2(y));
    Normal3D modLocalNormal(x, y, z);
    if (dimInfo.isLeftHanded)
        modLocalNormal.y *= -1; // DirectX convention
    return modLocalNormal;
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(readModifiedNormalFromNormalMap2ch);

RT_CALLABLE_PROGRAM Normal3D RT_DC_NAME(readModifiedNormalFromHeightMap)
(CUtexObject texture, shared::TexDimInfo dimInfo, TexCoord2D texCoord) {
    if (dimInfo.isNonPowerOfTwo && dimInfo.isBCTexture) {
        uint32_t bcWidth = (dimInfo.dimX + 3) / 4 * 4;
        uint32_t bcHeight = (dimInfo.dimY + 3) / 4 * 4;
        texCoord.u *= static_cast<float>(dimInfo.dimX) / bcWidth;
        texCoord.v *= static_cast<float>(dimInfo.dimY) / bcHeight;
    }
    float4 heightValues = tex2Dgather<float4>(texture, texCoord.u, texCoord.v, 0);
    constexpr float coeff = (5.0f / 1024);
    uint32_t width = dimInfo.dimX;
    uint32_t height = dimInfo.dimY;
    float dhdu = (coeff * width) * (heightValues.y - heightValues.x);
    float dhdv = (coeff * height) * (heightValues.x - heightValues.w);
    Normal3D modLocalNormal = normalize(Normal3D(-dhdu, dhdv, 1));
    return modLocalNormal;
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(readModifiedNormalFromHeightMap);



template <typename BSDFType>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody(
    const shared::SurfaceMaterial &matData,
    TexCoord2D texCoord, const WavelengthSamples &wls, uint32_t* bodyData, shared::BSDFBuildFlags flags);



class LambertBRDF {
    SampledSpectrum m_reflectance;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE LambertBRDF() {}
    CUDA_DEVICE_FUNCTION CUDA_INLINE LambertBRDF(const SampledSpectrum &reflectance) :
        m_reflectance(reflectance) {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE void getSurfaceParameters(
        SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness) const
    {
        *diffuseReflectance = m_reflectance;
        *specularReflectance = SampledSpectrum::Zero();
        *roughness = 1.0f;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateDHReflectanceEstimate(
        const shared::BSDFQuery &query) const
    {
        return m_reflectance;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool matches(shared::DirectionType dirType) const {
        using namespace shared;
        constexpr DirectionType type =
            DirectionType::Reflection() | DirectionType::LowFreq();
        return type.matches(dirType);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &sample,
        shared::BSDFQueryResult* result, shared::BSDFQueryReverseResult* revResult = nullptr) const
    {
        result->dirLocal = cosineSampleHemisphere(sample.uDir[0], sample.uDir[1]);
        const float oneOverPi = 1.0f / pi_v<float>;
        result->dirPDensity = result->dirLocal.z * oneOverPi;
        result->dirLocal.z *= query.dirLocal.z >= 0 ? 1 : -1;
        const SampledSpectrum fsValue = m_reflectance * oneOverPi;
        if (revResult) {
            revResult->fsValue = fsValue;
            revResult->dirPDensity = std::fabs(query.dirLocal.z) * oneOverPi;
        }
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        SampledSpectrum* revValue = nullptr) const
    {
        if (query.dirLocal.z * vSampled.z <= 0.0f) {
            const SampledSpectrum fs = SampledSpectrum::Zero();
            if (revValue)
                *revValue = fs;
            return fs;
        }
        const float oneOverPi = 1.0f / pi_v<float>;
        const SampledSpectrum fsValue = m_reflectance * oneOverPi;
        if (revValue)
            *revValue = fsValue;

        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        float* revValue = nullptr) const
    {
        if (query.dirLocal.z * vSampled.z <= 0.0f) {
            if (revValue)
                *revValue = 0.0f;
            return 0.0f;
        }
        const float oneOverPi = 1.0f / pi_v<float>;
        const float dirPDens = std::fabs(vSampled.z) * oneOverPi;
        if (revValue)
            *revValue = std::fabs(query.dirLocal.z) * oneOverPi;

        return dirPDens;
    }
};

template<>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<LambertBRDF>(
    const shared::SurfaceMaterial &matData,
    TexCoord2D texCoord, const WavelengthSamples &wls, uint32_t* bodyData, shared::BSDFBuildFlags /*flags*/)
{
    const auto &mat = reinterpret_cast<const shared::LambertianSurfaceMaterial &>(matData.body);
    const float4 reflectance = tex2DLod<float4>(mat.reflectance, texCoord.u, texCoord.v, 0.0f);
    auto &bsdfBody = *reinterpret_cast<LambertBRDF*>(bodyData);
    const TripletSpectrum sp = createTripletSpectrum(
        SpectrumType::Reflectance, ColorSpace::Rec709_D65, reflectance.x, reflectance.y, reflectance.z);
    bsdfBody = LambertBRDF(sp.evaluate(wls));
}



class SpecularBSDF {
    class FresnelDielectric {
        SampledSpectrum m_etaExt;
        SampledSpectrum m_etaInt;

    public:
        CUDA_DEVICE_FUNCTION CUDA_INLINE FresnelDielectric() {}
        CUDA_DEVICE_FUNCTION CUDA_INLINE FresnelDielectric(
            const SampledSpectrum &etaExt, const SampledSpectrum &etaInt) :
            m_etaExt(etaExt), m_etaInt(etaInt) {}

        CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum etaExt() const { return m_etaExt; }
        CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum etaInt() const { return m_etaInt; }

        CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluate(float cosEnter) const {
            cosEnter = clamp(cosEnter, -1.0f, 1.0f);

            const bool entering = cosEnter > 0.0f;
            const SampledSpectrum &eEnter = entering ? m_etaExt : m_etaInt;
            const SampledSpectrum &eExit = entering ? m_etaInt : m_etaExt;

            const SampledSpectrum sinExit = eEnter / eExit * std::sqrt(std::fmax(0.0f, 1.0f - cosEnter * cosEnter));
            SampledSpectrum ret = SampledSpectrum::Zero();
            cosEnter = std::fabs(cosEnter);
            for (int i = 0; i < SampledSpectrum::NumComponents(); ++i) {
                if (sinExit[i] >= 1.0f) {
                    ret[i] = 1.0f;
                }
                else {
                    const float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit[i] * sinExit[i]));
                    ret[i] = evalF(eEnter[i], eExit[i], cosEnter, cosExit);
                }
            }
            return ret;
        }
        CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluate(float cosEnter, uint32_t wlIdx) const {
            cosEnter = clamp(cosEnter, -1.0f, 1.0f);

            const bool entering = cosEnter > 0.0f;
            const float &eEnter = entering ? m_etaExt[wlIdx] : m_etaInt[wlIdx];
            const float &eExit = entering ? m_etaInt[wlIdx] : m_etaExt[wlIdx];

            const float sinExit = eEnter / eExit * std::sqrt(std::fmax(0.0f, 1.0f - cosEnter * cosEnter));
            cosEnter = std::fabs(cosEnter);
            if (sinExit >= 1.0f) {
                return 1.0f;
            }
            else {
                const float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit * sinExit));
                return evalF(eEnter, eExit, cosEnter, cosExit);
            }
        }

        CUDA_DEVICE_FUNCTION CUDA_INLINE static float evalF(
            float etaEnter, float etaExit, float cosEnter, float cosExit)
        {
            const float Rparl =
                ((etaExit * cosEnter) - (etaEnter * cosExit)) /
                ((etaExit * cosEnter) + (etaEnter * cosExit));
            const float Rperp =
                ((etaEnter * cosEnter) - (etaExit * cosExit)) /
                ((etaEnter * cosEnter) + (etaExit * cosExit));
            return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
        }
    };

    SampledSpectrum m_etaExt;
    SampledSpectrum m_etaInt;
    uint32_t m_dispersive;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE SpecularBSDF() {}
    CUDA_DEVICE_FUNCTION CUDA_INLINE SpecularBSDF(
        const SampledSpectrum &etaExt, const SampledSpectrum &etaInt, bool dispersive) :
        m_etaExt(etaExt), m_etaInt(etaInt), m_dispersive(dispersive) {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE void getSurfaceParameters(
        SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness) const
    {
        *diffuseReflectance = SampledSpectrum::Zero();
        *specularReflectance = SampledSpectrum::Zero();
        *roughness = 0.0f;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateDHReflectanceEstimate(
        const shared::BSDFQuery &query) const
    {
        return SampledSpectrum::One();
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool matches(shared::DirectionType dirType) const {
        using namespace shared;
        constexpr DirectionType type =
            DirectionType::Reflection() | DirectionType::Delta0D();
        return type.matches(dirType);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &sample,
        shared::BSDFQueryResult* result, shared::BSDFQueryReverseResult* revResult = nullptr) const
    {
        using namespace shared;

        if (revResult)
            revResult->dirPDensity = 0.0f;

        const bool entering = query.dirLocal.z >= 0.0f;

        const SampledSpectrum &eEnter = entering ? m_etaExt : m_etaInt;
        const SampledSpectrum &eExit = entering ? m_etaInt : m_etaExt;
        const FresnelDielectric fresnel(eEnter, eExit);

        const Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        const SampledSpectrum F = fresnel.evaluate(dirV.z);
        const float reflectProb = F.importance(query.wlHint);
        const float uComp = sample.uDir[1];
        if (uComp < reflectProb) {
            if (dirV.z == 0.0f) {
                result->dirPDensity = 0.0f;
                return SampledSpectrum::Zero();
            }
            const Vector3D dirL = Vector3D(-dirV.x, -dirV.y, dirV.z);
            result->dirLocal = entering ? dirL : -dirL;
            result->dirPDensity = reflectProb;
            result->sampledType = DirectionType::Reflection() | DirectionType::Delta0D();
            const SampledSpectrum ret = F * (1.0f / std::fabs(dirV.z));

            if (revResult) {
                revResult->fsValue = ret;
                revResult->dirPDensity = result->dirPDensity;
            }

            return ret;
        }
        else {
            const float sinEnter2 = 1.0f - dirV.z * dirV.z;
            const float recRelIOR = eEnter[query.wlHint] / eExit[query.wlHint];// reciprocal of relative IOR.
            const float sinExit2 = recRelIOR * recRelIOR * sinEnter2;

            if (sinExit2 >= 1.0f) {
                result->dirPDensity = 0.0f;
                return SampledSpectrum::Zero();
            }
            const float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit2));
            const Vector3D dirL = Vector3D(recRelIOR * -dirV.x, recRelIOR * -dirV.y, -cosExit);
            result->dirLocal = entering ? dirL : -dirL;
            result->dirPDensity = (1.0f - reflectProb);
            result->sampledType =
                DirectionType::Transmission() | DirectionType::Delta0D() |
                (m_dispersive ? DirectionType::Dispersive() : DirectionType());

            SampledSpectrum ret = SampledSpectrum::Zero();
            ret[query.wlHint] = (1.0f - F[query.wlHint]);

            if (revResult) {
                float revSqueezeFactor = 1.0f;
                if (static_cast<TransportMode>(query.transportMode) == TransportMode::Importance)
                    revSqueezeFactor = pow2(eExit[query.wlHint] / eEnter[query.wlHint]);
                SampledSpectrum revRet = ret;
                revRet[query.wlHint] *= revSqueezeFactor / std::fabs(query.dirLocal.z);
                revResult->fsValue = revRet;
                revResult->dirPDensity = revSqueezeFactor * result->dirPDensity;
            }

            float squeezeFactor = 1.0f;
            if (static_cast<TransportMode>(query.transportMode) == TransportMode::Radiance)
                squeezeFactor *= pow2(eEnter[query.wlHint] / eExit[query.wlHint]);
            ret[query.wlHint] *= squeezeFactor / std::fabs(cosExit);
            result->dirPDensity *= squeezeFactor;

            return ret;
        }
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        SampledSpectrum* revValue = nullptr) const
    {
        if (revValue)
            *revValue = SampledSpectrum::Zero();
        return SampledSpectrum::Zero();
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        float* revValue = nullptr) const
    {
        if (revValue)
            *revValue = 0.0f;
        return 0.0f;
    }
};

template<>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<SpecularBSDF>(
    const shared::SurfaceMaterial &matData,
    TexCoord2D texCoord, const WavelengthSamples &wls, uint32_t* bodyData, shared::BSDFBuildFlags /*flags*/)
{
    const auto &mat = reinterpret_cast<const shared::SpecularScatteringSurfaceMaterial &>(matData.body);
    SampledSpectrum etaExt;
    SampledSpectrum etaInt;
    for (uint32_t i = 0; i < NumSpectralSamples; ++i) {
        etaExt[i] = calcIor(mat.iorExt, mat.abbeNumExt, wls[i]);
        etaInt[i] = calcIor(mat.iorInt, mat.abbeNumInt, wls[i]);
    }
    auto &bsdfBody = *reinterpret_cast<SpecularBSDF*>(bodyData);
    bsdfBody = SpecularBSDF(etaExt, etaInt, !wls.singleIsSelected());
}



// DichromaticBRDFのDirectional-Hemispherical Reflectanceを事前計算して
// テクスチャー化した結果をフィッティングする。
// Diffuse、Specular成分はそれぞれ
// - baseColor * diffusePreInt(cosV, roughness)
// - specularF0 * specularPreIntA(cosV, roughness) + (1 - specularF0) * specularPreIntB(cosV, roughness)
// で表される。
// https://www.shadertoy.com/view/WtjfRD
CUDA_DEVICE_FUNCTION CUDA_INLINE void calcFittedPreIntegratedTerms(
    float cosV, float roughness,
    float* diffusePreInt, float* specularPreIntA, float* specularPreIntB) {
        {
            float u = cosV;
            float v = roughness;
            float uu = u * u;
            float uv = u * v;
            float vv = v * v;

            *diffusePreInt = min(max(
                -0.417425f * uu +
                -0.958929f * uv +
                -0.096977f * vv +
                1.050356f * u +
                0.534528f * v +
                0.407112f * 1.0f,
                0.0f), 1.0f);
        }
        {
            float u = std::atan2(roughness, cosV);
            float v = std::sqrt(cosV * cosV + roughness * roughness);
            float uu = u * u;
            float uv = u * v;
            float vv = v * v;

            *specularPreIntA = min(max(
                0.133105f * uu +
                -0.278877f * uv +
                -0.417142f * vv +
                -0.192809f * u +
                0.426076f * v +
                0.996565f * 1.0f,
                0.0f), 1.0f);
            *specularPreIntB = min(max(
                0.055070f * uu +
                -0.163511f * uv +
                1.211598f * vv +
                0.089837f * u +
                -1.956888f * v +
                0.741397f * 1.0f,
                0.0f), 1.0f);
        }
}

#define USE_HEIGHT_CORRELATED_SMITH 1
#define USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS 0
#define USE_FITTED_PRE_INTEGRATION_FOR_DH_REFLECTANCE 0

class DichromaticBRDF {
    struct GGXMicrofacetDistribution {
        float alpha_g;

        CUDA_DEVICE_FUNCTION float evaluate(const Normal3D &m) const {
            if (m.z <= 0.0f)
                return 0.0f;
            float temp = pow2(m.x) + pow2(m.y) + pow2(m.z * alpha_g);
            return pow2(alpha_g) / (pi_v<float> * pow2(temp));
        }
        CUDA_DEVICE_FUNCTION float evaluateSmithG1(const Vector3D &v, const Normal3D &m) const {
            if (dot(v, m) * v.z <= 0)
                return 0.0f;
            float temp = pow2(alpha_g) * (pow2(v.x) + pow2(v.y)) / pow2(v.z);
            return 2 / (1 + std::sqrt(1 + temp));
        }
        CUDA_DEVICE_FUNCTION float evaluateHeightCorrelatedSmithG(
            const Vector3D &v1, const Vector3D &v2, const Normal3D &m) {
            float alpha_g2_tanTheta2_1 = pow2(alpha_g) * (pow2(v1.x) + pow2(v1.y)) / pow2(v1.z);
            float alpha_g2_tanTheta2_2 = pow2(alpha_g) * (pow2(v2.x) + pow2(v2.y)) / pow2(v2.z);
            float Lambda1 = (-1 + std::sqrt(1 + alpha_g2_tanTheta2_1)) / 2;
            float Lambda2 = (-1 + std::sqrt(1 + alpha_g2_tanTheta2_2)) / 2;
            float chi1 = (dot(v1, m) / v1.z) > 0 ? 1 : 0;
            float chi2 = (dot(v2, m) / v2.z) > 0 ? 1 : 0;
            return chi1 * chi2 / (1 + Lambda1 + Lambda2);
        }
        CUDA_DEVICE_FUNCTION float sample(
            const Vector3D &v, float u0, float u1, Normal3D* m, float* mPDensity) const {
            // stretch view
            Vector3D sv = normalize(Vector3D(alpha_g * v.x, alpha_g * v.y, v.z));

            // orthonormal basis
            float distIn2D = std::sqrt(sv.x * sv.x + sv.y * sv.y);
            float recDistIn2D = 1.0f / distIn2D;
            Vector3D T1 = (sv.z < 0.9999f) ?
                Vector3D(sv.y * recDistIn2D, -sv.x * recDistIn2D, 0) :
                Vector3D::Ex();
            Vector3D T2(T1.y * sv.z, -T1.x * sv.z, distIn2D);

            // sample point with polar coordinates (r, phi)
            float a = 1.0f / (1.0f + sv.z);
            float r = std::sqrt(u0);
            float phi = pi_v<float> * ((u1 < a) ? u1 / a : 1 + (u1 - a) / (1.0f - a));
            float sinPhi, cosPhi;
            sincos(phi, &sinPhi, &cosPhi);
            float P1 = r * cosPhi;
            float P2 = r * sinPhi * ((u1 < a) ? 1.0f : sv.z);

            // compute normal
            *m = P1 * T1 + P2 * T2 + std::sqrt(1.0f - P1 * P1 - P2 * P2) * sv;

            // unstretch
            *m = normalize(Normal3D(alpha_g * m->x, alpha_g * m->y, m->z));

            float D = evaluate(*m);
            *mPDensity = evaluateSmithG1(v, *m) * absDot(v, *m) * D / std::fabs(v.z);

            return D;
        }
        CUDA_DEVICE_FUNCTION float evaluatePDF(const Vector3D &v, const Normal3D &m) {
            return evaluateSmithG1(v, m) * absDot(v, m) * evaluate(m) / std::fabs(v.z);
        }
    };

protected:
    SampledSpectrum m_diffuseColor;
    SampledSpectrum m_specularF0Color;
    float m_roughness;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE DichromaticBRDF() {}
    CUDA_DEVICE_FUNCTION CUDA_INLINE DichromaticBRDF(
        const SampledSpectrum &diffuseColor, const SampledSpectrum &specularF0Color, float smoothness) {
        m_diffuseColor = diffuseColor;
        m_specularF0Color = specularF0Color;
        m_roughness = 1 - smoothness;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE void getSurfaceParameters(
        SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness) const
    {
        *diffuseReflectance = m_diffuseColor;
        *specularReflectance = m_specularF0Color;
        *roughness = m_roughness;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateDHReflectanceEstimate(
        const shared::BSDFQuery &query) const
    {
        const bool entering = query.dirLocal.z >= 0.0f;
        const Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        SampledSpectrum diffuseDHR;
        SampledSpectrum specularDHR;
        if constexpr (USE_FITTED_PRE_INTEGRATION_FOR_DH_REFLECTANCE) {
            float diffusePreInt;
            float specularPreIntA, specularPreIntB;
            calcFittedPreIntegratedTerms(dirV.z, m_roughness, &diffusePreInt, &specularPreIntA, &specularPreIntB);

            diffuseDHR = m_diffuseColor * diffusePreInt;
            specularDHR =
                m_specularF0Color * specularPreIntA +
                (SampledSpectrum::One() - m_specularF0Color) * specularPreIntB;
        }
        else {
            const float expectedCosTheta_d = dirV.z;
            const float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(expectedCosTheta_d);
            const float oneMinusDotVN5 = pow5(1 - dirV.z);
            const float expectedDiffFGiven = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
            const float expectedDiffFSampled = 1.0f; // ad-hoc
            diffuseDHR = m_diffuseColor *
                expectedDiffFGiven * expectedDiffFSampled * lerp(1.0f, 1.0f / 1.51f, m_roughness);

            //float expectedOneMinusDotVH5 = oneMinusDotVN5;
            // (1 - m_roughness) is an ad-hoc adjustment.
            const float expectedOneMinusDotVH5 = pow5(1 - dirV.z) * (1 - m_roughness);

            specularDHR = lerp(m_specularF0Color, SampledSpectrum(1.0f), expectedOneMinusDotVH5);
        }

        return min(diffuseDHR + specularDHR, SampledSpectrum::One());
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool matches(shared::DirectionType dirType) const {
        using namespace shared;
        constexpr DirectionType type =
            DirectionType::Reflection() | DirectionType::LowFreq() | DirectionType::HighFreq();
        return type.matches(dirType);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &sample,
        shared::BSDFQueryResult* result, shared::BSDFQueryReverseResult* revResult = nullptr) const
    {
        if (revResult)
            revResult->dirPDensity = 0.0f;

        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        const bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirL;
        const Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        const float oneMinusDotVN5 = pow5(1 - dirV.z);

        float diffuseWeight;
        float specularWeight;
        if constexpr (USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS) {
            float diffusePreInt;
            float specularPreIntA, specularPreIntB;
            calcFittedPreIntegratedTerms(dirV.z, m_roughness, &diffusePreInt, &specularPreIntA, &specularPreIntB);

            diffuseWeight = (m_diffuseColor * diffusePreInt).importance(query.wlHint);
            specularWeight =
                (m_specularF0Color * specularPreIntA +
                 (SampledSpectrum::One() - m_specularF0Color) * specularPreIntB).importance(query.wlHint);
        }
        else {
            const float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * query.dirLocal.z * query.dirLocal.z;
            const float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
            const float iBaseColor = m_diffuseColor.importance(query.wlHint) * pow2(expectedDiffuseFresnel) *
                lerp(1.0f, 1.0f / 1.51f, m_roughness);

            const float expectedOneMinusDotVH5 = pow5(1 - dirV.z);
            const float iSpecularF0 = m_specularF0Color.importance(query.wlHint);

            diffuseWeight = iBaseColor;
            specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);
        }
        const float sumWeights = diffuseWeight + specularWeight;
        if (sumWeights == 0.0f) {
            result->dirPDensity = 0.0f;
            return SampledSpectrum::Zero();
        }

        float uDir1;
        float compProb;
        const uint32_t compIdx = rtc10::shared::sampleDiscrete(
            sample.uDir[1], &compProb, &uDir1, diffuseWeight, specularWeight);

        float diffuseDirPDens, specularDirPDens;
        Normal3D m;
        float dotLH;
        float D;
        if (compIdx == 0) {
            // JP: コサイン分布からサンプルする。
            // EN: sample based on cosine distribution.
            dirL = cosineSampleHemisphere(sample.uDir[0], uDir1);
            diffuseDirPDens = dirL.z / pi_v<float>;

            // JP: 同じ方向サンプルをスペキュラー層からサンプルする確率密度を求める。
            // EN: calculate PDF to generate the sampled direction from the specular layer.
            m = halfVector(dirL, dirV);
            dotLH = min(dot(dirL, m), 1.0f);
            const float commonPDFTerm = 1.0f / (4 * dotLH);
            specularDirPDens = commonPDFTerm * ggx.evaluatePDF(dirV, m);

            D = ggx.evaluate(m);
        }
        else {
            // JP: スペキュラー層のマイクロファセット分布からサンプルする。
            // EN: sample based on the specular microfacet distribution.
            float mPDF;
            D = ggx.sample(dirV, sample.uDir[0], uDir1, &m, &mPDF);
            const float dotVH = min(dot(dirV, m), 1.0f);
            dotLH = dotVH;
            dirL = Vector3D(2 * dotVH * m) - dirV;
            if (dirL.z * dirV.z <= 0) {
                result->dirPDensity = 0.0f;
                return SampledSpectrum::Zero();
            }
            const float commonPDFTerm = 1.0f / (4 * dotLH);
            specularDirPDens = commonPDFTerm * mPDF;

            // JP: 同じ方向サンプルをコサイン分布からサンプルする確率密度を求める。
            // EN: calculate PDF to generate the sampled direction from the cosine distribution.
            diffuseDirPDens = dirL.z / pi_v<float>;
        }

        const float oneMinusDotLH5 = pow5(1 - dotLH);

        float G;
        if constexpr (USE_HEIGHT_CORRELATED_SMITH)
            G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
        else
            G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
        constexpr float F90 = 1.0f;
        const SampledSpectrum F = lerp(m_specularF0Color, SampledSpectrum(F90), oneMinusDotLH5);

        const float microfacetDenom = 4 * dirL.z * dirV.z;
        SampledSpectrum specularValue = F * ((D * G) / microfacetDenom);
        if (G == 0)
            specularValue = SampledSpectrum::Zero();

        const float F_D90 = 0.5f * m_roughness + 2 * m_roughness * dotLH * dotLH;
        const float oneMinusDotLN5 = pow5(1 - dirL.z);
        const float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        const float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);
        const SampledSpectrum diffuseValue = m_diffuseColor *
            (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / pi_v<float>);

        const SampledSpectrum ret = diffuseValue + specularValue;

        result->dirLocal = entering ? dirL : -dirL;

        // PDF based on one-sample model MIS.
        result->dirPDensity = (diffuseDirPDens * diffuseWeight + specularDirPDens * specularWeight) / sumWeights;

        if (revResult) {
            const float revDiffuseDirPDens = dirV.z / pi_v<float>;
            const float commonPDFTerm = 1.0f / (4 * dotLH);
            const float revSpecularDirPDens = commonPDFTerm * ggx.evaluatePDF(dirL, m);

            const float revExpectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(dirL.z);
            const float revOneMinusDotVN5 = pow5(1 - dirL.z);
            const float revExpectedDiffuseFresnel = lerp(1.0f, revExpectedF_D90, revOneMinusDotVN5);
            const float revIBaseColor = m_diffuseColor.importance(query.wlHint) * pow2(revExpectedDiffuseFresnel) * lerp(1.0f, 1.0f / 1.51f, m_roughness);

            const float revExpectedOneMinusDotVH5 = pow5(1 - dirL.z);
            const float revISpecularF0 = m_specularF0Color.importance(query.wlHint);

            const float revDiffuseWeight = revIBaseColor;
            const float revSpecularWeight = lerp(revISpecularF0, 1.0f, revExpectedOneMinusDotVH5);

            revResult->fsValue = ret;
            revResult->dirPDensity =
                (revDiffuseDirPDens * revDiffuseWeight + revSpecularDirPDens * revSpecularWeight) /
                (revDiffuseWeight + revSpecularWeight);
        }

        return ret;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        SampledSpectrum* revValue = nullptr) const
    {
        if (revValue)
            *revValue = SampledSpectrum::Zero();

        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        if (vSampled.z * query.dirLocal.z <= 0)
            return SampledSpectrum::Zero();

        const bool entering = query.dirLocal.z >= 0.0f;
        const Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        const Vector3D dirL = entering ? vSampled : -vSampled;

        const Normal3D m = halfVector(dirL, dirV);
        const float dotLH = dot(dirL, m);

        const float oneMinusDotLH5 = pow5(1 - dotLH);

        const float D = ggx.evaluate(m);
        float G;
        if constexpr (USE_HEIGHT_CORRELATED_SMITH)
            G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
        else
            G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
        constexpr float F90 = 1.0f;
        const SampledSpectrum F = lerp(m_specularF0Color, SampledSpectrum(F90), oneMinusDotLH5);

        const float microfacetDenom = 4 * dirL.z * dirV.z;
        SampledSpectrum specularValue = F * ((D * G) / microfacetDenom);
        if (G == 0)
            specularValue = SampledSpectrum::Zero();

        const float F_D90 = 0.5f * m_roughness + 2 * m_roughness * dotLH * dotLH;
        const float oneMinusDotVN5 = pow5(1 - dirV.z);
        const float oneMinusDotLN5 = pow5(1 - dirL.z);
        const float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        const float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);

        const SampledSpectrum diffuseValue = m_diffuseColor *
            (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / pi_v<float>);

        const SampledSpectrum ret = diffuseValue + specularValue;

        if (revValue)
            *revValue = ret;

        return ret;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        float* revValue = nullptr) const
    {
        if (revValue)
            *revValue = 0.0f;

        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        const bool entering = query.dirLocal.z >= 0.0f;
        const Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        const Vector3D dirL = entering ? vSampled : -vSampled;

        const Normal3D m = halfVector(dirL, dirV);
        const float dotLH = dot(dirL, m);
        const float commonPDFTerm = 1.0f / (4 * dotLH);

        float diffuseWeight;
        float specularWeight;
        if constexpr (USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS) {
            float diffusePreInt;
            float specularPreIntA, specularPreIntB;
            calcFittedPreIntegratedTerms(dirV.z, m_roughness, &diffusePreInt, &specularPreIntA, &specularPreIntB);

            diffuseWeight = (m_diffuseColor * diffusePreInt).importance(query.wlHint);
            specularWeight =
                (m_specularF0Color * specularPreIntA +
                 (SampledSpectrum::One() - m_specularF0Color) * specularPreIntB).importance(query.wlHint);
        }
        else {
            const float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(dirV.z);
            const float oneMinusDotVN5 = pow5(1 - dirV.z);
            const float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
            const float iBaseColor = m_diffuseColor.importance(query.wlHint) * pow2(expectedDiffuseFresnel) *
                lerp(1.0f, 1.0f / 1.51f, m_roughness);

            const float expectedOneMinusDotVH5 = pow5(1 - dirV.z);
            const float iSpecularF0 = m_specularF0Color.importance(query.wlHint);

            diffuseWeight = iBaseColor;
            specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);
        }
        const float sumWeights = diffuseWeight + specularWeight;
        if (sumWeights == 0.0f)
            return 0.0f;

        const float diffuseDirPDens = dirL.z / pi_v<float>;
        const float specularDirPDens = commonPDFTerm * ggx.evaluatePDF(dirV, m);

        const float ret = (diffuseDirPDens * diffuseWeight + specularDirPDens * specularWeight) / sumWeights;

        if (revValue) {
            const float revDiffuseDirPDens = dirV.z / pi_v<float>;
            const float commonPDFTerm = 1.0f / (4 * dotLH);
            const float revSpecularDirPDens = commonPDFTerm * ggx.evaluatePDF(dirL, m);

            const float revExpectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(dirL.z);
            const float revOneMinusDotVN5 = pow5(1 - dirL.z);
            const float revExpectedDiffuseFresnel = lerp(1.0f, revExpectedF_D90, revOneMinusDotVN5);
            const float revIBaseColor = m_diffuseColor.importance(query.wlHint) *
                pow2(revExpectedDiffuseFresnel) * lerp(1.0f, 1.0f / 1.51f, m_roughness);

            const float revExpectedOneMinusDotVH5 = pow5(1 - dirL.z);
            const float revISpecularF0 = m_specularF0Color.importance(query.wlHint);

            const float revDiffuseWeight = revIBaseColor;
            const float revSpecularWeight = lerp(revISpecularF0, 1.0f, revExpectedOneMinusDotVH5);

            *revValue =
                (revDiffuseDirPDens * revDiffuseWeight + revSpecularDirPDens * revSpecularWeight) /
                (revDiffuseWeight + revSpecularWeight);
        }

        return ret;
    }
};

class SimplePBR_BRDF : public DichromaticBRDF {
public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE SimplePBR_BRDF() {}
    CUDA_DEVICE_FUNCTION CUDA_INLINE SimplePBR_BRDF(
        const SampledSpectrum &baseColor, float reflectance, float smoothness, float metallic)
    {
        m_diffuseColor = baseColor * (1 - metallic);
        m_specularF0Color = SampledSpectrum(0.16f * pow2(reflectance) * (1 - metallic)) + baseColor * metallic;
        m_roughness = 1 - smoothness;
    }
};

template<>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<DichromaticBRDF>(
    const shared::SurfaceMaterial &matData,
    TexCoord2D texCoord, const WavelengthSamples &wls, uint32_t* bodyData, shared::BSDFBuildFlags flags)
{
    const auto &mat = reinterpret_cast<const shared::DichromaticSurfaceMaterial &>(matData.body);
    const float4 diffuseColor = tex2DLod<float4>(mat.diffuse, texCoord.u, texCoord.v, 0.0f);
    const float4 specularF0Color = tex2DLod<float4>(mat.specular, texCoord.u, texCoord.v, 0.0f);
    const float smoothness = tex2DLod<float>(mat.smoothness, texCoord.u, texCoord.v, 0.0f);
    //const bool regularize = (flags & BSDFFlags::Regularize) != 0;
    //if (regularize)
    //    smoothness *= 0.5f;
    auto &bsdfBody = *reinterpret_cast<DichromaticBRDF*>(bodyData);
    const auto spBaseColor = createTripletSpectrum(
        SpectrumType::Reflectance, ColorSpace::Rec709_D65, diffuseColor.x, diffuseColor.y, diffuseColor.z);
    const auto spF0Color = createTripletSpectrum(
        SpectrumType::Reflectance, ColorSpace::Rec709_D65, specularF0Color.x, specularF0Color.y, specularF0Color.z);
    bsdfBody = DichromaticBRDF(spBaseColor.evaluate(wls), spF0Color.evaluate(wls), min(smoothness, 0.999f));
}

template<>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<SimplePBR_BRDF>(
    const shared::SurfaceMaterial &matData,
    TexCoord2D texCoord, const WavelengthSamples &wls, uint32_t* bodyData, shared::BSDFBuildFlags flags)
{
    const auto &mat = reinterpret_cast<const shared::SimplePBRSurfaceMaterial &>(matData.body);
    const float4 baseColor_opacity = tex2DLod<float4>(
        mat.baseColor_opacity, texCoord.u, texCoord.v, 0.0f);
    const float4 occlusion_roughness_metallic = tex2DLod<float4>(
        mat.occlusion_roughness_metallic, texCoord.u, texCoord.v, 0.0f);
    const float smoothness = min(1.0f - occlusion_roughness_metallic.y, 0.999f);
    const float metallic = occlusion_roughness_metallic.z;
    //const bool regularize = (flags & BSDFFlags::Regularize) != 0;
    //if (regularize)
    //    smoothness *= 0.5f;
    auto &bsdfBody = *reinterpret_cast<SimplePBR_BRDF*>(bodyData);
    const auto spBaseColor = createTripletSpectrum(
        SpectrumType::Reflectance, ColorSpace::Rec709_D65,
        baseColor_opacity.x, baseColor_opacity.y, baseColor_opacity.z);
    bsdfBody = SimplePBR_BRDF(spBaseColor.evaluate(wls), 0.5f, smoothness, metallic);
}



#define DEFINE_BSDF_CALLABLES(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(BSDFType ## _getSurfaceParameters)(\
        const uint32_t* data,\
        SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness)\
    {\
        const auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.getSurfaceParameters(diffuseReflectance, specularReflectance, roughness);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _getSurfaceParameters);\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDFType ## _evaluateDHReflectanceEstimate)(\
        const uint32_t* data, const shared::BSDFQuery &query)\
    {\
        const auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluateDHReflectanceEstimate(query);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluateDHReflectanceEstimate);\
    RT_CALLABLE_PROGRAM bool RT_DC_NAME(BSDFType ## _matches)(\
        const uint32_t* data, shared::DirectionType dirType)\
    {\
        const auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.matches(dirType);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _matches);\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDFType ## _sampleF)(\
        const uint32_t* data, const shared::BSDFQuery &query, const shared::BSDFSample &sample,\
        shared::BSDFQueryResult* result)\
    {\
        const auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.sampleF(query, sample, result);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _sampleF);\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDFType ## _sampleFWithRev)(\
        const uint32_t* data, const shared::BSDFQuery &query, const shared::BSDFSample &sample,\
        shared::BSDFQueryResult* result, shared::BSDFQueryReverseResult* revResult)\
    {\
        const auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.sampleF(query, sample, result, revResult);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _sampleFWithRev);\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDFType ## _evaluateF)(\
        const uint32_t* data, const shared::BSDFQuery &query, const Vector3D &vSampled)\
    {\
        const auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluateF(query, vSampled);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluateF);\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDFType ## _evaluateFWithRev)(\
        const uint32_t* data, const shared::BSDFQuery &query, const Vector3D &vSampled,\
        SampledSpectrum* revValue)\
    {\
        const auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluateF(query, vSampled, revValue);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluateFWithRev);\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(BSDFType ## _evaluatePDF)(\
        const uint32_t* data, const shared::BSDFQuery &query, const Vector3D &vSampled)\
    {\
        const auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluatePDF(query, vSampled);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluatePDF);\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(BSDFType ## _evaluatePDFWithRev)(\
        const uint32_t* data, const shared::BSDFQuery &query, const Vector3D &vSampled,\
        float* revValue)\
    {\
        const auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluatePDF(query, vSampled, revValue);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluatePDFWithRev);

DEFINE_BSDF_CALLABLES(LambertBRDF);
DEFINE_BSDF_CALLABLES(SpecularBSDF);
DEFINE_BSDF_CALLABLES(DichromaticBRDF);

#undef DEFINE_SETUP_BSDF_CALLABLE

#define DEFINE_SETUP_BSDF_CALLABLE(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(setup ## BSDFType)(\
        const shared::SurfaceMaterial &matData,\
        TexCoord2D texCoord, const WavelengthSamples &wls, uint32_t* bodyData, shared::BSDFBuildFlags flags)\
    {\
        setupBSDFBody<BSDFType>(matData, texCoord, wls, bodyData, flags);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(setup ## BSDFType)

DEFINE_SETUP_BSDF_CALLABLE(LambertBRDF);
DEFINE_SETUP_BSDF_CALLABLE(SpecularBSDF);
DEFINE_SETUP_BSDF_CALLABLE(DichromaticBRDF);
DEFINE_SETUP_BSDF_CALLABLE(SimplePBR_BRDF);

#undef DEFINE_SETUP_BSDF_CALLABLE



class IsotropicPhaseFunction {
public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE IsotropicPhaseFunction() {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool matches(shared::DirectionType dirType) const {
        using namespace shared;
        constexpr DirectionType type =
            DirectionType::WholeSphere() | DirectionType::LowFreq();
        return type.matches(dirType);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &sample,
        shared::BSDFQueryResult* result, shared::BSDFQueryReverseResult* revResult = nullptr) const
    {
        result->dirLocal = uniformSampleSphere(sample.uDir[0], sample.uDir[1]);
        result->dirPDensity = 1.0f / (4 * pi_v<float>);
        const SampledSpectrum fpValue = SampledSpectrum(1.0f / (4 * pi_v<float>));
        if (revResult) {
            revResult->fsValue = fpValue;
            revResult->dirPDensity = result->dirPDensity;
        }
        return fpValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        SampledSpectrum* revValue = nullptr) const
    {
        const SampledSpectrum fpValue = SampledSpectrum(1.0f / (4 * pi_v<float>));
        if (revValue)
            *revValue = fpValue;
        return fpValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        float* revValue = nullptr) const
    {
        const float dirPDens = 1.0f / (4 * pi_v<float>);
        if (revValue)
            *revValue = dirPDens;
        return dirPDens;
    }
};



class SchlickPhaseFunction {
    float m_k;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE SchlickPhaseFunction(float k) : m_k(k) {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool matches(shared::DirectionType dirType) const {
        using namespace shared;
        constexpr DirectionType type =
            DirectionType::WholeSphere() | DirectionType::LowFreq();
        return type.matches(dirType);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &sample,
        shared::BSDFQueryResult* result, shared::BSDFQueryReverseResult* revResult = nullptr) const
    {
        const float cosTheta = clamp(
            (2 * sample.uDir[1] + m_k - 1) / (2 * m_k * sample.uDir[1] - m_k + 1),
            -1.0f, 1.0f);
        const float phi = 2 * pi_v<float> * sample.uDir[0];

        const float dTerm = (1 - m_k * cosTheta);
        const float value = (1 - pow2(m_k)) / (4 * pi_v<float> * pow2(dTerm));
        const float sinTheta = std::sqrt(1 - pow2(cosTheta));
        result->dirLocal = Vector3D(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
        result->dirPDensity = value;
        const SampledSpectrum fpValue = SampledSpectrum(value);
        if (revResult) {
            revResult->fsValue = fpValue;
            revResult->dirPDensity = value;
        }
        return fpValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        SampledSpectrum* revValue = nullptr) const
    {
        const float cosTheta = vSampled.z;
        const float dTerm = (1 - m_k * cosTheta);
        const float value = (1 - pow2(m_k)) / (4 * pi_v<float> * pow2(dTerm));
        const SampledSpectrum fpValue = SampledSpectrum(value);
        if (revValue)
            *revValue = fpValue;
        return fpValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        float* revValue = nullptr) const
    {
        const float cosTheta = vSampled.z;
        const float dTerm = (1 - m_k * cosTheta);
        const float dirPDens = (1 - pow2(m_k)) / (4 * pi_v<float> * pow2(dTerm));
        if (revValue)
            *revValue = dirPDens;
        return dirPDens;
    }
};



class DiffuseEDF {
public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE DiffuseEDF() {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool matches(shared::DirectionType dirType) const {
        using namespace shared;
        constexpr DirectionType type =
            DirectionType::Emission() | DirectionType::LowFreq();
        return type.matches(dirType);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::EDFQuery &query, const shared::EDFSample &sample,
        shared::EDFQueryResult* result) const
    {
        result->dirLocal = cosineSampleHemisphere(sample.uDir[0], sample.uDir[1]);
        const float oneOverPi = 1.0f / pi_v<float>;
        result->dirPDensity = result->dirLocal.z * oneOverPi;
        const SampledSpectrum feValue = oneOverPi;
        return feValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::EDFQuery &query, const Vector3D &vSampled) const
    {
        if (vSampled.z <= 0.0f) {
            const SampledSpectrum fs = SampledSpectrum::Zero();
            return fs;
        }
        const float oneOverPi = 1.0f / pi_v<float>;
        const SampledSpectrum fsValue = oneOverPi;
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::EDFQuery &query, const Vector3D &vSampled) const
    {
        if (vSampled.z <= 0.0f) {
            return 0.0f;
        }
        const float oneOverPi = 1.0f / pi_v<float>;
        const float dirPDens = std::fabs(vSampled.z) * oneOverPi;
        return dirPDens;
    }
};



enum class BSDFGrade {
    Unidirectional = 0,
    Bidirectional,
};



template <BSDFGrade bsdfGrade, bool isGeneric>
class BSDFTemplate;

template <BSDFGrade bsdfGrade>
class BSDFTemplate<bsdfGrade, false> {
    union {
        HARD_CODED_BSDF m_bsdf;
        struct {
            IsotropicPhaseFunction pf;
            //SchlickPhaseFunction pf;
            SampledSpectrum scatteringAlbedo;
        } m_m;
        struct {
            uint32_t isDirectionalEdf : 1;
        } m_e;
    };
    uint32_t m_inMedium : 1;
    uint32_t m_isEdf : 1;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE BSDFTemplate() {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const shared::SurfaceMaterial &matData, const TexCoord2D &texCoord, const WavelengthSamples &wls,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        using namespace shared;
        const bool isEdf = flags & BSDFBuildFlags::FromEDF;
        if (isEdf) {
            m_e.isDirectionalEdf = EmitterType(matData.emitterType) == EmitterType::Directional;
        }
        else {
            setupBSDFBody<HARD_CODED_BSDF>(matData, texCoord, wls, reinterpret_cast<uint32_t*>(&m_bsdf), flags);
        }
        m_inMedium = false;
        m_isEdf = isEdf;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const SampledSpectrum &scatteringAlbedo,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        m_m.scatteringAlbedo = scatteringAlbedo;
        //m_m.pf = SchlickPhaseFunction(0.0f);
        m_m.pf = IsotropicPhaseFunction();
        m_inMedium = true;
        m_isEdf = false;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool matches(shared::DirectionType dirType) const {
        using namespace shared;
        if (m_isEdf) {
            const DirectionType type = DirectionType::Emission() |
                (m_e.isDirectionalEdf ? DirectionType::Delta0D() : DirectionType::LowFreq());
            return type.matches(dirType);
        }
        else if (m_inMedium) {
            return m_m.pf.matches(dirType);
        }
        else {
            return m_bsdf.matches(dirType);
        }
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE void getSurfaceParameters(
        SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness) const
    {
        Assert(!m_isEdf, "EDF does not assume this function is called.");
        if (m_inMedium) {
            *diffuseReflectance = m_m.scatteringAlbedo;
            *specularReflectance = SampledSpectrum::Zero();
            *roughness = 0.0f;
            return;
        }

        return m_bsdf.getSurfaceParameters(diffuseReflectance, specularReflectance, roughness);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateDHReflectanceEstimate(
        const shared::BSDFQuery &query) const
    {
        Assert(!m_isEdf, "EDF does not assume this function is called.");
        if (m_inMedium)
            return SampledSpectrum::Zero();
        else
            return m_bsdf.evaluateDHReflectanceEstimate(query);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &smp,
        shared::BSDFQueryResult* result, shared::BSDFQueryReverseResult* revResult = nullptr) const
    {
        Assert(!m_isEdf, "EDF does not assume this function is called.");
        SampledSpectrum fsValue;
        if (m_inMedium)
            fsValue = m_m.scatteringAlbedo * m_m.pf.sampleF(query, smp, result, revResult);
        else
            fsValue = m_bsdf.sampleF(query, smp, result, revResult);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        SampledSpectrum* revValue = nullptr) const
    {
        SampledSpectrum fsValue;
        if (m_isEdf) {
            if (m_e.isDirectionalEdf)
                fsValue = SampledSpectrum::Zero();
            else
                fsValue = vSampled.z > 0.0f ? 1.0f / pi_v<float> : 0.0f;

            if (revValue)
                *revValue = SampledSpectrum::Zero();
        }
        else if (m_inMedium) {
            fsValue = m_m.scatteringAlbedo * m_m.pf.evaluateF(query, vSampled, revValue);
        }
        else {
            fsValue = m_bsdf.evaluateF(query, vSampled, revValue);
        }
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        float* revValue = nullptr) const
    {
        float dirPDensity;
        if (m_isEdf) {
            if (m_e.isDirectionalEdf)
                dirPDensity = 0.0f;
            else
                dirPDensity = vSampled.z > 0.0f ? vSampled.z / pi_v<float> : 0.0f;

            if (revValue)
                *revValue = 0.0f;
        }
        else if (m_inMedium) {
            dirPDensity = m_m.pf.evaluatePDF(query, vSampled, revValue);
        }
        else {
            dirPDensity = m_bsdf.evaluatePDF(query, vSampled, revValue);
        }
        return dirPDensity;
    }

    CUDA_DEVICE_FUNCTION bool isInMedium() const {
        return m_inMedium;
    }
};

template <>
class BSDFTemplate<BSDFGrade::Unidirectional, true> {
    union {
        struct {
            static constexpr uint32_t NumDwords = 16;
            shared::BSDFGetSurfaceParameters getSurfaceParameters;
            shared::BSDFEvaluateDHReflectanceEstimate evaluateDHReflectanceEstimate;
            shared::BSDFMatches matches;
            shared::BSDFSampleF sampleF;
            shared::BSDFEvaluateF evaluateF;
            shared::BSDFEvaluatePDF evaluatePDF;
            uint32_t data[NumDwords];
        } m_s;
        struct {
            //SchlickPhaseFunction pf;
            IsotropicPhaseFunction pf;
            SampledSpectrum scatteringAlbedo;
        } m_m;
    };
    uint32_t m_inMedium : 1;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE BSDFTemplate() {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const shared::SurfaceMaterial &matData, const TexCoord2D &texCoord, const WavelengthSamples &wls,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        using namespace shared;
        Assert(
            flags & BSDFBuildFlags::FromEDF == 0,
            "Unidirectional-grade BSDF does not support construction from EDF.");
        matData.setupBSDFBody(matData, texCoord, m_s.data, flags);
        const shared::BSDFProcedureSet &procSet = getBSDFProcedureSet(matData.bsdfProcSetSlot);
        m_s.getSurfaceParameters = procSet.getSurfaceParameters;
        m_s.evaluateDHReflectanceEstimate = procSet.evaluateDHReflectanceEstimate;
        m_s.matches = procSet.matches;
        m_s.sampleF = procSet.sampleF;
        m_s.evaluateF = procSet.evaluateF;
        m_s.evaluatePDF = procSet.evaluatePDF;
        m_inMedium = false;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const SampledSpectrum scatteringAlbedo,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        m_m.scatteringAlbedo = scatteringAlbedo;
        //m_m.pf = SchlickPhaseFunction(1.0f);
        m_m.pf = IsotropicPhaseFunction();
        m_inMedium = true;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE void getSurfaceParameters(
        SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness) const
    {
        if (m_inMedium) {
            *diffuseReflectance = m_m.scatteringAlbedo;
            *specularReflectance = SampledSpectrum::Zero();
            *roughness = 0.0f;
            return;
        }

        return m_s.getSurfaceParameters(m_s.data, diffuseReflectance, specularReflectance, roughness);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateDHReflectanceEstimate(
        const shared::BSDFQuery &query) const
    {
        if (m_inMedium)
            return SampledSpectrum::Zero();
        else
            return m_s.evaluateDHReflectanceEstimate(m_s.data, query);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool matches(shared::DirectionType dirType) const {
        if (m_inMedium)
            return m_m.pf.matches(dirType);
        else
            return m_s.matches(m_s.data, dirType);
    }

    CUDA_DEVICE_FUNCTION SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &smp,
        shared::BSDFQueryResult* result) const
    {
        SampledSpectrum fsValue;
        if (m_inMedium)
            fsValue = m_m.scatteringAlbedo * m_m.pf.sampleF(query, smp, result);
        else
            fsValue = m_s.sampleF(m_s.data, query, smp, result);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const
    {
        SampledSpectrum fsValue;
        if (m_inMedium)
            fsValue = m_m.scatteringAlbedo * m_m.pf.evaluateF(query, vSampled);
        else
            fsValue = m_s.evaluateF(m_s.data, query, vSampled);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const
    {
        float dirPDensity;
        if (m_inMedium)
            dirPDensity = m_m.pf.evaluatePDF(query, vSampled);
        else
            dirPDensity = m_s.evaluatePDF(m_s.data, query, vSampled);
        return dirPDensity;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool isInMedium() const {
        return m_inMedium;
    }
};

template <>
class BSDFTemplate<BSDFGrade::Bidirectional, true> {
    union {
        struct {
            static constexpr uint32_t NumDwords = 16;
            shared::BSDFGetSurfaceParameters getSurfaceParameters;
            shared::BSDFEvaluateDHReflectanceEstimate evaluateDHReflectanceEstimate;
            shared::BSDFMatches matches;
            shared::BSDFSampleFWithRev sampleFWithRev;
            shared::BSDFEvaluateFWithRev evaluateFWithRev;
            shared::BSDFEvaluatePDFWithRev evaluatePDFWithRev;
            uint32_t data[NumDwords];
        } m_s;
        struct {
            //SchlickPhaseFunction pf;
            IsotropicPhaseFunction pf;
            SampledSpectrum scatteringAlbedo;
        } m_m;
        struct {
            uint32_t isDirectionalEdf : 1;
        } m_e;
    };
    uint32_t m_inMedium : 1;
    uint32_t m_isEdf : 1;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE BSDFTemplate() {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const shared::SurfaceMaterial &matData, const TexCoord2D &texCoord, const WavelengthSamples &wls,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        using namespace shared;
        const bool isEdf = flags & BSDFBuildFlags::FromEDF;
        if (isEdf) {
            m_e.isDirectionalEdf = EmitterType(matData.emitterType) == EmitterType::Directional;
        }
        else {
            matData.setupBSDFBody(matData, texCoord, m_s.data, flags);
            const shared::BSDFProcedureSet &procSet = getBSDFProcedureSet(matData.bsdfProcSetSlot);
            m_s.getSurfaceParameters = procSet.getSurfaceParameters;
            m_s.evaluateDHReflectanceEstimate = procSet.evaluateDHReflectanceEstimate;
            m_s.matches = procSet.matches;
            m_s.sampleFWithRev = procSet.sampleFWithRev;
            m_s.evaluateFWithRev = procSet.evaluateFWithRev;
            m_s.evaluatePDFWithRev = procSet.evaluatePDFWithRev;
        }
        m_inMedium = false;
        m_isEdf = isEdf;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const SampledSpectrum scatteringAlbedo,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        m_m.scatteringAlbedo = scatteringAlbedo;
        //m_m.pf = SchlickPhaseFunction(1.0f);
        m_m.pf = IsotropicPhaseFunction();
        m_inMedium = true;
        m_isEdf = false;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool matches(shared::DirectionType dirType) const {
        using namespace shared;
        if (m_isEdf) {
            const DirectionType type = DirectionType::Emission() |
                (m_e.isDirectionalEdf ? DirectionType::Delta0D() : DirectionType::LowFreq());
            return type.matches(dirType);
        }
        else if (m_inMedium)
            return m_m.pf.matches(dirType);
        else
            return m_s.matches(m_s.data, dirType);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE void getSurfaceParameters(
        SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness) const
    {
        Assert(!m_isEdf, "EDF does not assume this function is called.");
        if (m_inMedium) {
            *diffuseReflectance = m_m.scatteringAlbedo;
            *specularReflectance = SampledSpectrum::Zero();
            *roughness = 0.0f;
            return;
        }

        return m_s.getSurfaceParameters(m_s.data, diffuseReflectance, specularReflectance, roughness);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateDHReflectanceEstimate(
        const shared::BSDFQuery &query) const
    {
        Assert(!m_isEdf, "EDF does not assume this function is called.");
        if (m_inMedium)
            return SampledSpectrum::Zero();
        else
            return m_s.evaluateDHReflectanceEstimate(m_s.data, query);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &smp,
        shared::BSDFQueryResult* result, shared::BSDFQueryReverseResult* revResult) const
    {
        Assert(!m_isEdf, "EDF does not assume this function is called.");
        SampledSpectrum fsValue;
        if (m_inMedium)
            fsValue = m_m.scatteringAlbedo * m_m.pf.sampleF(query, smp, result, revResult);
        else
            fsValue = m_s.sampleFWithRev(m_s.data, query, smp, result, revResult);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        SampledSpectrum* revValue) const
    {
        SampledSpectrum fsValue;
        if (m_isEdf) {
            if (m_e.isDirectionalEdf)
                fsValue = SampledSpectrum::Zero();
            else
                fsValue = vSampled.z > 0.0f ? 1.0f / pi_v<float> : 0.0f;

            if (revValue)
                *revValue = SampledSpectrum::Zero();
        }
        else if (m_inMedium) {
            fsValue = m_m.scatteringAlbedo * m_m.pf.evaluateF(query, vSampled, revValue);
        }
        else {
            fsValue = m_s.evaluateFWithRev(m_s.data, query, vSampled, revValue);
        }
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        float* revValue) const
    {
        float dirPDensity;
        if (m_isEdf) {
            if (m_e.isDirectionalEdf)
                dirPDensity = 0.0f;
            else
                dirPDensity = vSampled.z > 0.0f ? vSampled.z / pi_v<float> : 0.0f;

            if (revValue)
                *revValue = 0.0f;
        }
        else if (m_inMedium) {
            dirPDensity = m_m.pf.evaluatePDF(query, vSampled, revValue);
        }
        else {
            dirPDensity = m_s.evaluatePDFWithRev(m_s.data, query, vSampled, revValue);
        }
        return dirPDensity;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool isInMedium() const {
        return m_inMedium;
    }
};



template <shared::TransportMode transportMode, BSDFGrade bsdfGrade>
class BSDF;

template <shared::TransportMode transportMode>
class BSDF<transportMode, BSDFGrade::Unidirectional> {
    BSDFTemplate<BSDFGrade::Unidirectional, useGenericBSDF> m_body;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const shared::SurfaceMaterial &matData, const TexCoord2D &texCoord, const WavelengthSamples &wls,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        m_body.setup(matData, texCoord, wls, flags);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const SampledSpectrum &scatteringAlbedo,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        m_body.setup(scatteringAlbedo, flags);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool hasNonDelta() const {
        using namespace shared;
        return m_body.matches(DirectionType::WholeSphere() | DirectionType::NonDelta());
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE void getSurfaceParameters(
        SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness) const
    {
        return m_body.getSurfaceParameters(diffuseReflectance, specularReflectance, roughness);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateDHReflectanceEstimate(
        const shared::BSDFQuery &query) const
    {
        return m_body.evaluateDHReflectanceEstimate(query);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &smp,
        shared::BSDFQueryResult* result) const
    {
        using namespace shared;
        SampledSpectrum fsValue = m_body.sampleF(query, smp, result);
        float snCorrection = 1.0f;
        if (!m_body.isInMedium()) {
            const Vector3D dirLocal = transportMode == TransportMode::Radiance ?
                result->dirLocal : query.dirLocal;
            snCorrection = std::fabs(dirLocal.z / dot(dirLocal, query.geometricNormalLocal));
            if (!isfinite(snCorrection))
                return SampledSpectrum::Zero();
            fsValue *= snCorrection;
        }
        Assert(
            (result->dirPDensity > 0 && fsValue.allNonNegativeFinite()) ||
            result->dirPDensity == 0,
            "sampleBSDF: smp: (%g, %g), qDir: (%g, %g, %g), gNormal: (%g, %g, %g)"
            "rDir: (%g, %g, %g), dirPDF: %g, "
            "snCorrection: %g",
            smp.uDir[0], smp.uDir[1],
            v3print(query.dirLocal), v3print(query.geometricNormalLocal),
            v3print(result->dirLocal), result->dirPDensity,
            snCorrection);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const
    {
        using namespace shared;
        SampledSpectrum fsValue = m_body.evaluateF(query, vSampled);
        float snCorrection = 1.0f;
        if (!m_body.isInMedium()) {
            const Vector3D dirLocal = transportMode == TransportMode::Radiance ?
                vSampled : query.dirLocal;
            snCorrection = std::fabs(dirLocal.z / dot(dirLocal, query.geometricNormalLocal));
            if (!isfinite(snCorrection))
                return SampledSpectrum::Zero();
            fsValue *= snCorrection;
        }
        Assert(
            fsValue.allNonNegativeFinite(),
            "evalBSDF: qDir: (%g, %g, %g), gNormal: (%g, %g, %g), "
            "rDir: (%g, %g, %g), "
            "snCorrection: %g",
            v3print(query.dirLocal), v3print(query.geometricNormalLocal),
            v3print(vSampled),
            snCorrection);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const
    {
        return m_body.evaluatePDF(query, vSampled);
    }
};

template <shared::TransportMode transportMode>
class BSDF<transportMode, BSDFGrade::Bidirectional> {
    BSDFTemplate<BSDFGrade::Bidirectional, useGenericBSDF> m_body;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const shared::SurfaceMaterial &matData, const TexCoord2D &texCoord, const WavelengthSamples &wls,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        m_body.setup(matData, texCoord, wls, flags);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE void setup(
        const SampledSpectrum &scatteringAlbedo,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None)
    {
        m_body.setup(scatteringAlbedo, flags);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE void getSurfaceParameters(
        SampledSpectrum* diffuseReflectance, SampledSpectrum* specularReflectance, float* roughness) const
    {
        return m_body.getSurfaceParameters(diffuseReflectance, specularReflectance, roughness);
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateDHReflectanceEstimate(
        const shared::BSDFQuery &query) const
    {
        return m_body.evaluateDHReflectanceEstimate(query);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE bool hasNonDelta() const {
        using namespace shared;
        return m_body.matches(DirectionType::WholeSphere() | DirectionType::NonDelta());
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &smp,
        shared::BSDFQueryResult* result, shared::BSDFQueryReverseResult* revResult) const
    {
        using namespace shared;
        SampledSpectrum fsValue = m_body.sampleF(query, smp, result, revResult);
        float snCorrection = 1.0f;
        if (!m_body.isInMedium()) {
            const Vector3D dirLocal = transportMode == TransportMode::Radiance ?
                result->dirLocal : query.dirLocal;
            snCorrection = std::fabs(dirLocal.z / dot(dirLocal, query.geometricNormalLocal));
            if (!isfinite(snCorrection))
                return SampledSpectrum::Zero();
            fsValue *= snCorrection;
            revResult->fsValue *= snCorrection;
        }
        Assert(
            (result->dirPDensity > 0 && fsValue.allNonNegativeFinite()) ||
            result->dirPDensity == 0,
            "sampleBSDF: smp: (%g, %g), qDir: (%g, %g, %g), gNormal: (%g, %g, %g)"
            "rDir: (%g, %g, %g), dirPDF: %g, "
            "snCorrection: %g",
            smp.uDir[0], smp.uDir[1],
            v3print(query.dirLocal), v3print(query.geometricNormalLocal),
            v3print(result->dirLocal), result->dirPDensity,
            snCorrection);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE SampledSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        SampledSpectrum* revValue) const
    {
        using namespace shared;
        SampledSpectrum fsValue = m_body.evaluateF(query, vSampled, revValue);
        float snCorrection = 1.0f;
        if (!m_body.isInMedium()) {
            const Vector3D dirLocal = transportMode == TransportMode::Radiance ?
                vSampled : query.dirLocal;
            snCorrection = std::fabs(dirLocal.z / dot(dirLocal, query.geometricNormalLocal));
            if (!isfinite(snCorrection))
                return SampledSpectrum::Zero();
            fsValue *= snCorrection;
            *revValue *= snCorrection;
        }
        Assert(
            fsValue.allNonNegativeFinite(),
            "evalBSDF: qDir: (%g, %g, %g), gNormal: (%g, %g, %g), "
            "rDir: (%g, %g, %g), "
            "snCorrection: %g",
            v3print(query.dirLocal), v3print(query.geometricNormalLocal),
            v3print(vSampled),
            snCorrection);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled,
        float* revValue) const
    {
        return m_body.evaluatePDF(query, vSampled, revValue);
    }
};



CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMinMax_block(
    shared::BoundingBox3DAsOrderedInt* dstAabb, const shared::BoundingBox3DAsOrderedInt &aabb) {
    atomicMin_block(&dstAabb->minP.x, aabb.minP.x);
    atomicMin_block(&dstAabb->minP.y, aabb.minP.y);
    atomicMin_block(&dstAabb->minP.z, aabb.minP.z);
    atomicMax_block(&dstAabb->maxP.x, aabb.maxP.x);
    atomicMax_block(&dstAabb->maxP.y, aabb.maxP.y);
    atomicMax_block(&dstAabb->maxP.z, aabb.maxP.z);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMinMax(
    shared::BoundingBox3DAsOrderedInt* dstAabb, const shared::BoundingBox3DAsOrderedInt &aabb) {
    atomicMin(&dstAabb->minP.x, aabb.minP.x);
    atomicMin(&dstAabb->minP.y, aabb.minP.y);
    atomicMin(&dstAabb->minP.z, aabb.minP.z);
    atomicMax(&dstAabb->maxP.x, aabb.maxP.x);
    atomicMax(&dstAabb->maxP.y, aabb.maxP.y);
    atomicMax(&dstAabb->maxP.z, aabb.maxP.z);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMin_RGBSpectrum_block(
    shared::RGBSpectrumAsOrderedInt* dstValue, const shared::RGBSpectrumAsOrderedInt &value) {
    atomicMin_block(&dstValue->r, value.r);
    atomicMin_block(&dstValue->g, value.g);
    atomicMin_block(&dstValue->b, value.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMax_RGBSpectrum_block(
    shared::RGBSpectrumAsOrderedInt* dstValue, const shared::RGBSpectrumAsOrderedInt &value) {
    atomicMax_block(&dstValue->r, value.r);
    atomicMax_block(&dstValue->g, value.g);
    atomicMax_block(&dstValue->b, value.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicAdd_RGBSpectrum_block(
    RGBSpectrum* dstValue, const RGBSpectrum &value) {
    atomicAdd_block(&dstValue->r, value.r);
    atomicAdd_block(&dstValue->g, value.g);
    atomicAdd_block(&dstValue->b, value.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMin_RGBSpectrum(
    shared::RGBSpectrumAsOrderedInt* dstValue, const shared::RGBSpectrumAsOrderedInt &value) {
    atomicMin(&dstValue->r, value.r);
    atomicMin(&dstValue->g, value.g);
    atomicMin(&dstValue->b, value.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicMax_RGBSpectrum(
    shared::RGBSpectrumAsOrderedInt* dstValue, const shared::RGBSpectrumAsOrderedInt &value) {
    atomicMax(&dstValue->r, value.r);
    atomicMax(&dstValue->g, value.g);
    atomicMax(&dstValue->b, value.b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void atomicAdd_RGBSpectrum(
    RGBSpectrum* dstValue, const RGBSpectrum &value) {
    atomicAdd(&dstValue->r, value.r);
    atomicAdd(&dstValue->g, value.g);
    atomicAdd(&dstValue->b, value.b);
}

} // namespace rtc10::device



namespace rtc10::shared {

CUDA_DEVICE_FUNCTION SampledSpectrum ImageBasedEnvironmentalLight::sample(
    const WavelengthSamples &wls,
    float uDir0, float uDir1, float* phi, float* theta, float* dirPDensity) const {
    float prob;
    uint2 pix = importanceMap.sample(uDir0, uDir1, &uDir0, &uDir1, &prob);
    float2 uv = make_float2((pix.x + uDir0) / imageWidth,
                            (pix.y + uDir1) / imageHeight);
    float uvPDensity = (imageWidth * imageHeight) * prob;
    float u = uv.x, v = uv.y;
    *phi = 2 * pi_v<float> * u;
    *theta = pi_v<float> * v;

    *dirPDensity = uvPDensity / (2 * pow2(pi_v<float>) * std::sin(*theta));
    if (!isfinite(*dirPDensity)) {
        *dirPDensity = 0.0f;
        return SampledSpectrum::Zero();
    }

    float4 texValue = tex2DLod<float4>(texObj, u, v, 0.0f);
    TripletSpectrum spRadiance = createTripletSpectrum(
        SpectrumType::LightSource, ColorSpace::Rec709_D65, texValue.x, texValue.y, texValue.z);
    SampledSpectrum ret = spRadiance.evaluate(wls);

    return ret;
}

CUDA_DEVICE_FUNCTION SampledSpectrum ImageBasedEnvironmentalLight::evaluate(
    const WavelengthSamples &wls,
    float phi, float theta, float* hypDirPDensity) const {
    float2 texCoord = make_float2(phi / (2 * pi_v<float>), theta / pi_v<float>);
    float4 texValue = tex2DLod<float4>(texObj, texCoord.x, texCoord.y, 0.0f);
    TripletSpectrum spRadiance = createTripletSpectrum(
        SpectrumType::LightSource, ColorSpace::Rec709_D65, texValue.x, texValue.y, texValue.z);
    SampledSpectrum ret = spRadiance.evaluate(wls);

    uint2 pix = make_uint2(imageWidth * texCoord.x, imageHeight * texCoord.y);
    float prob = importanceMap.evaluate(pix);
    float uvPDensity = (imageWidth * imageHeight) * prob;
    *hypDirPDensity = uvPDensity / (2 * pow2(pi_v<float>) * std::sin(theta));
    if (!isfinite(*hypDirPDensity)) {
        *hypDirPDensity = 0.0f;
        return SampledSpectrum::Zero();
    }

    return ret;
}

RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(ImageBasedEnvironmentalLight_sample)(
    const uint32_t* data, const WavelengthSamples &wls,
    float uDir0, float uDir1, float* phi, float* theta, float* dirPDensity) {
    auto &envLight = *reinterpret_cast<const ImageBasedEnvironmentalLight*>(data);
    return envLight.sample(wls, uDir0, uDir1, phi, theta, dirPDensity);
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(ImageBasedEnvironmentalLight_sample);
RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(ImageBasedEnvironmentalLight_evaluate)(
    const uint32_t* data, const WavelengthSamples &wls,
    float phi, float theta, float* hypDirPDensity) {
    auto &envLight = *reinterpret_cast<const ImageBasedEnvironmentalLight*>(data);
    return envLight.evaluate(wls, phi, theta, hypDirPDensity);
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(ImageBasedEnvironmentalLight_evaluate);

} // namespace rtc8::shared

#endif // #if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)
