#pragma once

#include "spectrum_base.h"

#define RGBFMT "%g, %g, %g"
#define rgbprint(sp) sp.r, sp.g, sp.b

namespace rtc10 {

template <typename RealType>
struct RGBWavelengthSamplesTemplate {
    struct {
        unsigned int _selectedLambdaIndex : 16;
        unsigned int _singleIsSelected : 1;
    };

    CUDA_DEVICE_FUNCTION RGBWavelengthSamplesTemplate() {}
    CUDA_DEVICE_FUNCTION RGBWavelengthSamplesTemplate(const RGBWavelengthSamplesTemplate &wls) {
        _selectedLambdaIndex = wls._selectedLambdaIndex;
        _singleIsSelected = wls._singleIsSelected;
    }

    CUDA_DEVICE_FUNCTION uint32_t selectedLambdaIndex() const {
        return _selectedLambdaIndex;
    }
    CUDA_DEVICE_FUNCTION void setSelectedLambdaIndex(uint32_t index) const {
        _selectedLambdaIndex = index;
    }
    CUDA_DEVICE_FUNCTION bool singleIsSelected() const {
        return _singleIsSelected;
    }
    CUDA_DEVICE_FUNCTION void setSingleIsSelected() {
        _singleIsSelected = true;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE static RGBWavelengthSamplesTemplate createWithEqualOffsets(
        RealType offset, RealType uLambda, RealType* PDF) {
        Assert(offset >= 0 && offset < 1, "\"offset\" must be in range [0, 1).");
        Assert(uLambda >= 0 && uLambda < 1, "\"uLambda\" must be in range [0, 1).");
        RGBWavelengthSamplesTemplate wls;
        wls._selectedLambdaIndex = rtc10::min<uint16_t>(3 * uLambda, 3 - 1);
        wls._singleIsSelected = false;
        *PDF = 1;
        return wls;
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr uint32_t NumComponents() {
        return 3;
    }
};



template <typename RealType>
struct RGBSpectrumTemplate {
    RealType r, g, b;

    CUDA_COMMON_FUNCTION RGBSpectrumTemplate() {}
    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate(RealType v) :
        r(v), g(v), b(v) {}
    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate(RealType rr, RealType gg, RealType bb) :
        r(rr), g(gg), b(bb) {}
    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate(RealType rgb[3]) :
        r(rgb[0]), g(rgb[1]), b(rgb[2]) {}

    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate operator-() const {
        return RGBSpectrumTemplate(-r, -g, -b);
    }

    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate &operator+=(const RGBSpectrumTemplate &v) {
        r += v.r;
        g += v.g;
        b += v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate &operator-=(const RGBSpectrumTemplate &v) {
        r -= v.r;
        g -= v.g;
        b -= v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate &operator*=(const RGBSpectrumTemplate &v) {
        r *= v.r;
        g *= v.g;
        b *= v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate &operator*=(RealType s) {
        r *= s;
        g *= s;
        b *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate &operator/=(const RGBSpectrumTemplate &v) {
        r /= v.r;
        g /= v.g;
        b /= v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBSpectrumTemplate &operator/=(RealType s) {
        RealType r = static_cast<RealType>(1.0) / s;
        return *this *= r;
    }

    CUDA_COMMON_FUNCTION constexpr RealType &operator[](uint32_t ch) {
        Assert(ch <= 2, "\"ch\" is out of range [0, 2].");
        return *(&r + ch);
    }
    CUDA_COMMON_FUNCTION constexpr const RealType &operator[](uint32_t ch) const {
        Assert(ch <= 2, "\"ch\" is out of range [0, 2].");
        return *(&r + ch);
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNan() const {
        using rtc10::isnan;
        return isnan(r) || isnan(g) || isnan(b);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        using rtc10::isinf;
        return isinf(r) || isinf(g) || isinf(b);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        using rtc10::isfinite;
        return isfinite(r) && isfinite(g) && isfinite(b);
    }
    CUDA_COMMON_FUNCTION constexpr bool hasNonZero() const {
        return r != 0 || g != 0 || b != 0;
    }
    CUDA_COMMON_FUNCTION constexpr bool hasNegative() const {
        return r < 0 || g < 0 || b < 0;
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allNonNegativeFinite() const {
        return !hasNegative() && allFinite();
    }

    CUDA_COMMON_FUNCTION constexpr RealType min() const {
        return std::fmin(r, std::fmin(g, b));
    }
    CUDA_COMMON_FUNCTION constexpr RealType max() const {
        return std::fmax(r, std::fmax(g, b));
    }
    CUDA_COMMON_FUNCTION constexpr RealType luminance() const {
        return
            static_cast<RealType>(0.2126729) * r +
            static_cast<RealType>(0.7151522) * g +
            static_cast<RealType>(0.0721750) * b;
    }

    // setting "primary" to 1.0 might introduce bias.
    CUDA_DEVICE_FUNCTION RealType importance(uint16_t selectedLambda) const {
        RealType sum = r + g + b;
        const RealType primary = 0.9f;
        const RealType marginal = (1 - primary) / 2;
        return sum * marginal + (*this)[selectedLambda] * (primary - marginal);
    }

    CUDA_DEVICE_FUNCTION const RGBSpectrumTemplate &evaluate(
        const RGBWavelengthSamplesTemplate<RealType> &wls) const {
        return *this;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBSpectrumTemplate Zero() {
        return RGBSpectrumTemplate(0, 0, 0);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBSpectrumTemplate One() {
        return RGBSpectrumTemplate(1, 1, 1);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBSpectrumTemplate Infinity() {
        return RGBSpectrumTemplate(INFINITY, INFINITY, INFINITY);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBSpectrumTemplate NaN() {
        return RGBSpectrumTemplate(NAN, NAN, NAN);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBSpectrumTemplate fromXYZ(RealType xyz[3]) {
        RGBSpectrumTemplate ret(
            static_cast<RealType>(3.2404542) * xyz[0] +
            static_cast<RealType>(-1.5371385) * xyz[1] +
            static_cast<RealType>(-0.4985314) * xyz[2],
            static_cast<RealType>(-0.9692660) * xyz[0] +
            static_cast<RealType>(1.8760108) * xyz[1] +
            static_cast<RealType>(0.0415560) * xyz[2],
            static_cast<RealType>(0.0556434) * xyz[0] +
            static_cast<RealType>(-0.2040259) * xyz[1] +
            static_cast<RealType>(1.0572252) * xyz[2]);
        return ret;
    }

    // ----------------------------------------------------------------
    // Methods for compatibility with DiscretizedSpectrumTemplate

    CUDA_DEVICE_FUNCTION void toXYZ(RealType XYZ[3]) const {
        const RealType RGB[3] = { r, g, b };
        transformFromRenderingRGB(SpectrumType::LightSource, RGB, XYZ);
    }

    CUDA_DEVICE_FUNCTION RGBSpectrumTemplate &add(const RGBWavelengthSamplesTemplate<RealType> &wls, const RGBSpectrumTemplate<RealType> &val) {
        *this += val;
        return *this;
    }

#if defined(__CUDA_ARCH__) || defined(RTC10_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION void atomicAdd(const RGBWavelengthSamplesTemplate<RealType> &wls, const RGBSpectrumTemplate<RealType> &val) {
        ::atomicAdd(&r, val.r);
        ::atomicAdd(&g, val.g);
        ::atomicAdd(&b, val.b);
    }
#endif

    // ----------------------------------------------------------------
};

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator==(
    const RGBSpectrumTemplate<RealType> &va, const RGBSpectrumTemplate<RealType> &vb) {
    return va.r == vb.r && va.g == vb.g && va.b == vb.b;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!=(
    const RGBSpectrumTemplate<RealType> &va, const RGBSpectrumTemplate<RealType> &vb) {
    return va.r != vb.r || va.g != vb.g || va.b != vb.b;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> operator+(
    const RGBSpectrumTemplate<RealType> &va, const RGBSpectrumTemplate<RealType> &vb) {
    RGBSpectrumTemplate<RealType> ret = va;
    ret += vb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> operator-(
    const RGBSpectrumTemplate<RealType> &va, const RGBSpectrumTemplate<RealType> &vb) {
    RGBSpectrumTemplate<RealType> ret = va;
    ret -= vb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> operator*(
    const RGBSpectrumTemplate<RealType> &va, const RGBSpectrumTemplate<RealType> &vb) {
    RGBSpectrumTemplate<RealType> ret = va;
    ret *= vb;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> operator*(
    const RGBSpectrumTemplate<RealType> &v, ScalarType s) {
    RGBSpectrumTemplate<RealType> ret = v;
    ret *= s;
    return ret;
}

template <typename ScalarType, typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> operator*(
    ScalarType s, const RGBSpectrumTemplate<RealType> &v) {
    RGBSpectrumTemplate<RealType> ret = v;
    ret *= s;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> operator/(
    const RGBSpectrumTemplate<RealType> &v, ScalarType s) {
    RGBSpectrumTemplate<RealType> ret = v;
    ret /= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> min(
    const RGBSpectrumTemplate<RealType> &va, const RGBSpectrumTemplate<RealType> &vb) {
    return RGBSpectrumTemplate<RealType>(
        std::fmin(va.r, vb.r),
        std::fmin(va.g, vb.g),
        std::fmin(va.b, vb.b));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> max(
    const RGBSpectrumTemplate<RealType> &va, const RGBSpectrumTemplate<RealType> &vb) {
    return RGBSpectrumTemplate<RealType>(
        std::fmax(va.r, vb.r),
        std::fmax(va.g, vb.g),
        std::fmax(va.b, vb.b));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> lerp(
    const RGBSpectrumTemplate<RealType> &va, const RGBSpectrumTemplate<RealType> &vb, RealType t) {
    return (1 - t) * va + t * vb;
}

template <typename RealType>
CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> safeDivide(
    const RGBSpectrumTemplate<RealType> &a, const RGBSpectrumTemplate<RealType> &b) {
    RealType zero = static_cast<RealType>(0);
    return RGBSpectrumTemplate<RealType>(
        b.r != 0 ? a.r / b.r : zero,
        b.g != 0 ? a.g / b.g : zero,
        b.b != 0 ? a.b / b.b : zero);
}

template <typename RealType>
CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr RGBSpectrumTemplate<RealType> safeDivide(
    const RGBSpectrumTemplate<RealType> &a, RealType b) {
    RealType zero = static_cast<RealType>(0);
    return RGBSpectrumTemplate<RealType>(
        b != 0 ? a.r / b : zero,
        b != 0 ? a.g / b : zero,
        b != 0 ? a.b / b : zero);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RGBSpectrumTemplate<RealType> HSVtoRGB(
    RealType h, RealType s, RealType v) {
    if (s == 0)
        return RGBSpectrumTemplate<RealType>(v, v, v);

    h = h - std::floor(h);
    int32_t hi = static_cast<int32_t>(h * 6);
    RealType f = h * 6 - hi;
    RealType m = v * (1 - s);
    RealType n = v * (1 - s * f);
    RealType k = v * (1 - s * (1 - f));
    if (hi == 0)
        return RGBSpectrumTemplate<RealType>(v, k, m);
    else if (hi == 1)
        return RGBSpectrumTemplate<RealType>(n, v, m);
    else if (hi == 2)
        return RGBSpectrumTemplate<RealType>(m, v, k);
    else if (hi == 3)
        return RGBSpectrumTemplate<RealType>(m, n, v);
    else if (hi == 4)
        return RGBSpectrumTemplate<RealType>(k, m, v);
    else if (hi == 5)
        return RGBSpectrumTemplate<RealType>(v, m, n);
    return RGBSpectrumTemplate<RealType>(0, 0, 0);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE void RGBtoHSV(
    RGBSpectrumTemplate<RealType> &input, RealType* h, RealType* s, RealType* v) {
    RealType minV = min(min(input.r, input.g), input.b);
    RealType maxV = max(max(input.r, input.g), input.b);

    *v = maxV;
    RealType delta = maxV - minV;
    if (delta < 1e-5f || maxV == 0.0f) {
        *h = 0.0f;
        *s = 0.0f;
        return;
    }

    *s = delta / maxV;

    if (input.r >= maxV)
        *h = (input.g - input.b) / delta;
    else if (input.g >= maxV)
        *h = 2.0f + (input.b - input.r) / delta;
    else
        *h = 4.0f + (input.r - input.g) / delta;
    *h /= 6;
    *h = std::fmod(*h + 1.0f, 1.0f);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RealType simpleToneMap_s(RealType value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    return static_cast<RealType>(1) - std::exp(-value);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RGBSpectrumTemplate<RealType> sRGB_degamma(
    const RGBSpectrumTemplate<RealType> &value) {
    return RGBSpectrumTemplate<RealType>(
        sRGB_degamma(value.r),
        sRGB_degamma(value.g),
        sRGB_degamma(value.b));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RGBSpectrumTemplate<RealType> sRGB_gamma(
    const RGBSpectrumTemplate<RealType> &value) {
    return RGBSpectrumTemplate<RealType>(
        sRGB_gamma(value.r),
        sRGB_gamma(value.g),
        sRGB_gamma(value.b));
}



template <typename RealType>
class RGBStorageTemplate {
    typedef RGBSpectrumTemplate<RealType> ValueType;
    CompensatedSum<ValueType> value;

public:
    CUDA_DEVICE_FUNCTION RGBStorageTemplate(const ValueType &v = ValueType::Zero()) : value(v) {}

    CUDA_DEVICE_FUNCTION void reset() {
        value = CompensatedSum<ValueType>(ValueType::Zero());
    }

    CUDA_DEVICE_FUNCTION RGBStorageTemplate &add(
        const RGBWavelengthSamplesTemplate<RealType> &wls, const RGBSpectrumTemplate<RealType> &val) {
        value += val;
        return *this;
    }

    CUDA_DEVICE_FUNCTION RGBStorageTemplate &add(const RGBSpectrumTemplate<RealType> &val) {
        value += val;
        return *this;
    }

    CUDA_DEVICE_FUNCTION const CompensatedSum<ValueType> &getValue() const {
        return value;
    }
    CUDA_DEVICE_FUNCTION CompensatedSum<ValueType> &getValue() {
        return value;
    }
};



using RGBSpectrum = RGBSpectrumTemplate<float>;

} // namespace rtc10
