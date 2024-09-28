#pragma once

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define RTC9_Platform_Windows
#    if defined(_MSC_VER)
#        define RTC9_Platform_Windows_MSVC
#       if defined(__INTELLISENSE__)
#           define RTC9_Platform_CodeCompletion
#       endif
#    endif
#elif defined(__APPLE__)
#    define RTC9_Platform_macOS
#endif

#ifdef _DEBUG
#   define ENABLE_ASSERT
#   define DEBUG_SELECT(A, B) A
#else
#   define DEBUG_SELECT(A, B) B
#endif



#if defined(RTC9_Platform_Windows_MSVC)
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#   undef RGB
#endif

// #includes
#if defined(__CUDA_ARCH__)
#else
#   include <cstdio>
#   include <cstdlib>
#   include <cstdint>
#   include <cmath>

#   include <numbers>
#   include <limits>
#   include <algorithm>

#   include <immintrin.h>
#endif

#include "../common/utils/cuda_util.h"

#if defined(__CUDA_ARCH__)
#   define HOST_STATIC_CONSTEXPR
#else
#   define HOST_STATIC_CONSTEXPR static constexpr
#endif



namespace rtc9 {

#ifdef RTC9_Platform_Windows_MSVC
#   if defined(__CUDA_ARCH__)
#       define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   else
void devPrintf(const char* fmt, ...);
#   endif
#else
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

#ifdef ENABLE_ASSERT
#   if defined(__CUDA_ARCH__)
#       define Assert(expr, fmt, ...) do { if (!(expr)) { printf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); printf(fmt"\n", ##__VA_ARGS__); /*assert(false)*/; } } while (0)
#   else
#       define Assert(expr, fmt, ...) do { if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } } while (0)
#   endif
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_ShouldNotBeCalled() Assert(false, "Should not be called!")
#define Assert_NotImplemented() Assert(false, "Not implemented yet!")



template <typename T, size_t size>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr size_t lengthof(const T (&array)[size]) {
    return size;
}



template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T alignUp(T value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t tzcnt(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __clz(__brev(x));
#else
    return _tzcnt_u32(x);
#endif
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t lzcnt(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __clz(x);
#else
    return _lzcnt_u32(x);
#endif
}

CUDA_COMMON_FUNCTION CUDA_INLINE int32_t popcnt(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#else
    return _mm_popcnt_u32(x);
#endif
}

//     0: 0
//     1: 0
//  2- 3: 1
//  4- 7: 2
//  8-15: 3
// 16-31: 4
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowOf2Exponent(uint32_t x) {
    if (x == 0)
        return 0;
    return 31 - lzcnt(x);
}

//    0: 0
//    1: 0
//    2: 1
// 3- 4: 2
// 5- 8: 3
// 9-16: 4
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowOf2Exponent(uint32_t x) {
    if (x == 0)
        return 0;
    return 32 - lzcnt(x - 1);
}

//     0: 0
//     1: 1
//  2- 3: 2
//  4- 7: 4
//  8-15: 8
// 16-31: 16
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowerOf2(uint32_t x) {
    if (x == 0)
        return 0;
    return 1 << prevPowOf2Exponent(x);
}

//    0: 0
//    1: 1
//    2: 2
// 3- 4: 4
// 5- 8: 8
// 9-16: 16
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowerOf2(uint32_t x) {
    if (x == 0)
        return 0;
    return 1 << nextPowOf2Exponent(x);
}

template <typename IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType nextMultiplesForPowOf2(IntType x, uint32_t exponent) {
    IntType mask = (1 << exponent) - 1;
    return (x + mask) & ~mask;
}

template <typename IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType nextMultiplierForPowOf2(IntType x, uint32_t exponent) {
    return nextMultiplesForPowOf2(x, exponent) >> exponent;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nthSetBit(uint32_t value, int32_t n) {
    uint32_t idx = 0;
    int32_t count;
    if (n >= popcnt(value))
        return 0xFFFFFFFF;

    for (uint32_t width = 16; width >= 1; width >>= 1) {
        if (value == 0)
            return 0xFFFFFFFF;

        uint32_t mask = (1 << width) - 1;
        count = popcnt(value & mask);
        if (n >= count) {
            value >>= width;
            n -= count;
            idx += width;
        }
    }

    return idx;
}



template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T min(const T &a, const T &b) {
    return b < a ? b : a;
}
template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T max(const T &a, const T &b) {
    return b > a ? b : a;
}
template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T clamp(const T &v, const T &minv, const T &maxv) {
    return min(max(v, minv), maxv);
}

template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void _swap(T &a, T &b) {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow2(const T &x) {
    return x * x;
}
template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow3(const T &x) {
    return x * pow2(x);
}
template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow4(const T &x) {
    return pow2(pow2(x));
}
template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow5(const T &x) {
    return x * pow4(x);
}

template <typename T, typename RealType>
CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr T lerp(const T &v0, const T &v1, RealType t) {
    return (1 - t) * v0 + t * v1;
}

template <typename RealType>
CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr RealType safeDivide(RealType a, RealType b) {
    return b != 0 ? a / b : static_cast<RealType>(0);
}

}
