#pragma once

#include "common_host.h"
#include "common_renderer_types.h"

namespace rtc10 {

template <typename RealType, bool useAliasTable>
class DiscreteDistribution1DTemplate;

template <typename RealType>
class DiscreteDistribution1DTemplate<RealType, false> {
    cudau::TypedBuffer<RealType> m_weights;
    cudau::TypedBuffer<RealType> m_CDF;
    RealType m_integral;
    uint32_t m_numValues;
    unsigned int m_isInitialized : 1;

public:
    using DeviceType = shared::DiscreteDistribution1DTemplate<RealType, false>;

    DiscreteDistribution1DTemplate() :
        m_isInitialized(false), m_integral(0.0f) {}
    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues);
    void finalize() {
        if (!m_isInitialized)
            return;
        if (m_CDF.isInitialized() && m_weights.isInitialized()) {
            m_CDF.finalize();
            m_weights.finalize();
        }
    }

    DiscreteDistribution1DTemplate &operator=(DiscreteDistribution1DTemplate &&v) {
        m_weights = std::move(v.m_weights);
        m_CDF = std::move(v.m_CDF);
        m_integral = v.m_integral;
        m_numValues = v.m_numValues;
        return *this;
    }

    RealType getIntengral() const {
        return m_integral;
    }

    bool isInitialized() const {
        return m_isInitialized;
    }

    void getDeviceType(DeviceType* instance) const {
        new (instance) DeviceType(
            m_weights.isInitialized() ? m_weights.getDevicePointer() : nullptr,
            m_CDF.isInitialized() ? m_CDF.getDevicePointer() : nullptr,
            m_integral, m_numValues);
    }

    RealType* weightsOnDevice() const {
        return m_weights.getDevicePointer();
    }

    RealType* cdfOnDevice() const {
        return m_CDF.getDevicePointer();
    }
};

template <typename RealType>
class DiscreteDistribution1DTemplate<RealType, true> {
    cudau::TypedBuffer<RealType> m_weights;
    cudau::TypedBuffer<shared::AliasTableEntry<RealType>> m_aliasTable;
    cudau::TypedBuffer<shared::AliasValueMap<RealType>> m_valueMaps;
    RealType m_integral;
    uint32_t m_numValues;
    unsigned int m_isInitialized : 1;

public:
    using DeviceType = shared::DiscreteDistribution1DTemplate<RealType, true>;

    DiscreteDistribution1DTemplate() :
        m_isInitialized(false), m_integral(0.0f) {}
    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues);
    void finalize() {
        if (!m_isInitialized)
            return;
        if (m_valueMaps.isInitialized() && m_aliasTable.isInitialized() && m_weights.isInitialized()) {
            m_valueMaps.finalize();
            m_aliasTable.finalize();
            m_weights.finalize();
        }
    }

    DiscreteDistribution1DTemplate &operator=(DiscreteDistribution1DTemplate &&v) {
        m_weights = std::move(v.m_weights);
        m_aliasTable = std::move(v.m_aliasTable);
        m_valueMaps = std::move(v.m_valueMaps);
        m_integral = v.m_integral;
        m_numValues = v.m_numValues;
        return *this;
    }

    RealType getIntengral() const {
        return m_integral;
    }

    bool isInitialized() const {
        return m_isInitialized;
    }

    void getDeviceType(DeviceType* instance) const {
        new (instance) DeviceType(
            m_weights.isInitialized() ? m_weights.getDevicePointer() : nullptr,
            m_aliasTable.isInitialized() ? m_aliasTable.getDevicePointer() : nullptr,
            m_valueMaps.isInitialized() ? m_valueMaps.getDevicePointer() : nullptr,
            m_integral, m_numValues);
    }

    RealType* weightsOnDevice() const {
        return m_weights.getDevicePointer();
    }
};



template <typename RealType, bool useAliasTable>
class RegularConstantContinuousDistribution1DTemplate;

template <typename RealType>
class RegularConstantContinuousDistribution1DTemplate<RealType, false> {
    cudau::TypedBuffer<RealType> m_weights;
    cudau::TypedBuffer<RealType> m_CDF;
    RealType m_integral;
    uint32_t m_numValues;
    unsigned int m_isInitialized : 1;

public:
    using DeviceType = shared::RegularConstantContinuousDistribution1DTemplate<RealType, false>;

    RegularConstantContinuousDistribution1DTemplate() :
        m_isInitialized(false) {}

    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues);
    void finalize(CUcontext cuContext) {
        if (!m_isInitialized)
            return;
        if (m_CDF.isInitialized() && m_weights.isInitialized()) {
            m_CDF.finalize();
            m_weights.finalize();
        }
    }

    RegularConstantContinuousDistribution1DTemplate &operator=(
        RegularConstantContinuousDistribution1DTemplate &&v) {
        m_weights = std::move(v.m_weights);
        m_CDF = std::move(v.m_CDF);
        m_integral = v.m_integral;
        m_numValues = v.m_numValues;
        return *this;
    }

    RealType getIntegral() const {
        return m_integral;
    }
    uint32_t getNumValues() const {
        return m_numValues;
    }

    bool isInitialized() const {
        return m_isInitialized;
    }

    void getDeviceType(DeviceType* instance) const {
        new (instance) DeviceType(
            m_weights.getDevicePointer(), m_CDF.getDevicePointer(), m_integral, m_numValues);
    }
};

template <typename RealType>
class RegularConstantContinuousDistribution1DTemplate<RealType, true> {
    cudau::TypedBuffer<RealType> m_PDF;
    cudau::TypedBuffer<shared::AliasTableEntry<RealType>> m_aliasTable;
    cudau::TypedBuffer<shared::AliasValueMap<RealType>> m_valueMaps;
    RealType m_integral;
    uint32_t m_numValues;
    unsigned int m_isInitialized : 1;

public:
    using DeviceType = shared::RegularConstantContinuousDistribution1DTemplate<RealType, true>;

    RegularConstantContinuousDistribution1DTemplate() :
        m_isInitialized(false) {}

    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues);
    void finalize(CUcontext cuContext) {
        if (!m_isInitialized)
            return;
        if (m_valueMaps.isInitialized() && m_aliasTable.isInitialized() && m_PDF.isInitialized()) {
            m_valueMaps.finalize();
            m_aliasTable.finalize();
            m_PDF.finalize();
        }
    }

    RegularConstantContinuousDistribution1DTemplate &operator=(
        RegularConstantContinuousDistribution1DTemplate &&v) {
        m_PDF = std::move(v.m_PDF);
        m_aliasTable = std::move(v.m_aliasTable);
        m_valueMaps = std::move(v.m_valueMaps);
        m_integral = v.m_integral;
        m_numValues = v.m_numValues;
        return *this;
    }

    RealType getIntegral() const {
        return m_integral;
    }
    uint32_t getNumValues() const {
        return m_numValues;
    }

    bool isInitialized() const {
        return m_isInitialized;
    }

    void getDeviceType(DeviceType* instance) const {
        new (instance) DeviceType(
            m_PDF.getDevicePointer(), m_aliasTable.getDevicePointer(), m_valueMaps.getDevicePointer(),
            m_integral, m_numValues);
    }
};



template <typename RealType, bool useAliasTable>
class RegularConstantContinuousDistribution2DTemplate {
    using _1DType = RegularConstantContinuousDistribution1DTemplate<RealType, useAliasTable>;
    using _1DDeviceType = shared::RegularConstantContinuousDistribution1DTemplate<RealType, useAliasTable>;

    cudau::TypedBuffer<_1DDeviceType> m_raw1DDists;
    _1DType* m_1DDists;
    _1DType m_top1DDist;
    unsigned int m_isInitialized : 1;

public:
    using DeviceType = shared::RegularConstantContinuousDistribution2DTemplate<RealType, useAliasTable>;

    RegularConstantContinuousDistribution2DTemplate() :
        m_1DDists(nullptr), m_isInitialized(false) {}

    RegularConstantContinuousDistribution2DTemplate &operator=(
        RegularConstantContinuousDistribution2DTemplate &&v) {
        m_raw1DDists = std::move(v.m_raw1DDists);
        m_1DDists = std::move(v.m_1DDists);
        m_top1DDist = std::move(v.m_top1DDist);
        return *this;
    }

    void initialize(
        CUcontext cuContext, cudau::BufferType type,
        const RealType* values, size_t numD1, size_t numD2) {
        Assert(!m_isInitialized, "Already initialized!");
        m_1DDists = new _1DType[numD2];
        m_raw1DDists.initialize(cuContext, type, static_cast<uint32_t>(numD2));

        if (values) {
            _1DDeviceType* rawDists = m_raw1DDists.map();

            // JP: まず各行に関するDistribution1Dを作成する。
            // EN: First, create Distribution1D's for every rows.
            CompensatedSum<RealType> sum(0);
            RealType* integrals = new RealType[numD2];
            for (uint32_t i = 0; i < numD2; ++i) {
                _1DType &dist = m_1DDists[i];
                dist.initialize(cuContext, type, values + i * numD1, numD1);
                dist.getDeviceType(&rawDists[i]);
                integrals[i] = dist.getIntegral();
                sum += integrals[i];
            }

            // JP: 各行の積分値を用いてDistribution1Dを作成する。
            // EN: create a Distribution1D using integral values of each row.
            m_top1DDist.initialize(cuContext, type, integrals, numD2);
            delete[] integrals;

            Assert(std::isfinite(m_top1DDist.getIntegral()), "invalid integral value.");

            m_raw1DDists.unmap();

            m_isInitialized = true;
        }
    }
    void finalize(CUcontext cuContext) {
        if (!m_isInitialized)
            return;

        m_top1DDist.finalize(cuContext);

        for (int i = m_top1DDist.getNumValues() - 1; i >= 0; --i) {
            m_1DDists[i].finalize(cuContext);
        }

        m_raw1DDists.finalize();
        delete[] m_1DDists;
        m_1DDists = nullptr;
    }

    bool isInitialized() const {
        return m_isInitialized;
    }

    void getDeviceType(DeviceType* instance) const {
        _1DDeviceType top1DDist;
        m_top1DDist.getDeviceType(&top1DDist);
        new (instance) DeviceType(m_raw1DDists.getDevicePointer(), top1DDist);
    }
};



using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float, false>;
using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float, false>;
using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float, false>;
using LightDistribution = DiscreteDistribution1D;



template <uint32_t log2BlockWidth>
void saveImage(
    const std::filesystem::path &filepath,
    optixu::HostBlockBuffer2D<float4, log2BlockWidth> &buffer,
    const SDRImageSaverConfig &config) {
    uint32_t width = buffer.getWidth();
    uint32_t height = buffer.getHeight();
    auto data = new float4[width * height];
    buffer.map();
    for (int y = 0; y < static_cast<int32_t>(height); ++y) {
        for (int x = 0; x < static_cast<int32_t>(width); ++x) {
            data[y * width + x] = buffer(x, y);
        }
    }
    buffer.unmap();
    saveImage(filepath, width, height, 4, reinterpret_cast<float*>(data), config);
    delete[] data;
}

void saveImage(
    const std::filesystem::path &filepath,
    uint32_t width, cudau::TypedBuffer<float4> &buffer,
    const SDRImageSaverConfig &config);

void saveImage(
    const std::filesystem::path &filepath,
    cudau::Array &array,
    const SDRImageSaverConfig &config);

}
