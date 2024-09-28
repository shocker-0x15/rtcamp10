#include "common_renderer_host.h"

namespace rtc10 {

template <typename RealType>
void DiscreteDistribution1DTemplate<RealType, false>::
initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues) {
    Assert(!m_isInitialized, "Already initialized!");
    m_numValues = static_cast<uint32_t>(numValues);
    if (m_numValues == 0) {
        m_integral = 0.0f;
        return;
    }

    m_weights.initialize(cuContext, type, m_numValues);
    m_CDF.initialize(cuContext, type, m_numValues);

    if (values == nullptr) {
        m_integral = 0.0f;
        m_isInitialized = true;
        return;
    }

    RealType* weights = m_weights.map();
    std::memcpy(weights, values, sizeof(RealType) * m_numValues);
    m_weights.unmap();

    RealType* CDF = m_CDF.map();

    CompensatedSum<RealType> sum(0);
    for (uint32_t i = 0; i < m_numValues; ++i) {
        CDF[i] = sum;
        sum += values[i];
    }
    m_integral = sum;

    m_CDF.unmap();

    m_isInitialized = true;
}

template <typename RealType>
void DiscreteDistribution1DTemplate<RealType, true>::
initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues) {
    Assert(!m_isInitialized, "Already initialized!");
    m_numValues = static_cast<uint32_t>(numValues);
    if (m_numValues == 0) {
        m_integral = 0.0f;
        return;
    }

    m_weights.initialize(cuContext, type, m_numValues);
    m_aliasTable.initialize(cuContext, type, m_numValues);
    m_valueMaps.initialize(cuContext, type, m_numValues);

    if (values == nullptr) {
        m_integral = 0.0f;
        m_isInitialized = true;
        return;
    }

    RealType* weights = m_weights.map();
    std::memcpy(weights, values, sizeof(RealType) * m_numValues);
    m_weights.unmap();

    CompensatedSum<RealType> sum(0);
    for (uint32_t i = 0; i < m_numValues; ++i)
        sum += values[i];
    RealType avgWeight = sum / m_numValues;
    m_integral = sum;

    struct IndexAndWeight {
        uint32_t index;
        RealType weight;
        IndexAndWeight() {}
        IndexAndWeight(uint32_t _index, RealType _weight) :
            index(_index), weight(_weight) {}
    };

    std::vector<IndexAndWeight> smallGroup;
    std::vector<IndexAndWeight> largeGroup;
    for (uint32_t i = 0; i < m_numValues; ++i) {
        RealType weight = values[i];
        IndexAndWeight entry(i, weight);
        if (weight <= avgWeight)
            smallGroup.push_back(entry);
        else
            largeGroup.push_back(entry);
    }
    shared::AliasTableEntry<RealType>* aliasTable = m_aliasTable.map();
    shared::AliasValueMap<RealType>* valueMaps = m_valueMaps.map();
    for (int i = 0; !smallGroup.empty() && !largeGroup.empty(); ++i) {
        IndexAndWeight smallPair = smallGroup.back();
        smallGroup.pop_back();
        IndexAndWeight &largePair = largeGroup.back();
        uint32_t secondIndex = largePair.index;
        RealType reducedWeight = (largePair.weight + smallPair.weight) - avgWeight;
        largePair.weight = reducedWeight;
        if (largePair.weight <= avgWeight) {
            smallGroup.push_back(largePair);
            largeGroup.pop_back();
        }
        RealType probToPickFirst = smallPair.weight / avgWeight;
        aliasTable[smallPair.index] = shared::AliasTableEntry<RealType>(secondIndex, probToPickFirst);

        shared::AliasValueMap<RealType> valueMap;
        RealType probToPickSecond = 1 - probToPickFirst;
        valueMap.scaleForFirst = avgWeight / values[smallPair.index];
        valueMap.scaleForSecond = avgWeight / values[secondIndex];
        valueMap.offsetForSecond = (reducedWeight - smallPair.weight) / values[secondIndex];
        valueMaps[smallPair.index] = valueMap;
    }
    while (!smallGroup.empty() || !largeGroup.empty()) {
        IndexAndWeight pair;
        if (!smallGroup.empty()) {
            pair = smallGroup.back();
            smallGroup.pop_back();
        }
        else {
            pair = largeGroup.back();
            largeGroup.pop_back();
        }
        aliasTable[pair.index] = shared::AliasTableEntry<RealType>(0xFFFFFFFF, 1.0f);

        shared::AliasValueMap<RealType> valueMap;
        valueMap.scaleForFirst = avgWeight / values[pair.index];
        valueMap.scaleForSecond = 0;
        valueMap.offsetForSecond = 0;
        valueMaps[pair.index] = valueMap;
    }
    m_valueMaps.unmap();
    m_aliasTable.unmap();

    m_isInitialized = true;
}

template class DiscreteDistribution1DTemplate<float, false>;
template class DiscreteDistribution1DTemplate<float, true>;



template <typename RealType>
void RegularConstantContinuousDistribution1DTemplate<RealType, false>::
initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues) {
    Assert(!m_isInitialized, "Already initialized!");
    m_numValues = static_cast<uint32_t>(numValues);
    if (m_numValues == 0) {
        m_integral = 0.0f;
        return;
    }

    m_weights.initialize(cuContext, type, m_numValues);
    m_CDF.initialize(cuContext, type, m_numValues);

    if (values == nullptr) {
        m_integral = 0.0f;
        m_isInitialized = true;
        return;
    }

    RealType* weights = m_weights.map();
    std::memcpy(weights, values, sizeof(RealType) * m_numValues);
    m_weights.unmap();

    RealType* CDF = m_CDF.map();

    CompensatedSum<RealType> sum(0);
    for (uint32_t i = 0; i < m_numValues; ++i) {
        CDF[i] = sum;
        sum += weights[i];
    }
    m_integral = sum / m_numValues;

    m_CDF.unmap();

    m_isInitialized = true;
}

template <typename RealType>
void RegularConstantContinuousDistribution1DTemplate<RealType, true>::
initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues) {
    Assert(!m_isInitialized, "Already initialized!");
    m_numValues = static_cast<uint32_t>(numValues);
    if (m_numValues == 0) {
        m_integral = 0.0f;
        return;
    }

    m_PDF.initialize(cuContext, type, m_numValues);
    m_aliasTable.initialize(cuContext, type, m_numValues);
    m_valueMaps.initialize(cuContext, type, m_numValues);

    if (values == nullptr) {
        m_integral = 0.0f;
        m_isInitialized = true;
        return;
    }

    RealType* PDF = m_PDF.map();
    std::memcpy(PDF, values, sizeof(RealType) * m_numValues);

    CompensatedSum<RealType> sum(0);
    for (uint32_t i = 0; i < m_numValues; ++i)
        sum += values[i];
    RealType avgWeight = sum / m_numValues;
    m_integral = avgWeight;

    for (uint32_t i = 0; i < m_numValues; ++i)
        PDF[i] /= m_integral;
    m_PDF.unmap();

    struct IndexAndWeight {
        uint32_t index;
        RealType weight;
        IndexAndWeight() {}
        IndexAndWeight(uint32_t _index, RealType _weight) :
            index(_index), weight(_weight) {}
    };

    std::vector<IndexAndWeight> smallGroup;
    std::vector<IndexAndWeight> largeGroup;
    for (uint32_t i = 0; i < m_numValues; ++i) {
        RealType weight = values[i];
        IndexAndWeight entry(i, weight);
        if (weight <= avgWeight)
            smallGroup.push_back(entry);
        else
            largeGroup.push_back(entry);
    }

    shared::AliasTableEntry<RealType>* aliasTable = m_aliasTable.map();
    shared::AliasValueMap<RealType>* valueMaps = m_valueMaps.map();
    for (int i = 0; !smallGroup.empty() && !largeGroup.empty(); ++i) {
        IndexAndWeight smallPair = smallGroup.back();
        smallGroup.pop_back();
        IndexAndWeight &largePair = largeGroup.back();
        uint32_t secondIndex = largePair.index;
        RealType reducedWeight = (largePair.weight + smallPair.weight) - avgWeight;
        largePair.weight = reducedWeight;
        if (largePair.weight <= avgWeight) {
            smallGroup.push_back(largePair);
            largeGroup.pop_back();
        }
        RealType probToPickFirst = smallPair.weight / avgWeight;
        aliasTable[smallPair.index] = shared::AliasTableEntry<RealType>(secondIndex, probToPickFirst);

        shared::AliasValueMap<RealType> valueMap;
        RealType probToPickSecond = 1 - probToPickFirst;
        valueMap.scaleForFirst = avgWeight / values[smallPair.index];
        valueMap.scaleForSecond = avgWeight / values[secondIndex];
        valueMap.offsetForSecond = (reducedWeight - smallPair.weight) / values[secondIndex];
        valueMaps[smallPair.index] = valueMap;
    }
    while (!smallGroup.empty() || !largeGroup.empty()) {
        IndexAndWeight pair;
        if (!smallGroup.empty()) {
            pair = smallGroup.back();
            smallGroup.pop_back();
        }
        else {
            pair = largeGroup.back();
            largeGroup.pop_back();
        }
        aliasTable[pair.index] = shared::AliasTableEntry<RealType>(0xFFFFFFFF, 1.0f);

        shared::AliasValueMap<RealType> valueMap;
        valueMap.scaleForFirst = avgWeight / values[pair.index];
        valueMap.scaleForSecond = 0;
        valueMap.offsetForSecond = 0;
        valueMaps[pair.index] = valueMap;
    }
    m_valueMaps.unmap();
    m_aliasTable.unmap();

    m_isInitialized = true;
}

template class RegularConstantContinuousDistribution1DTemplate<float, false>;
template class RegularConstantContinuousDistribution1DTemplate<float, true>;



template class RegularConstantContinuousDistribution2DTemplate<float, false>;
template class RegularConstantContinuousDistribution2DTemplate<float, true>;



void saveImage(
    const std::filesystem::path &filepath,
    uint32_t width, cudau::TypedBuffer<float4> &buffer,
    const SDRImageSaverConfig &config) {
    Assert(buffer.numElements() % width == 0, "Buffer's length is not divisible by the width.");
    uint32_t height = buffer.numElements() / width;
    auto data = reinterpret_cast<float*>(buffer.map());
    saveImage(filepath, width, height, 4, data, config);
    buffer.unmap();
}

void saveImage(
    const std::filesystem::path &filepath,
    cudau::Array &array,
    const SDRImageSaverConfig &config) {
    auto data = array.map<float>();
    saveImage(filepath, array.getWidth(), array.getHeight(), 4, data, config);
    array.unmap();
}


}
