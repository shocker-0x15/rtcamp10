#pragma once

#include "basic_types.h"
#include "rgb_spectrum_types.h"

#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <variant>
#include <chrono>
#include <random>
#include <functional>
#include <filesystem>

#if 1
#   define hpprintf(fmt, ...) do { rtc10::devPrintf(fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__); } while (0)
#else
#   define hpprintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

namespace rtc10 {

std::filesystem::path getExecutableDirectory();

std::string readTxtFile(const std::filesystem::path& filepath);

template <typename... Types>
void _throwRuntimeError(const char* fmt, const Types &... args) {
    char str[2048];
    snprintf(str, sizeof(str), fmt, args...);
    throw std::runtime_error(str);
}

template <typename... Types>
void throwRuntimeError(bool expr, const char* fmt, const Types &... args) {
    if (!expr)
        _throwRuntimeError(fmt, args...);
}



class SlotFinder {
    uint32_t m_numLayers;
    uint32_t m_numLowestFlagBins;
    uint32_t m_numTotalCompiledFlagBins;
    uint32_t* m_flagBins;
    uint32_t* m_offsetsToOR_AND;
    uint32_t* m_numUsedFlagsUnderBinList;
    uint32_t* m_offsetsToNumUsedFlags;
    uint32_t* m_numFlagsInLayerList;

    SlotFinder(const SlotFinder &) = delete;
    SlotFinder &operator=(const SlotFinder &) = delete;

    void aggregate();

    uint32_t getNumLayers() const {
        return m_numLayers;
    }

    const uint32_t* getOffsetsToOR_AND() const {
        return m_offsetsToOR_AND;
    }

    const uint32_t* getOffsetsToNumUsedFlags() const {
        return m_offsetsToNumUsedFlags;
    }

    const uint32_t* getNumFlagsInLayerList() const {
        return m_numFlagsInLayerList;
    }

public:
    static constexpr uint32_t InvalidSlotIndex = 0xFFFFFFFF;

    SlotFinder() :
        m_numLayers(0), m_numLowestFlagBins(0), m_numTotalCompiledFlagBins(0),
        m_flagBins(nullptr), m_offsetsToOR_AND(nullptr),
        m_numUsedFlagsUnderBinList(nullptr), m_offsetsToNumUsedFlags(nullptr),
        m_numFlagsInLayerList(nullptr) {
    }
    ~SlotFinder() {
    }

    void initialize(uint32_t numSlots);

    void finalize();

    SlotFinder &operator=(SlotFinder &&inst) {
        finalize();

        m_numLayers = inst.m_numLayers;
        m_numLowestFlagBins = inst.m_numLowestFlagBins;
        m_numTotalCompiledFlagBins = inst.m_numTotalCompiledFlagBins;
        m_flagBins = inst.m_flagBins;
        m_offsetsToOR_AND = inst.m_offsetsToOR_AND;
        m_numUsedFlagsUnderBinList = inst.m_numUsedFlagsUnderBinList;
        m_offsetsToNumUsedFlags = inst.m_offsetsToNumUsedFlags;
        m_numFlagsInLayerList = inst.m_numFlagsInLayerList;
        inst.m_flagBins = nullptr;
        inst.m_offsetsToOR_AND = nullptr;
        inst.m_numUsedFlagsUnderBinList = nullptr;
        inst.m_offsetsToNumUsedFlags = nullptr;
        inst.m_numFlagsInLayerList = nullptr;

        return *this;
    }
    SlotFinder(SlotFinder &&inst) {
        *this = std::move(inst);
    }

    void resize(uint32_t numSlots);

    void reset() {
        std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
        std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
    }



    void setInUse(uint32_t slotIdx);

    void setNotInUse(uint32_t slotIdx);

    bool getUsage(uint32_t slotIdx) const {
        uint32_t binIdx = slotIdx / 32;
        uint32_t flagIdxInBin = slotIdx % 32;
        uint32_t flagBin = m_flagBins[binIdx];

        return (bool)((flagBin >> flagIdxInBin) & 0x1);
    }

    uint32_t getFirstAvailableSlot() const;

    uint32_t getFirstUsedSlot() const;

    uint32_t find_nthUsedSlot(uint32_t n) const;

    uint32_t getNumSlots() const {
        return m_numFlagsInLayerList[0];
    }

    uint32_t getNumUsed() const {
        return m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[m_numLayers - 1]];
    }

    void debugPrint() const;
};



struct MovingAverageTime {
    float values[60];
    uint32_t index;
    uint32_t numValidValues;
    MovingAverageTime() : index(0), numValidValues(0) {}
    void append(float value) {
        values[index] = value;
        index = (index + 1) % lengthof(values);
        numValidValues = std::min<uint32_t>(numValidValues + 1, static_cast<uint32_t>(lengthof(values)));
    }
    float getAverage() const {
        float sum = 0.0f;
        for (uint32_t i = 0; i < numValidValues; ++i)
            sum += values[(index - 1 - i + lengthof(values)) % lengthof(values)];
        return numValidValues > 0 ? sum / numValidValues : 0.0f;
    }
};



void saveImage(
    const std::filesystem::path &filepath, uint32_t width, uint32_t height, const uint32_t* data);
void saveImageHDR(
    const std::filesystem::path &filepath, uint32_t width, uint32_t height, uint32_t numChs,
    float brightnessScale,
    const float* data, bool flipY = false);

struct SDRImageSaverConfig {
    float alphaForOverride;
    float brightnessScale;
    unsigned int applyToneMap : 1;
    unsigned int apply_sRGB_gammaCorrection : 1;
    unsigned int flipY : 1;

    SDRImageSaverConfig() :
        brightnessScale(1.0f),
        applyToneMap(false), apply_sRGB_gammaCorrection(false),
        flipY(false),
        alphaForOverride(-1) {}
};

void saveImage(
    const std::filesystem::path &filepath,
    uint32_t width, uint32_t height, uint32_t numChs, const float* data,
    const SDRImageSaverConfig &config);



void initImageSaverThread();

void enqueueSaveImage(
    const std::filesystem::path &filepath,
    cudau::Array &array,
    const SDRImageSaverConfig &config);

void finishImageSaverThread();

} // namespace rtc10
