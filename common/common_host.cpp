﻿#include "common_host.h"
#include "stopwatch.h"

#include "../ext/stb/stb_image_write.h"
#include "../ext/fpng/src/fpng.h"
#include "tinyexr.h"

#include <deque>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace rtc10 {

void devPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[4096];
    vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
    va_end(args);
    OutputDebugString(str);
}



std::filesystem::path getExecutableDirectory() {
    static std::filesystem::path ret;

    static bool done = false;
    if (!done) {
#if defined(RTC10_Platform_Windows_MSVC)
        TCHAR filepath[1024];
        auto length = GetModuleFileName(NULL, filepath, 1024);
        Assert(length > 0, "Failed to query the executable path.");

        ret = filepath;
#else
        static_assert(false, "Not implemented");
#endif
        ret = ret.remove_filename();

        done = true;
}

    return ret;
}



std::string readTxtFile(const std::filesystem::path& filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string(sstream.str());
}



void SlotFinder::initialize(uint32_t numSlots) {
    m_numLayers = 1;
    m_numLowestFlagBins = nextMultiplierForPowOf2(numSlots, 5);

    // e.g. factor 4
    // 0 | 1101 | 0011 | 1001 | 1011 | 0010 | 1010 | 0000 | 1011 | 1110 | 0101 | 111* | **** | **** | **** | **** | **** | 43 flags
    // OR bins:
    // 1 | 1      1      1      1    | 1      1      0      1    | 1      1      1      *    | *      *      *      *    | 11
    // 2 | 1                           1                           1                           *                         | 3
    // AND bins
    // 1 | 0      0      0      0    | 0      0      0      0    | 0      0      1      *    | *      *      *      *    | 11
    // 2 | 0                           0                           0                           *                         | 3
    //
    // numSlots: 43
    // numLowestFlagBins: 11
    // numLayers: 3
    //
    // Memory Order
    // LowestFlagBins (layer 0) | OR, AND Bins (layer 1) | ... | OR, AND Bins (layer n-1)
    // Offset Pair to OR, AND (layer 0) | ... | Offset Pair to OR, AND (layer n-1)
    // NumUsedFlags (layer 0) | ... | NumUsedFlags (layer n-1)
    // Offset to NumUsedFlags (layer 0) | ... | Offset to NumUsedFlags (layer n-1)
    // NumFlags (layer 0) | ... | NumFlags (layer n-1)

    uint32_t numFlagBinsInLayer = m_numLowestFlagBins;
    m_numTotalCompiledFlagBins = 0;
    while (numFlagBinsInLayer > 1) {
        ++m_numLayers;
        numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 5);
        m_numTotalCompiledFlagBins += 2 * numFlagBinsInLayer; // OR bins and AND bins
    }

    size_t memSize = sizeof(uint32_t) *
        ((m_numLowestFlagBins + m_numTotalCompiledFlagBins) +
         m_numLayers * 2 +
         (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2) +
         m_numLayers +
         m_numLayers);
    void* mem = malloc(memSize);

    uintptr_t memHead = (uintptr_t)mem;
    m_flagBins = (uint32_t*)memHead;
    memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins);

    m_offsetsToOR_AND = (uint32_t*)memHead;
    memHead += sizeof(uint32_t) * m_numLayers * 2;

    m_numUsedFlagsUnderBinList = (uint32_t*)memHead;
    memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2);

    m_offsetsToNumUsedFlags = (uint32_t*)memHead;
    memHead += sizeof(uint32_t) * m_numLayers;

    m_numFlagsInLayerList = (uint32_t*)memHead;

    uint32_t layer = 0;
    uint32_t offsetToOR_AND = 0;
    uint32_t offsetToNumUsedFlags = 0;
    {
        m_numFlagsInLayerList[layer] = numSlots;

        numFlagBinsInLayer = nextMultiplierForPowOf2(numSlots, 5);

        m_offsetsToOR_AND[2 * layer + 0] = offsetToOR_AND;
        m_offsetsToOR_AND[2 * layer + 1] = offsetToOR_AND;
        m_offsetsToNumUsedFlags[layer] = offsetToNumUsedFlags;

        offsetToOR_AND += numFlagBinsInLayer;
        offsetToNumUsedFlags += numFlagBinsInLayer;
    }
    while (numFlagBinsInLayer > 1) {
        ++layer;
        m_numFlagsInLayerList[layer] = numFlagBinsInLayer;

        numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 5);

        m_offsetsToOR_AND[2 * layer + 0] = offsetToOR_AND;
        m_offsetsToOR_AND[2 * layer + 1] = offsetToOR_AND + numFlagBinsInLayer;
        m_offsetsToNumUsedFlags[layer] = offsetToNumUsedFlags;

        offsetToOR_AND += 2 * numFlagBinsInLayer;
        offsetToNumUsedFlags += numFlagBinsInLayer;
    }

    std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
    std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
}

void SlotFinder::finalize() {
    if (m_flagBins)
        free(m_flagBins);
    m_flagBins = nullptr;
}

void SlotFinder::aggregate() {
    uint32_t offsetToOR_last = m_offsetsToOR_AND[2 * 0 + 0];
    uint32_t offsetToAND_last = m_offsetsToOR_AND[2 * 0 + 1];
    uint32_t offsetToNumUsedFlags_last = m_offsetsToNumUsedFlags[0];
    for (int layer = 1; layer < static_cast<int32_t>(m_numLayers); ++layer) {
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        uint32_t offsetToOR = m_offsetsToOR_AND[2 * layer + 0];
        uint32_t offsetToAND = m_offsetsToOR_AND[2 * layer + 1];
        uint32_t offsetToNumUsedFlags = m_offsetsToNumUsedFlags[layer];
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t &ORFlagBin = m_flagBins[offsetToOR + binIdx];
            uint32_t &ANDFlagBin = m_flagBins[offsetToAND + binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[offsetToNumUsedFlags + binIdx];

            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            for (int bit = 0; bit < static_cast<int32_t>(numFlagsInBin); ++bit) {
                uint32_t lBinIdx = 32 * binIdx + bit;
                uint32_t lORFlagBin = m_flagBins[offsetToOR_last + lBinIdx];
                uint32_t lANDFlagBin = m_flagBins[offsetToAND_last + lBinIdx];
                uint32_t lNumFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer - 1] - 32 * lBinIdx);
                if (lORFlagBin != 0)
                    ORFlagBin |= 1 << bit;
                if (popcnt(lANDFlagBin) == lNumFlagsInBin)
                    ANDFlagBin |= 1 << bit;
                numUsedFlagsUnderBin += m_numUsedFlagsUnderBinList[offsetToNumUsedFlags_last + lBinIdx];
            }
        }

        offsetToOR_last = offsetToOR;
        offsetToAND_last = offsetToAND;
        offsetToNumUsedFlags_last = offsetToNumUsedFlags;
    }
}

void SlotFinder::resize(uint32_t numSlots) {
    if (numSlots == m_numFlagsInLayerList[0])
        return;

    SlotFinder newFinder;
    newFinder.initialize(numSlots);

    uint32_t numLowestFlagBins = std::min(m_numLowestFlagBins, newFinder.m_numLowestFlagBins);
    for (int binIdx = 0; binIdx < static_cast<int32_t>(numLowestFlagBins); ++binIdx) {
        uint32_t numFlagsInBin = std::min(32u, numSlots - 32 * binIdx);
        uint32_t mask = numFlagsInBin >= 32 ? 0xFFFFFFFF : ((1 << numFlagsInBin) - 1);
        uint32_t value = m_flagBins[0 + binIdx] & mask;
        newFinder.m_flagBins[0 + binIdx] = value;
        newFinder.m_numUsedFlagsUnderBinList[0 + binIdx] = popcnt(value);
    }

    newFinder.aggregate();

    *this = std::move(newFinder);
}

void SlotFinder::setInUse(uint32_t slotIdx) {
    if (getUsage(slotIdx))
        return;

    bool setANDFlag = false;
    uint32_t flagIdxInLayer = slotIdx;
    for (int layer = 0; layer < static_cast<int32_t>(m_numLayers); ++layer) {
        uint32_t binIdx = flagIdxInLayer / 32;
        uint32_t flagIdxInBin = flagIdxInLayer % 32;

        // JP: 最下層ではOR/ANDは同じ実体だがsetANDFlagが初期値falseであるので設定は1回きり。
        uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
        uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
        uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
        ORFlagBin |= (1 << flagIdxInBin);
        if (setANDFlag)
            ANDFlagBin |= (1 << flagIdxInBin);
        ++numUsedFlagsUnderBin;

        // JP: このビンに利用可能なスロットが無くなった場合は次のANDレイヤーもフラグを立てる。
        uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
        setANDFlag = popcnt(ANDFlagBin) == numFlagsInBin;

        flagIdxInLayer = binIdx;
    }
}

void SlotFinder::setNotInUse(uint32_t slotIdx) {
    if (!getUsage(slotIdx))
        return;

    bool resetORFlag = false;
    uint32_t flagIdxInLayer = slotIdx;
    for (int layer = 0; layer < static_cast<int32_t>(m_numLayers); ++layer) {
        uint32_t binIdx = flagIdxInLayer / 32;
        uint32_t flagIdxInBin = flagIdxInLayer % 32;

        // JP: 最下層ではOR/ANDは同じ実体だがresetORFlagが初期値falseであるので設定は1回きり。
        uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
        uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
        uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
        if (resetORFlag)
            ORFlagBin &= ~(1 << flagIdxInBin);
        ANDFlagBin &= ~(1 << flagIdxInBin);
        --numUsedFlagsUnderBin;

        // JP: このビンに使用中スロットが無くなった場合は次のORレイヤーのフラグを下げる。
        uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
        resetORFlag = ORFlagBin == 0;

        flagIdxInLayer = binIdx;
    }
}

uint32_t SlotFinder::getFirstAvailableSlot() const {
    uint32_t binIdx = 0;
    for (int layer = m_numLayers - 1; layer >= 0; --layer) {
        uint32_t ANDFlagBinOffset = m_offsetsToOR_AND[2 * layer + 1];
        uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        uint32_t ANDFlagBin = m_flagBins[ANDFlagBinOffset + binIdx];

        if (popcnt(ANDFlagBin) != numFlagsInBin) {
            // JP: このビンに利用可能なスロットを発見。
            binIdx = tzcnt(~ANDFlagBin) + 32 * binIdx;
        }
        else {
            // JP: 利用可能なスロットが見つからなかった。
            return 0xFFFFFFFF;
        }
    }

    Assert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
    return binIdx;
}

uint32_t SlotFinder::getFirstUsedSlot() const {
    uint32_t binIdx = 0;
    for (int layer = m_numLayers - 1; layer >= 0; --layer) {
        uint32_t ORFlagBinOffset = m_offsetsToOR_AND[2 * layer + 0];
        uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        uint32_t ORFlagBin = m_flagBins[ORFlagBinOffset + binIdx];

        if (ORFlagBin != 0) {
            // JP: このビンに使用中のスロットを発見。
            binIdx = tzcnt(ORFlagBin) + 32 * binIdx;
        }
        else {
            // JP: 使用中スロットが見つからなかった。
            return 0xFFFFFFFF;
        }
    }

    Assert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
    return binIdx;
}

uint32_t SlotFinder::find_nthUsedSlot(uint32_t n) const {
    if (n >= getNumUsed())
        return 0xFFFFFFFF;

    uint32_t startBinIdx = 0;
    uint32_t accNumUsed = 0;
    for (int layer = m_numLayers - 1; layer >= 0; --layer) {
        uint32_t numUsedFlagsOffset = m_offsetsToNumUsedFlags[layer];
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        for (int binIdx = startBinIdx; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[numUsedFlagsOffset + binIdx];

            // JP: 現在のビンの配下にインデックスnの使用中スロットがある。
            if (accNumUsed + numUsedFlagsUnderBin > n) {
                startBinIdx = 32 * binIdx;
                if (layer == 0) {
                    uint32_t flagBin = m_flagBins[binIdx];
                    startBinIdx += nthSetBit(flagBin, n - accNumUsed);
                }
                break;
            }

            accNumUsed += numUsedFlagsUnderBin;
        }
    }

    Assert(startBinIdx < m_numFlagsInLayerList[0], "Invalid value.");
    return startBinIdx;
}

void SlotFinder::debugPrint() const {
    uint32_t numLowestFlagBins = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
    hpprintf("----");
    for (int binIdx = 0; binIdx < static_cast<int32_t>(numLowestFlagBins); ++binIdx) {
        hpprintf("------------------------------------");
    }
    hpprintf("\n");
    for (int layer = m_numLayers - 1; layer > 0; --layer) {
        hpprintf("layer %u (%u):\n", layer, m_numFlagsInLayerList[layer]);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        hpprintf(" OR:");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
            for (int i = 0; i < 32; ++i) {
                if (i % 8 == 0)
                    hpprintf(" ");

                bool valid = binIdx * 32 + i < static_cast<int32_t>(m_numFlagsInLayerList[layer]);
                if (!valid)
                    continue;

                bool b = (ORFlagBin >> i) & 0x1;
                hpprintf("%c", b ? '|' : '_');
            }
        }
        hpprintf("\n");
        hpprintf("AND:");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
            for (int i = 0; i < 32; ++i) {
                if (i % 8 == 0)
                    hpprintf(" ");

                bool valid = binIdx * 32 + i < static_cast<int32_t>(m_numFlagsInLayerList[layer]);
                if (!valid)
                    continue;

                bool b = (ANDFlagBin >> i) & 0x1;
                hpprintf("%c", b ? '|' : '_');
            }
        }
        hpprintf("\n");
        hpprintf("    ");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
            hpprintf("                            %8u", numUsedFlagsUnderBin);
        }
        hpprintf("\n");
    }
    {
        hpprintf("layer 0 (%u):\n", m_numFlagsInLayerList[0]);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
        hpprintf("   :");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t ORFlagBin = m_flagBins[binIdx];
            for (int i = 0; i < 32; ++i) {
                if (i % 8 == 0)
                    hpprintf(" ");

                bool valid = binIdx * 32 + i < static_cast<int32_t>(m_numFlagsInLayerList[0]);
                if (!valid)
                    continue;

                bool b = (ORFlagBin >> i) & 0x1;
                hpprintf("%c", b ? '|' : '_');
            }
        }
        hpprintf("\n");
        hpprintf("    ");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[binIdx];
            hpprintf("                            %8u", numUsedFlagsUnderBin);
        }
        hpprintf("\n");
    }
}



static bool s_fpng_initialized = false;

void saveImage(
    const std::filesystem::path &filepath, uint32_t width, uint32_t height, const uint32_t* data) {
    if (filepath.extension() == ".png") {
        //stbi_write_png(filepath.string().c_str(), width, height, 4, data,
        //               width * sizeof(uint32_t));
        if (!s_fpng_initialized) {
            fpng::fpng_init();
            s_fpng_initialized = true;
        }
        bool success = fpng::fpng_encode_image_to_file(
            filepath.string().c_str(), data, width, height, 4, 0);
        Assert(success, "failed to write the png file: %s", filepath.string().c_str());
    }
    else if (filepath.extension() == ".bmp") {
        stbi_write_bmp(filepath.string().c_str(), width, height, 4, data);
    }
    else {
        Assert_ShouldNotBeCalled();
    }
}

void saveImageHDR(
    const std::filesystem::path &filepath, uint32_t width, uint32_t height, uint32_t numChs,
    float brightnessScale,
    const float* data, bool flipY) {
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 4;

    std::vector<float> images[4];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);
    images[3].resize(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint32_t srcIdx = y * width + x;
            uint32_t dstIdx = (flipY ? (height - 1 - y) : y) * width + x;
            images[0][dstIdx] = brightnessScale * data[numChs * srcIdx + 0];
            images[1][dstIdx] = 0.0f;
            images[2][dstIdx] = 0.0f;
            images[3][dstIdx] = 0.0f;
            if (numChs >= 2) {
                images[1][dstIdx] = brightnessScale * data[numChs * srcIdx + 1];
                if (numChs >= 3) {
                    images[2][dstIdx] = brightnessScale * data[numChs * srcIdx + 2];
                    if (numChs == 4)
                        images[3][dstIdx] = brightnessScale * data[numChs * srcIdx + 3];
                }
            }
        }
    }

    float* image_ptr[4];
    image_ptr[0] = &(images[3].at(0)); // A
    image_ptr[1] = &(images[2].at(0)); // B
    image_ptr[2] = &(images[1].at(0)); // G
    image_ptr[3] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 4;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "A", 255); header.channels[0].name[strlen("A")] = '\0';
    strncpy(header.channels[1].name, "B", 255); header.channels[1].name[strlen("B")] = '\0';
    strncpy(header.channels[2].name, "G", 255); header.channels[2].name[strlen("G")] = '\0';
    strncpy(header.channels[3].name, "R", 255); header.channels[3].name[strlen("R")] = '\0';

    header.pixel_types = (int32_t*)malloc(sizeof(int32_t) * header.num_channels);
    header.requested_pixel_types = (int32_t*)malloc(sizeof(int32_t) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err = nullptr;
    int32_t ret = SaveEXRImageToFile(&image, &header, filepath.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Save EXR err: %s\n", err);
        FreeEXRErrorMessage(err);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

void saveImage(
    const std::filesystem::path &filepath,
    uint32_t width, uint32_t height, uint32_t numChs, const float* data,
    const SDRImageSaverConfig &config) {
    auto image = new uint32_t[width * height];
    for (int y = 0; y < static_cast<int32_t>(height); ++y) {
        uint32_t sy = config.flipY ? (height - 1 - y) : y;
        for (int x = 0; x < static_cast<int32_t>(width); ++x) {
            uint32_t idx = sy * width + x;
            RGBSpectrum value = RGBSpectrum::Zero();
            float alpha = 1.0f;

            value.r = data[numChs * idx + 0];
            if (numChs >= 2) {
                value.g = data[numChs * idx + 1];
                if (numChs >= 3) {
                    value.b = data[numChs * idx + 2];
                    if (numChs == 4)
                        alpha = data[numChs * idx + 3];
                }
            }
            if (config.applyToneMap) {
                if (!value.allFinite())
                    value = RGBSpectrum::Zero();
                if (value.hasNegative()) {
                    //hpprintf("%4u, %4u: negative value " RGBFMT "\n",
                    //         x, y, rgbprint(value));
                    value = max(value, RGBSpectrum::Zero());
                }
                float lum = value.luminance();
                float lumT = simpleToneMap_s(config.brightnessScale * lum);
                float s = lum > 0.0f ? lumT / lum : 0.0f;
                value *= s;
            }
            if (config.apply_sRGB_gammaCorrection) {
                if (value.hasNegative()) {
                    //hpprintf("%4u, %4u: negative value " RGBFMT "\n",
                    //         x, y, rgbprint(value));
                    value = max(value, RGBSpectrum::Zero());
                }
                value = sRGB_gamma(value);
            }

            if (config.alphaForOverride >= 0.0f)
                alpha = config.alphaForOverride;

            uint32_t &dst = image[y * width + x];
            dst = ((std::min<uint32_t>(static_cast<uint32_t>(value.r * 255), 255) << 0) |
                   (std::min<uint32_t>(static_cast<uint32_t>(value.g * 255), 255) << 8) |
                   (std::min<uint32_t>(static_cast<uint32_t>(value.b * 255), 255) << 16) |
                   (std::min<uint32_t>(static_cast<uint32_t>(alpha * 255), 255) << 24));
        }
    }

    saveImage(filepath, width, height, image);

    delete[] image;
}

void saveImage(
    const std::filesystem::path &filepath, uint32_t width, uint32_t height, const float4* data,
    const SDRImageSaverConfig &config) {
    saveImage(filepath, width, height, 4, reinterpret_cast<const float*>(data), config);
}



struct ImageSaverItem {
    std::filesystem::path filePath;
    uint32_t width;
    uint32_t height;
    std::shared_ptr<float4[]> data;
    SDRImageSaverConfig config;
};

class ImageSaver {
    std::deque<ImageSaverItem> m_imageSaverItems;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    std::atomic_bool m_finishable;
    std::thread m_thread;

public:
    void init() {
        if (!s_fpng_initialized) {
            fpng::fpng_init();
            s_fpng_initialized = true;
        }
        m_finishable = false;
        m_thread = std::thread(std::bind(&ImageSaver::run, this));
    }

    void run() {
        StopWatchHiRes cpuTimer;
        while (true) {
            ImageSaverItem item = {};
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                while (!m_finishable && m_imageSaverItems.empty())
                    m_cond.wait(lock);

                if (m_finishable && m_imageSaverItems.empty())
                    return;

                item = m_imageSaverItems.front();
                m_imageSaverItems.pop_front();
            }

            cpuTimer.start();
            saveImage(item.filePath, item.width, item.height, 4,
                      reinterpret_cast<const float*>(item.data.get()), item.config);
            uint64_t saveTime = cpuTimer.getElapsed(StopWatchDurationType::Milliseconds);
            cpuTimer.reset();
            hpprintf("Save %s: %.3f [s] (Background)\n", item.filePath.string().c_str(), saveTime * 1e-3f);
        }
    }

    void enqueue(const ImageSaverItem &item) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_imageSaverItems.push_back(item);
        }
        m_cond.notify_all();
    }

    void finish() {
        m_finishable = true;
        m_cond.notify_all();
        if (m_thread.joinable())
            m_thread.join();
    }
};

ImageSaver g_imageSaverThread;

void initImageSaverThread() {
    g_imageSaverThread.init();
}

void enqueueSaveImage(
    const std::filesystem::path &filePath,
    cudau::Array &array,
    const SDRImageSaverConfig &config) {
    ImageSaverItem item = {};
    item.filePath = filePath;
    item.width = array.getWidth();
    item.height = array.getHeight();
    item.data = std::make_shared<float4[]>(item.width * item.height);
    item.config = config;

    auto data = array.map<float4>();
    std::copy_n(data, item.width * item.height, item.data.get());
    array.unmap();

    g_imageSaverThread.enqueue(item);
}

void finishImageSaverThread() {
    g_imageSaverThread.finish();
}

}