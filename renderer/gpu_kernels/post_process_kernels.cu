#define PURE_CUDA
#include "../renderer_shared.h"

using namespace rtc9;
using namespace rtc9::shared;
using namespace rtc9::device;



CUDA_DEVICE_FUNCTION CUDA_INLINE RGBSpectrum applySimpleToneMap(
    const RGBSpectrum &input, float brightness) {
    RGBSpectrum color = input;
    float lum = input.luminance();
    if (lum > 0.0f) {
        float lumT = 1 - std::exp(-brightness * lum);
        color *= lumT / lum;
    }
    return color;
}

CUDA_DEVICE_KERNEL void applyToneMap() {
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= plp.s->imageSize.x ||
        launchIndex.y >= plp.s->imageSize.y)
        return;

    plp.s->accumBuffer[launchIndex].add(
        plp.s->ltTargetBuffer[launchIndex] * plp.s->samleSizeCorrectionFactor);
    const DiscretizedSpectrum &accumResult = plp.s->accumBuffer[launchIndex].getValue().result;
    float colorXYZ[3];
    accumResult.toXYZ(colorXYZ);
    float colorRGB[3];
    transformTristimulus(mat_XYZ_to_Rec709_D65, colorXYZ, colorRGB);
    float recNumAccums = 1.0f / (plp.f->numAccumFrames + 1);
    colorRGB[0] *= recNumAccums;
    colorRGB[1] *= recNumAccums;
    colorRGB[2] *= recNumAccums;
    //constexpr float gamma = 1.0f / 0.6f;
    //accumResult.r = std::pow(accumResult.r, gamma);
    //accumResult.g = std::pow(accumResult.g, gamma);
    //accumResult.b = std::pow(accumResult.b, gamma);
    RGBSpectrum output = applySimpleToneMap(RGBSpectrum(colorRGB), /*plp.f->brighness*/1.0f);

    //constexpr float gamma = 1.0f / 0.6f;
    //output.r = std::pow(output.r, gamma);
    //output.g = std::pow(output.g, gamma);
    //output.b = std::pow(output.b, gamma);

    plp.f->outputBuffer.write(launchIndex, output);
}

CUDA_DEVICE_KERNEL void clearLtTargetBuffer() {
    const uint2 launchIndex = make_uint2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= plp.s->imageSize.x ||
        launchIndex.y >= plp.s->imageSize.y)
        return;

    if (plp.f->numAccumFrames == 0)
        plp.s->accumBuffer[launchIndex].reset();
    plp.s->ltTargetBuffer[launchIndex] = DiscretizedSpectrum::Zero();
}
