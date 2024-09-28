#define PURE_CUDA
#include "../renderer_shared.h"

using namespace rtc9;

CUDA_DEVICE_KERNEL void dummyKernel(
	RGBSpectrum rgb,
	WavelengthSamplesTemplate<float, NumSpectralSamples> wls,
	SpectrumStorageTemplate<float, NumStrataForStorage>* dst) {
	UpsampledSpectrum us(SpectrumType::Reflectance, ColorSpace::Rec709_D65, rgb.r, rgb.g, rgb.b);
	dst->add(wls, us.evaluate(wls));
}
