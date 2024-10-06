#include "renderer_scene.h"
#include "../common/common_host.h"
#include "../common/common_renderer_host.h"

// Include glfw3.h after our OpenGL definitions
#include "../common/utils/gl_util.h"
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "../common/stopwatch.h"



namespace ImGui {
    bool SliderInputFloat(
        const char* label, float* v,
        float v_min, float v_max,
        float step, float step_fast,
        const char* format = "%.3f",
        ImGuiSliderFlags flags = 0) {
        ImGuiIO &io = GetIO();
        ImGuiStyle &style = GetStyle();

        BeginGroup();
        PushID(label);

        const float buttonSize = ImGui::GetFrameHeight();
        const float innerSpacing = style.ItemInnerSpacing.x;

        SetNextItemWidth(CalcItemWidth() - 2 * (buttonSize + innerSpacing));
        bool valueChanged = false;
        valueChanged |= SliderFloat("", v, v_min, v_max, format, flags);

        PushButtonRepeat(true);
        SameLine(0, innerSpacing);
        if (Button("-", ImVec2(buttonSize, buttonSize))) {
            *v -= (io.KeyCtrl && step_fast > 0) ? step_fast : step;
            *v = std::min(std::max(*v, v_min), v_max);
            valueChanged = true;
        }
        SameLine(0, innerSpacing);
        if (Button("+", ImVec2(buttonSize, buttonSize))) {
            *v += (io.KeyCtrl && step_fast > 0) ? step_fast : step;
            *v = std::min(std::max(*v, v_min), v_max);
            valueChanged = true;
        }
        PopButtonRepeat();

        SameLine(0, innerSpacing);
        Text(label);

        PopID();
        EndGroup();

        return valueChanged;
    }
}



namespace rtc10 {

static StopWatchHiRes g_globalTimer;

static cudau::Array rngBuffer;
static optixu::HostBlockBuffer2D<DiscretizedSpectrum, 0> ltTargetBuffer;
static optixu::HostBlockBuffer2D<SpectrumStorage, 0> accumBuffer;
static glu::Texture2D gfxOutputBuffer;
static cudau::Array cudaOutputBuffer;
static bool withGfx = true;

const void initializeScreenRelatedBuffers(uint32_t screenWidth, uint32_t screenHeight) {
    rngBuffer.initialize2D(
        g_gpuEnv.cuContext, cudau::ArrayElementType::UInt32, nextPowerOf2((sizeof(shared::PCG32RNG) + 3) / 4),
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        screenWidth, screenHeight, 1);
    {
        auto rngs = rngBuffer.map<shared::PCG32RNG>();
        std::mt19937_64 rngSeed(591842031321323413);
        for (int y = 0; y < screenHeight; ++y) {
            for (int x = 0; x < screenWidth; ++x) {
                shared::PCG32RNG &rng = rngs[y * screenWidth + x];
                rng.setState(rngSeed());
            }
        }
        rngBuffer.unmap();
    }

    ltTargetBuffer.initialize(
        g_gpuEnv.cuContext, bufferType, screenWidth, screenHeight);
    accumBuffer.initialize(
        g_gpuEnv.cuContext, bufferType,
        screenWidth, screenHeight);
    if (withGfx) {
        gfxOutputBuffer.initialize(GL_RGBA32F, screenWidth, screenHeight, 1);
        cudaOutputBuffer.initializeFromGLTexture2D(
            g_gpuEnv.cuContext, gfxOutputBuffer.getHandle(),
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    }
    else {
        cudaOutputBuffer.initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::UInt32, nextPowerOf2((sizeof(RGBSpectrum) + 3) / 4),
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            screenWidth, screenHeight, 1);
    }
};

const void resizeScreenRelatedBuffers(uint32_t screenWidth, uint32_t screenHeight) {
    uint32_t prevWidth = rngBuffer.getWidth();
    uint32_t prevHeight = rngBuffer.getHeight();

    rngBuffer.resize(screenWidth, screenHeight);
    if (screenWidth > prevWidth || screenHeight > prevHeight)  {
        auto rngs = rngBuffer.map<shared::PCG32RNG>();
        std::mt19937_64 rngSeed(591842031321323413);
        for (int y = 0; y < screenHeight; ++y) {
            for (int x = 0; x < screenWidth; ++x) {
                shared::PCG32RNG &rng = rngs[y * screenWidth + x];
                rng.setState(rngSeed());
            }
        }
        rngBuffer.unmap();
    }

    cudaOutputBuffer.finalize();
    gfxOutputBuffer.finalize();
    ltTargetBuffer.resize(screenWidth, screenHeight);
    accumBuffer.resize(screenWidth, screenHeight);
    if (withGfx) {
        gfxOutputBuffer.initialize(GL_RGBA32F, screenWidth, screenHeight, 1);
        cudaOutputBuffer.initializeFromGLTexture2D(
            g_gpuEnv.cuContext, gfxOutputBuffer.getHandle(),
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    }
    else {
        cudaOutputBuffer.resize(screenWidth, screenHeight);
    }
};

const void finalizeScreenRelatedBuffers() {
    cudaOutputBuffer.finalize();
    gfxOutputBuffer.finalize();
    accumBuffer.finalize();
    ltTargetBuffer.finalize();

    rngBuffer.finalize();
};



static shared::StaticPipelineLaunchParameters staticPlpOnHost;
static shared::PerFramePipelineLaunchParameters perFramePlpOnHost;
static shared::PipelineLaunchParameters plpOnHost;
static CUdeviceptr staticPlpOnDevice;
static CUdeviceptr perFramePlpOnDevice;
static CUdeviceptr plpOnDevice;

static cudau::TypedBuffer<UpsampledSpectrum::spectrum_grid_cell_t> UpsampledSpectrum_spectrum_grid;
static cudau::TypedBuffer<UpsampledSpectrum::spectrum_data_point_t> UpsampledSpectrum_spectrum_data_points;

static cudau::TypedBuffer<shared::PCG32RNG> ltRngBuffer;
static cudau::TypedBuffer<shared::LightPathVertex> lightVertexCache;
static cudau::TypedBuffer<shared::LvcBptPassInfo> lvcBptPassInfo;
static constexpr uint32_t numLightTracingPaths = 1024 * 1024;

static void setUpPipelineLaunchParameters(uint32_t screenWidth, uint32_t screenHeight) {
    {
        staticPlpOnHost.DiscretizedSpectrum_xbar = DiscretizedSpectrumAlwaysSpectral::xbar;
        staticPlpOnHost.DiscretizedSpectrum_ybar = DiscretizedSpectrumAlwaysSpectral::ybar;
        staticPlpOnHost.DiscretizedSpectrum_zbar = DiscretizedSpectrumAlwaysSpectral::zbar;
        staticPlpOnHost.DiscretizedSpectrum_integralCMF = DiscretizedSpectrumAlwaysSpectral::integralCMF;
#if RTC10_SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        constexpr uint32_t NumSpectrumGridCells = 168;
        constexpr uint32_t NumSpectrumDataPoints = 186;
        UpsampledSpectrum_spectrum_grid.initialize(
            g_gpuEnv.cuContext, bufferType,
            UpsampledSpectrum::spectrum_grid, NumSpectrumGridCells);
        UpsampledSpectrum_spectrum_data_points.initialize(
            g_gpuEnv.cuContext, bufferType,
            UpsampledSpectrum::spectrum_data_points, NumSpectrumDataPoints);
        staticPlpOnHost.UpsampledSpectrum_spectrum_grid =
            UpsampledSpectrum_spectrum_grid.getDevicePointer();
        staticPlpOnHost.UpsampledSpectrum_spectrum_data_points =
            UpsampledSpectrum_spectrum_data_points.getDevicePointer();
#elif RTC10_SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
        UpsampledSpectrum_maxBrightnesses.initialize(
            m_cuContext, g_bufferType,
            UpsampledSpectrum::maxBrightnesses, UpsampledSpectrum::kTableResolution);
        UpsampledSpectrum_coefficients_sRGB_D65.initialize(
            m_cuContext, g_bufferType,
            UpsampledSpectrum::coefficients_sRGB_D65, 3 * pow3(UpsampledSpectrum::kTableResolution));
        UpsampledSpectrum_coefficients_sRGB_E.initialize(
            m_cuContext, g_bufferType,
            UpsampledSpectrum::coefficients_sRGB_E, 3 * pow3(UpsampledSpectrum::kTableResolution));
        staticPlpOnHost.UpsampledSpectrum_maxBrightnesses =
            UpsampledSpectrum_maxBrightnesses.getDevicePointer();
        staticPlpOnHost.UpsampledSpectrum_coefficients_sRGB_D65 =
            UpsampledSpectrum_coefficients_sRGB_D65.getDevicePointer();
        staticPlpOnHost.UpsampledSpectrum_coefficients_sRGB_E =
            UpsampledSpectrum_coefficients_sRGB_E.getDevicePointer();
#endif

        staticPlpOnHost.bsdfProcedureSets =
            g_gpuEnv.bsdfProcedureSetBuffer.getROBuffer<shared::enableBufferOobCheck>();
        staticPlpOnHost.surfaceMaterials = g_scene.getSurfaceMaterialsOnDevice();
        staticPlpOnHost.geometryInstances = g_scene.getGeometryInstancesOnDevice();
        staticPlpOnHost.geometryGroups = g_scene.getGeometryGroupsOnDevice();

        staticPlpOnHost.imageSize = int2(screenWidth, screenHeight);
        staticPlpOnHost.rngBuffer = rngBuffer.getSurfaceObject(0);
        staticPlpOnHost.ltTargetBuffer = ltTargetBuffer.getBlockBuffer2D();
        staticPlpOnHost.accumBuffer = accumBuffer.getBlockBuffer2D();

        ltRngBuffer.initialize(g_gpuEnv.cuContext, bufferType, numLightTracingPaths);
        {
            auto rngs = ltRngBuffer.map();
            std::mt19937_64 rngSeed(8125411377);
            for (int i = 0; i < numLightTracingPaths; ++i) {
                shared::PCG32RNG &rng = rngs[i];
                rng.setState(rngSeed());
            }
            ltRngBuffer.unmap();
        }
        staticPlpOnHost.ltRngBuffer = ltRngBuffer.getROBuffer<shared::enableBufferOobCheck>();

        lightVertexCache.initialize(g_gpuEnv.cuContext, bufferType, 10 * numLightTracingPaths);
        staticPlpOnHost.lightVertexCache = lightVertexCache.getRWBuffer<shared::enableBufferOobCheck>();
        lvcBptPassInfo.initialize(g_gpuEnv.cuContext, bufferType, 1);
        staticPlpOnHost.lvcBptPassInfo = lvcBptPassInfo.getDevicePointer();
        staticPlpOnHost.numLightPaths = numLightTracingPaths;
    }
    CUDADRV_CHECK(cuMemAlloc(&staticPlpOnDevice, sizeof(staticPlpOnHost)));
    CUDADRV_CHECK(cuMemcpyHtoD(staticPlpOnDevice, &staticPlpOnHost, sizeof(staticPlpOnHost)));



    {

    }
    CUDADRV_CHECK(cuMemAlloc(&perFramePlpOnDevice, sizeof(perFramePlpOnHost)));



    plpOnHost.s = reinterpret_cast<shared::StaticPipelineLaunchParameters*>(staticPlpOnDevice);
    plpOnHost.f = reinterpret_cast<shared::PerFramePipelineLaunchParameters*>(perFramePlpOnDevice);
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plpOnHost)));
    CUDADRV_CHECK(cuMemcpyHtoD(plpOnDevice, &plpOnHost, sizeof(plpOnHost)));
    CUDADRV_CHECK(cuMemcpyHtoD(g_gpuEnv.plpForPostProcessKernelsModule, &plpOnHost, sizeof(plpOnHost)));
    CUDADRV_CHECK(cuMemcpyHtoD(g_gpuEnv.computeLightProbs.debugPlp, &plpOnHost, sizeof(plpOnHost)));
}



template <uint32_t numBuffers>
class StreamChain {
    std::array<CUstream, numBuffers> m_streams;
    std::array<CUevent, numBuffers> m_endEvents;
    uint32_t m_curBufIdx;

public:
    StreamChain() {
        for (int i = 0; i < numBuffers; ++i) {
            m_streams[i] = nullptr;
            m_endEvents[i] = nullptr;
        }
    }

    void initialize(CUcontext cuContext) {
        for (int i = 0; i < numBuffers; ++i) {
            CUDADRV_CHECK(cuStreamCreate(&m_streams[i], 0));
            CUDADRV_CHECK(cuEventCreate(&m_endEvents[i], 0));
        }
        m_curBufIdx = 0;
    }

    void finalize() {
        for (int i = 1; i >= 0; --i) {
            CUDADRV_CHECK(cuStreamSynchronize(m_streams[i]));
            CUDADRV_CHECK(cuEventDestroy(m_endEvents[i]));
            CUDADRV_CHECK(cuStreamDestroy(m_streams[i]));
            m_endEvents[i] = nullptr;
            m_streams[i] = nullptr;
        }
    }

    void swap() {
        CUstream curStream = m_streams[m_curBufIdx];
        CUevent curEvent = m_endEvents[m_curBufIdx];
        CUDADRV_CHECK(cuEventRecord(curEvent, curStream));
        m_curBufIdx = (m_curBufIdx + 1) % numBuffers;
    }

    CUstream waitAvailableAndGetCurrentStream() const {
        CUstream curStream = m_streams[m_curBufIdx];
        CUevent prevStreamEndEvent = m_endEvents[(m_curBufIdx + numBuffers - 1) % numBuffers];
        CUDADRV_CHECK(cuStreamSynchronize(curStream));
        CUDADRV_CHECK(cuStreamWaitEvent(curStream, prevStreamEndEvent, 0));
        return curStream;
    }

    void waitAllWorkDone() const {
        for (int i = 0; i < numBuffers; ++i)
            CUDADRV_CHECK(cuStreamSynchronize(m_streams[i]));
    }
};



struct KeyState {
    uint64_t timesLastChanged[5];
    bool statesLastChanged[5];
    uint32_t lastIndex;

    KeyState() : lastIndex(0) {
        for (int i = 0; i < 5; ++i) {
            timesLastChanged[i] = 0;
            statesLastChanged[i] = false;
        }
    }

    void recordStateChange(bool state, uint64_t time) {
        bool lastState = statesLastChanged[lastIndex];
        if (state == lastState)
            return;

        lastIndex = (lastIndex + 1) % 5;
        statesLastChanged[lastIndex] = !lastState;
        timesLastChanged[lastIndex] = time;
    }

    bool getState(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return statesLastChanged[(lastIndex + goBack + 5) % 5];
    }

    uint64_t getTime(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return timesLastChanged[(lastIndex + goBack + 5) % 5];
    }
};

static KeyState g_keyForward;
static KeyState g_keyBackward;
static KeyState g_keyLeftward;
static KeyState g_keyRightward;
static KeyState g_keyUpward;
static KeyState g_keyDownward;
static KeyState g_keyTiltLeft;
static KeyState g_keyTiltRight;
static KeyState g_keyFasterPosMovSpeed;
static KeyState g_keySlowerPosMovSpeed;
static KeyState g_keyDebugPrint;
static KeyState g_buttonRotate;
static double g_mouseX;
static double g_mouseY;

static float g_cameraPositionalMovingSpeed;
static float g_cameraDirectionalMovingSpeed;
static float g_cameraTiltSpeed;
static Quaternion g_cameraOrientation;
static Quaternion g_tempCameraOrientation;
static Point3D g_cameraPosition;

static bool g_guiMode = true;
static std::filesystem::path g_sceneFilePath;
static float g_playbackSpeed = 1.0f;
static uint32_t g_spp = 8;
static uint32_t g_beginTimeStepIndex = 0;
static uint32_t g_endTimeStepIndex = UINT32_MAX;



static float computeHaltonSequence(uint32_t base, uint32_t idx) {
    const float recBase = 1.0f / base;
    float ret = 0.0f;
    float scale = 1.0f;
    while (idx) {
        scale *= recBase;
        ret += (idx % base) * scale;
        idx /= base;
    }
    return ret;
}

static float wavelengthRandoms[1024];
static float singleWavelengthRandoms[1024];

static int32_t runGuiApp() {
    const std::filesystem::path exeDir = getExecutableDirectory();

    RenderConfigs renderConfigs;
    loadScene(g_sceneFilePath, &renderConfigs);

    StreamChain<2> streamChain;
    streamChain.initialize(g_gpuEnv.cuContext);
    CUstream defaultStream = streamChain.waitAvailableAndGetCurrentStream();

    // ----------------------------------------------------------------
    // JP: OpenGL, GLFWの初期化。
    // EN: Initialize OpenGL and GLFW.

    glfwSetErrorCallback(
        [](int32_t error, const char* description) {
            hpprintf("Error %d: %s\n", error, description);
        });
    if (!glfwInit()) {
        hpprintf("Failed to initialize GLFW.\n");
        return -1;
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    constexpr bool enableGLDebugCallback = DEBUG_SELECT(true, false);

    // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
    // EN: Create an OpenGL 4.6 core profile context.
    const uint32_t OpenGLMajorVersion = 4;
    const uint32_t OpenGLMinorVersion = 6;
    const char* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if constexpr (enableGLDebugCallback)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    int32_t renderTargetSizeX = 1080;
    //int32_t renderTargetSizeX = 1920;
    int32_t renderTargetSizeY = 1080;

    // JP: ウインドウの初期化。
    // EN: Initialize a window.
    float contentScaleX, contentScaleY;
    glfwGetMonitorContentScale(monitor, &contentScaleX, &contentScaleY);
    float UIScaling = contentScaleX;
    GLFWwindow* window = glfwCreateWindow(
        static_cast<int32_t>(renderTargetSizeX * UIScaling),
        static_cast<int32_t>(renderTargetSizeY * UIScaling),
        "RTCamp10", NULL, NULL);
    glfwSetWindowUserPointer(window, nullptr);
    if (!window) {
        hpprintf("Failed to create a GLFW window.\n");
        glfwTerminate();
        return -1;
    }

    int32_t curFBWidth;
    int32_t curFBHeight;
    glfwGetFramebufferSize(window, &curFBWidth, &curFBHeight);

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1); // Enable vsync



    // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
    // EN: gl3wInit() must be called after some OpenGL context has been created.
    int32_t gl3wRet = gl3wInit();
    if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
        hpprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
        glfwTerminate();
        return -1;
    }

    if constexpr (enableGLDebugCallback) {
        glu::enableDebugCallback(true);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, false);
    }

    // END: Initialize OpenGL and GLFW.
    // ----------------------------------------------------------------



    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    glu::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();

    // JP: OptiXの結果をフレームバッファーにコピーするシェーダー。
    // EN: Shader to copy OptiX result to a frame buffer.
    glu::GraphicsProgram drawOptiXResultShader;
    drawOptiXResultShader.initializeVSPS(
        readTxtFile(exeDir / "renderer/shaders/drawOptiXResult.vert"),
        readTxtFile(exeDir / "renderer/shaders/drawOptiXResult.frag"));



    // ----------------------------------------------------------------
    // JP: 入力コールバックの設定。
    // EN: Set up input callbacks.

    glfwSetMouseButtonCallback(
        window,
        [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
            uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);

            switch (button) {
            case GLFW_MOUSE_BUTTON_1: {
                devPrintf("Mouse Left\n");
                g_buttonRotate.recordStateChange(
                    (mods & GLFW_MOD_ALT) ? action == GLFW_PRESS : false,
                    frameIndex);
                break;
            }
            case GLFW_MOUSE_BUTTON_MIDDLE: {
                devPrintf("Mouse Middle\n");
                g_buttonRotate.recordStateChange(action == GLFW_PRESS, frameIndex);
                break;
            }
            default:
                break;
            }
        });
    glfwSetCursorPosCallback(
        window,
        [](GLFWwindow* window, double x, double y) {
            g_mouseX = x;
            g_mouseY = y;
        });
    glfwSetKeyCallback(
        window,
        [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
            uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);

            switch (key) {
            case GLFW_KEY_W: {
                g_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_S: {
                g_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_A: {
                g_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_D: {
                g_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_R: {
                g_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_F: {
                g_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_Q: {
                g_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_E: {
                g_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_T: {
                g_keyFasterPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_G: {
                g_keySlowerPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            case GLFW_KEY_P: {
                g_keyDebugPrint.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
                break;
            }
            default:
                break;
            }
        });

    // END: Set up input callbacks.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: ImGuiの初期化。
    // EN: Initialize ImGui.

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Setup style
    // JP: ガンマ補正が有効なレンダーターゲットで、同じUIの見た目を得るためにデガンマされたスタイルも用意する。
    // EN: Prepare a degamma-ed style to have the identical UI appearance on gamma-corrected render target.
    ImGuiStyle guiStyle, guiStyleWithGamma;
    ImGui::StyleColorsDark(&guiStyle);
    guiStyleWithGamma = guiStyle;
    const auto degamma = [](const ImVec4 &color) {
        return ImVec4(sRGB_degamma(color.x),
                      sRGB_degamma(color.y),
                      sRGB_degamma(color.z),
                      color.w);
    };
    for (int i = 0; i < ImGuiCol_COUNT; ++i) {
        guiStyleWithGamma.Colors[i] = degamma(guiStyleWithGamma.Colors[i]);
    }
    ImGui::GetStyle() = guiStyleWithGamma;

    // END: Initialize ImGui.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: スクリーン関連のバッファーを初期化。
    // EN: Initialize screen-related buffers.

    cudau::InteropSurfaceObjectHolder<2> outputBufferHolder;
    glu::Sampler outputSampler;

    outputBufferHolder.initialize({ &cudaOutputBuffer });
    outputSampler.initialize(
        glu::Sampler::MinFilter::Nearest, glu::Sampler::MagFilter::Nearest,
        glu::Sampler::WrapMode::Repeat, glu::Sampler::WrapMode::Repeat);

    withGfx = true;
    initializeScreenRelatedBuffers(renderTargetSizeX, renderTargetSizeY);

    // END: Initialize screen-related buffers.
    // ----------------------------------------------------------------



    BoundingBox3D initialSceneAABB = g_scene.computeSceneAABB(renderConfigs.timeBegin);

    setUpPipelineLaunchParameters(renderTargetSizeX, renderTargetSizeY);



    uint64_t frameIndex = 0;
    glfwSetWindowUserPointer(window, &frameIndex);

    uint32_t numAccumFrames = 0;
    float timePoint = renderConfigs.timeBegin;
    bool enableFreeCamera = false;
    Ref<Camera> activeCamera;
    {
        const ActiveCameraInfo &activeCamInfo = renderConfigs.activeCameraInfos[0];
        activeCamera = renderConfigs.cameras.at(activeCamInfo.name);
        activeCamera->setUpDeviceData(&perFramePlpOnHost.camera, timePoint);
        g_cameraPosition = perFramePlpOnHost.camera.position;
        g_cameraOrientation = perFramePlpOnHost.camera.orientation;
        g_tempCameraOrientation = g_cameraOrientation;
        Vector3D sceneDim = initialSceneAABB.maxP - initialSceneAABB.minP;
        if (sceneDim.allFinite())
            g_cameraPositionalMovingSpeed = 0.003f * std::max({ sceneDim.x, sceneDim.y, sceneDim.z });
        else
            g_cameraPositionalMovingSpeed = 0.0f;
        g_cameraDirectionalMovingSpeed = 0.0015f;
        g_cameraTiltSpeed = 0.025f;
    }

    g_scene.setUpDeviceDataBuffers(defaultStream, timePoint);
    g_scene.setUpLightGeomDistributions(defaultStream);
    //g_scene.checkLightGeomDistributions();

    struct GPUTimer {
        cudau::Timer frame;
        cudau::Timer buildASs;
        cudau::Timer computeLightProbs;
        cudau::Timer rendering;

        void initialize() {
            frame.initialize(g_gpuEnv.cuContext);
            buildASs.initialize(g_gpuEnv.cuContext);
            computeLightProbs.initialize(g_gpuEnv.cuContext);
            rendering.initialize(g_gpuEnv.cuContext);
        }

        void finalize() {

        }
    };

    StopWatchHiRes cpuTimer;
    uint64_t frameTime = 0;
    uint64_t setUpDeviceDataTime = 0;

    GPUTimer gpuTimers[2];
    for (GPUTimer &gpuTimer : gpuTimers)
        gpuTimer.initialize();

    std::mt19937 perFrameRng(72139121);
    std::uniform_real_distribution<float> u01;

    while (true) {
        cpuTimer.start();

        uint32_t curBufIdx = frameIndex % 2;
        CUstream curCuStream = streamChain.waitAvailableAndGetCurrentStream();
        GPUTimer &curGpuTimer = gpuTimers[curBufIdx];

        if (glfwWindowShouldClose(window))
            break;
        glfwPollEvents();

        bool resized = false;
        int32_t newFBWidth;
        int32_t newFBHeight;
        glfwGetFramebufferSize(window, &newFBWidth, &newFBHeight);
        if (newFBWidth != curFBWidth || newFBHeight != curFBHeight) {
            curFBWidth = newFBWidth;
            curFBHeight = newFBHeight;

            renderTargetSizeX = curFBWidth / UIScaling;
            renderTargetSizeY = curFBHeight / UIScaling;

            glFinish();
            streamChain.waitAllWorkDone();

            resizeScreenRelatedBuffers(renderTargetSizeX, renderTargetSizeY);

            {
                staticPlpOnHost.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
                staticPlpOnHost.rngBuffer = rngBuffer.getSurfaceObject(0);
                staticPlpOnHost.ltTargetBuffer = ltTargetBuffer.getBlockBuffer2D();
                staticPlpOnHost.accumBuffer = accumBuffer.getBlockBuffer2D();
            }
            CUDADRV_CHECK(cuMemcpyHtoD(staticPlpOnDevice, &staticPlpOnHost, sizeof(staticPlpOnHost)));

            resized = true;
        }

        bool operatingCamera;
        bool cameraIsActuallyMoving;
        static bool operatedCameraOnPrevFrame = false;
        {
            const auto decideDirection = [](const KeyState& a, const KeyState& b) {
                int32_t dir = 0;
                if (a.getState() == true) {
                    if (b.getState() == true)
                        dir = 0;
                    else
                        dir = 1;
                }
                else {
                    if (b.getState() == true)
                        dir = -1;
                    else
                        dir = 0;
                }
                return dir;
            };

            int32_t trackZ = -decideDirection(g_keyForward, g_keyBackward);
            int32_t trackX = -decideDirection(g_keyLeftward, g_keyRightward);
            int32_t trackY = decideDirection(g_keyUpward, g_keyDownward);
            int32_t tiltZ = decideDirection(g_keyTiltRight, g_keyTiltLeft);
            int32_t adjustPosMoveSpeed = decideDirection(g_keyFasterPosMovSpeed, g_keySlowerPosMovSpeed);

            g_cameraPositionalMovingSpeed *= 1.0f + 0.02f * adjustPosMoveSpeed;
            g_cameraPositionalMovingSpeed = std::clamp(g_cameraPositionalMovingSpeed, 1e-6f, 1e+6f);

            static double deltaX = 0, deltaY = 0;
            static double lastX, lastY;
            static double g_prevMouseX = g_mouseX, g_prevMouseY = g_mouseY;
            if (g_buttonRotate.getState() == true) {
                if (g_buttonRotate.getTime() == frameIndex) {
                    lastX = g_mouseX;
                    lastY = g_mouseY;
                }
                else {
                    deltaX = g_mouseX - lastX;
                    deltaY = g_mouseY - lastY;
                }
            }

            float deltaAngle = std::sqrt(deltaX * deltaX + deltaY * deltaY);
            Vector3D axis(deltaY, deltaX, 0);
            axis /= deltaAngle;
            if (deltaAngle == 0.0f)
                axis = Vector3D(1, 0, 0);

            g_cameraOrientation = g_cameraOrientation * qRotateZ(g_cameraTiltSpeed * -tiltZ);
            g_tempCameraOrientation =
                g_cameraOrientation *
                qRotate(g_cameraDirectionalMovingSpeed * -deltaAngle, axis);
            g_cameraPosition +=
                g_tempCameraOrientation.toMatrix3x3() *
                (g_cameraPositionalMovingSpeed * Vector3D(trackX, trackY, trackZ));
            if (g_buttonRotate.getState() == false && g_buttonRotate.getTime() == frameIndex) {
                g_cameraOrientation = g_tempCameraOrientation;
                deltaX = 0;
                deltaY = 0;
            }

            operatingCamera =
                (g_keyForward.getState() || g_keyBackward.getState() ||
                 g_keyLeftward.getState() || g_keyRightward.getState() ||
                 g_keyUpward.getState() || g_keyDownward.getState() ||
                 g_keyTiltLeft.getState() || g_keyTiltRight.getState() ||
                 g_buttonRotate.getState());
            cameraIsActuallyMoving =
                (trackZ != 0 || trackX != 0 || trackY != 0 ||
                 tiltZ != 0 || (g_mouseX != g_prevMouseX) || (g_mouseY != g_prevMouseY))
                && operatingCamera;

            g_prevMouseX = g_mouseX;
            g_prevMouseY = g_mouseY;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();



        // Scene Window
        bool timeChanged = false;
        bool envMapChanged = false;
        {
            ImGui::Begin("Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("Time Range: %f - %f [s]", renderConfigs.timeBegin, renderConfigs.timeEnd);
            //timeChanged = ImGui::InputFloat("Cur. Time", &timePoint, 1.0f / renderConfigs.fps);
            //timePoint = clamp(timePoint, renderConfigs.timeBegin, renderConfigs.timeEnd);
            timeChanged = ImGui::SliderInputFloat(
                "Cur. Time", &timePoint,
                renderConfigs.timeBegin, renderConfigs.timeEnd,
                static_cast<float>(1) / renderConfigs.fps, 0.0f);
            if (timeChanged) {
                activeCamera = renderConfigs.cameras.at(renderConfigs.activeCameraInfos.begin()->name);
                for (auto it = renderConfigs.activeCameraInfos.begin();
                     it != renderConfigs.activeCameraInfos.end(); ++it) {
                    if (timePoint >= it->timePoint)
                        activeCamera = renderConfigs.cameras.at(it->name);
                    else
                        break;
                }
            }

            ImGui::InputFloat3("Cam. Pos.", reinterpret_cast<float*>(&g_cameraPosition));
            ImGui::InputFloat4("Cam. Ori.", reinterpret_cast<float*>(&g_tempCameraOrientation));
            ImGui::Text("Cam. Pos. Speed (T/G): %g", g_cameraPositionalMovingSpeed);
            ImGui::BeginDisabled(!enableFreeCamera);
            if (ImGui::Button("Cam. on track.")) {
                enableFreeCamera = false;
                cameraIsActuallyMoving = true;
            }
            ImGui::EndDisabled();

            ImGui::End();
        }



        // Stats Window
        {
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            static MovingAverageTime maFrameTime;
            static MovingAverageTime maSetUpDeviceDataTime;
            static MovingAverageTime maCudaFrameTime;
            static MovingAverageTime maUpdateTime;
            static MovingAverageTime maComputeLightProbsTime;
            static MovingAverageTime maRenderTime;

            maFrameTime.append(frameTime * 1e-3f);
            maSetUpDeviceDataTime.append(setUpDeviceDataTime * 1e-3f);
            maCudaFrameTime.append(curGpuTimer.frame.report());
            maUpdateTime.append(curGpuTimer.buildASs.report());
            maComputeLightProbsTime.append(curGpuTimer.computeLightProbs.report());
            maRenderTime.append(curGpuTimer.rendering.report());

            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("Frame %.3f [ms]:", maFrameTime.getAverage());
            ImGui::Text("SetUp Device Data %.3f [ms]:", maSetUpDeviceDataTime.getAverage());
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", maCudaFrameTime.getAverage());
            ImGui::Text("  Update: %.3f [ms]", maUpdateTime.getAverage());
            ImGui::Text("  Compute Light Probs: %.3f [ms]", maComputeLightProbsTime.getAverage());
            ImGui::Text("  Render: %.3f [ms]", maRenderTime.getAverage());

            ImGui::End();
        }



        // Debug Window
        bool resetAccumulation = false;
        static bool enableVolume = /*true*/false;
        static float log10VolumeDensity = -1.0f;
        static float scatteringAlbedo = 0.1f;
        {
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            if (ImGui::Checkbox("Volume", &enableVolume))
                resetAccumulation = true;
            if (ImGui::SliderFloat("Density (log10)", &log10VolumeDensity, -3, 3))
                resetAccumulation = true;
            if (ImGui::SliderFloat("Scattering Albedo", &scatteringAlbedo, 0, 1))
                resetAccumulation = true;

            ImGui::Separator();

            ImGui::AlignTextToFramePadding();
            ImGui::Text("Screen Shot:");
            ImGui::SameLine();
            bool saveSS_LDR = ImGui::Button("SDR");
            ImGui::SameLine();
            bool saveSS_HDR = ImGui::Button("HDR");
            ImGui::SameLine();
            if (ImGui::Button("Both"))
                saveSS_LDR = saveSS_HDR = true;
            if (saveSS_LDR || saveSS_HDR) {
                streamChain.waitAllWorkDone();
                auto rawImage = new float[4 * renderTargetSizeX * renderTargetSizeY];
                glGetTextureSubImage(
                    gfxOutputBuffer.getHandle(), 0,
                    0, 0, 0, renderTargetSizeX, renderTargetSizeY, 1,
                    GL_RGBA, GL_FLOAT, sizeof(float4) * renderTargetSizeX * renderTargetSizeY, rawImage);

                float brightness = 0.0f;
                if (saveSS_LDR) {
                    SDRImageSaverConfig config;
                    config.brightnessScale = std::pow(10.0f, brightness);
                    config.applyToneMap = false;
                    config.apply_sRGB_gammaCorrection = true;
                    config.alphaForOverride = 1.0f;
                    saveImage("output.png", renderTargetSizeX, renderTargetSizeY, 4, rawImage,
                              config);
                }

                accumBuffer.map();
                for (int y = 0; y < renderTargetSizeY; ++y) {
                    for (int x = 0; x < renderTargetSizeX; ++x) {
                        const DiscretizedSpectrum &accumResult = accumBuffer(x, y).getValue().result;
                        float colorXYZ[3];
                        accumResult.toXYZ(colorXYZ);
                        float colorRGB[3];
                        transformTristimulus(mat_XYZ_to_Rec709_D65, colorXYZ, colorRGB);
                        float recNumAccums = 1.0f / (numAccumFrames + 1);
                        colorRGB[0] *= recNumAccums;
                        colorRGB[1] *= recNumAccums;
                        colorRGB[2] *= recNumAccums;
                        rawImage[4 * (y * renderTargetSizeX + x) + 0] = colorRGB[0];
                        rawImage[4 * (y * renderTargetSizeX + x) + 1] = colorRGB[1];
                        rawImage[4 * (y * renderTargetSizeX + x) + 2] = colorRGB[2];
                        rawImage[4 * (y * renderTargetSizeX + x) + 3] = 1.0f;
                    }
                }
                accumBuffer.unmap();

                if (saveSS_HDR)
                    saveImageHDR("output.exr", renderTargetSizeX, renderTargetSizeY, 4,
                                 std::pow(10.0f, brightness), rawImage);
                delete[] rawImage;
            }

            ImGui::End();
        }



        // JP: newSequence: temporal accumulationなどのつながりが消えるという意味。
        //     firstAccumFrame: 純粋なサンプルサイズ増加の開始。
        bool newSequence =
            resized
            || frameIndex == 0
            || resetAccumulation;
        bool firstAccumFrame =
            /*animate || !enableAccumulation ||  */
            envMapChanged
            || cameraIsActuallyMoving
            || timeChanged
            || newSequence;
        if (firstAccumFrame)
            numAccumFrames = 0;
        else
            numAccumFrames = /*std::min(*/numAccumFrames + 1/*, (1u << log2MaxNumAccums))*/;
        if (newSequence)
            hpprintf("New sequence started.\n");

        if (operatingCamera)
            enableFreeCamera = true;
        if (enableFreeCamera) {
            perFramePlpOnHost.camera.position = g_cameraPosition;
            perFramePlpOnHost.camera.orientation = g_tempCameraOrientation;
        }
        else {
            perFramePlpOnHost.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
            activeCamera->setUpDeviceData(&perFramePlpOnHost.camera, timePoint);
            g_cameraPosition = perFramePlpOnHost.camera.position;
            g_cameraOrientation = perFramePlpOnHost.camera.orientation;
            g_tempCameraOrientation = g_cameraOrientation;
        }

        perFramePlpOnHost.instances = g_scene.getInstancesOnDevice();

        perFramePlpOnHost.mousePosition = int2(g_mouseX, g_mouseY);

        perFramePlpOnHost.numAccumFrames = numAccumFrames;
        perFramePlpOnHost.volumeDensity = std::pow(10.0f, log10VolumeDensity);
        perFramePlpOnHost.scatteringAlbedo = scatteringAlbedo;
        perFramePlpOnHost.enableVolume = enableVolume;
        perFramePlpOnHost.enableDebugPrint = g_keyDebugPrint.getState();



        cpuTimer.start();
        g_scene.setUpDeviceDataBuffers(curCuStream, timePoint);
        uint32_t setUpDeviceDataTimeIdx = cpuTimer.stop();

        curGpuTimer.frame.start(curCuStream);
        outputBufferHolder.beginCUDAAccess(curCuStream);

        perFramePlpOnHost.outputBuffer = outputBufferHolder.getNext();

        curGpuTimer.buildASs.start(curCuStream);
        perFramePlpOnHost.travHandle = g_scene.buildASs(curCuStream);
        curGpuTimer.buildASs.stop(curCuStream);

        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            perFramePlpOnDevice, &perFramePlpOnHost, sizeof(perFramePlpOnHost), curCuStream));

        curGpuTimer.computeLightProbs.start(curCuStream);
        {
            g_scene.setUpLightInstDistribution(
                curCuStream,
                perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, worldDimInfo),
                perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, lightInstDist),
                perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, dirLightInstDist));
        }
        curGpuTimer.computeLightProbs.stop(curCuStream);
        //g_scene.checkLightInstDistribution(
        //    perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, lightInstDist));
        //g_scene.checkLightInstDistribution(
        //    perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, dirLightInstDist));

        //{
        //    CUDADRV_CHECK(cuStreamSynchronize(curCuStream));
        //    shared::StaticPipelineLaunchParameters s;
        //    shared::PerFramePipelineLaunchParameters f;
        //    CUDADRV_CHECK(cuMemcpyDtoH(&s, staticPlpOnDevice, sizeof(s)));
        //    CUDADRV_CHECK(cuMemcpyDtoH(&f, perFramePlpOnDevice, sizeof(f)));
        //    printf("%g\n", f.lightInstDist.integral());
        //    printf("");
        //}

        curGpuTimer.rendering.start(curCuStream);

#if 0
        g_gpuEnv.pathTracing.setEntryPoint(PathTracingEntryPoint::pathTrace);
        g_gpuEnv.pathTracing.optixPipeline.launch(
            curCuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        //CUDADRV_CHECK(cuStreamSynchronize(curCuStream));

        //g_gpuEnv.clearLtTargetBuffer.launchWithThreadDim(
        //    curCuStream, cudau::dim3(renderTargetSizeX, renderTargetSizeY));
        //g_gpuEnv.lightTracing.setEntryPoint(LightTracingEntryPoint::lightTrace);
        //g_gpuEnv.lightTracing.optixPipeline.launch(
        //    curCuStream, plpOnDevice, numLightTracingPaths, 1, 1);
        //CUDADRV_CHECK(cuStreamSynchronize(curCuStream));
#else
        g_gpuEnv.clearLtTargetBuffer.launchWithThreadDim(
            curCuStream, cudau::dim3(renderTargetSizeX, renderTargetSizeY));

        shared::LvcBptPassInfo lvcBptPassInfoOnHost = {};
        lvcBptPassInfoOnHost.wls = WavelengthSamples::createWithEqualOffsets(
            wavelengthRandoms[numAccumFrames % lengthof(wavelengthRandoms)],
            singleWavelengthRandoms[numAccumFrames % lengthof(wavelengthRandoms)],
            &lvcBptPassInfoOnHost.wlPDens);
        lvcBptPassInfoOnHost.numLightVertices = 0;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            lvcBptPassInfo.getCUdeviceptr(), &lvcBptPassInfoOnHost,
            sizeof(lvcBptPassInfoOnHost), curCuStream));

        g_gpuEnv.lvcBpt.setEntryPoint(LvcBptEntryPoint::GenerateLightVertices);
        g_gpuEnv.lvcBpt.optixPipeline.launch(
            curCuStream, plpOnDevice, numLightTracingPaths, 1, 1);
        //CUDADRV_CHECK(cuStreamSynchronize(curCuStream));
        //CUDADRV_CHECK(cuMemcpyDtoH(
        //    &lvcBptPassInfoOnHost, lvcBptPassInfo.getCUdeviceptr(),
        //    sizeof(lvcBptPassInfoOnHost)));
        //std::vector<shared::LightPathVertex> lightVertexCacheOnHost = lightVertexCache;
        //std::vector<uint32_t> counts;
        //for (uint32_t i = 0; i < lvcBptPassInfoOnHost.numLightVertices; ++i) {
        //    const shared::LightPathVertex &v = lightVertexCacheOnHost[i];
        //    if (v.pathLength >= counts.size())
        //        counts.resize(v.pathLength + 1, 0u);
        //    ++counts[v.pathLength];

        //    if (v.pathLength == 1)
        //        printf("");
        //    else if (v.pathLength == 2)
        //        printf("");
        //    else if (v.pathLength == 3)
        //        printf("");
        //}

        g_gpuEnv.lvcBpt.setEntryPoint(LvcBptEntryPoint::EyePaths);
        g_gpuEnv.lvcBpt.optixPipeline.launch(
            curCuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
#endif

        curGpuTimer.rendering.stop(curCuStream);

        g_gpuEnv.applyToneMap.launchWithThreadDim(
            curCuStream, cudau::dim3(renderTargetSizeX, renderTargetSizeY));
        //CUDADRV_CHECK(cuStreamSynchronize(curCuStream));

        outputBufferHolder.endCUDAAccess(curCuStream, true);
        curGpuTimer.frame.stop(curCuStream);

        //CUDADRV_CHECK(cuStreamSynchronize(curCuStream));
        //glFinish();
        //{
        //    SDRImageSaverConfig config = {};
        //    config.alphaForOverride = 1.0f;
        //    config.applyToneMap = false;
        //    config.apply_sRGB_gammaCorrection = false;
        //    config.brightnessScale = 1.0f;
        //    config.flipY = false;
        //    //saveImage("output.png", accumBuffer, config);
        //    auto rawImage = new float4[renderTargetSizeX * renderTargetSizeY];
        //    std::fill(rawImage, rawImage + renderTargetSizeX * renderTargetSizeY, 1.0f);
        //    glGetTextureSubImage(
        //        gfxOutputBuffer.getHandle(), 0,
        //        0, 0, 0, renderTargetSizeX, renderTargetSizeY, 1,
        //        GL_RGBA, GL_FLOAT, sizeof(float4) * renderTargetSizeX * renderTargetSizeY, rawImage);
        //    saveImage("output.png", renderTargetSizeX, renderTargetSizeY, rawImage, config);
        //    delete[] rawImage;

        //    printf("");
        //}



        glEnable(GL_FRAMEBUFFER_SRGB);
        ImGui::GetStyle() = guiStyleWithGamma;
        glViewport(0, 0, curFBWidth, curFBHeight);
        //glClearColor(0.0f, 0.0f, 0.05f, 1.0f);
        ////glClearDepth(1.0f);
        //glClear(GL_COLOR_BUFFER_BIT/* | GL_DEPTH_BUFFER_BIT*/);

        glUseProgram(drawOptiXResultShader.getHandle());

        glUniform2ui(0, curFBWidth, curFBHeight);
        glBindTextureUnit(0, gfxOutputBuffer.getHandle());
        glBindSampler(0, outputSampler.getHandle());

        glBindVertexArray(vertexArrayForFullScreen.getHandle());
        glDrawArrays(GL_TRIANGLES, 0, 3);
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glDisable(GL_FRAMEBUFFER_SRGB);

        glfwSwapBuffers(window);
        streamChain.swap();

        ++frameIndex;

        uint32_t frameTimeIdx = cpuTimer.stop();

        setUpDeviceDataTime = cpuTimer.getMeasurement(
            setUpDeviceDataTimeIdx, StopWatchDurationType::Microseconds);
        frameTime = cpuTimer.getMeasurement(
            frameTimeIdx, StopWatchDurationType::Microseconds);
        cpuTimer.reset();
    }

    finalizeScreenRelatedBuffers();



    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();

    //streamChain.finalize();

    return 0;
}



static int32_t runApp() {
    hpprintf("Loading a scene:\n");
    RenderConfigs renderConfigs;
    loadScene(g_sceneFilePath, &renderConfigs);
    hpprintf("Loading a scene: done. (tp: %.3f [s])\n",
             g_globalTimer.getElapsedFromRoot(StopWatchDurationType::Milliseconds) * 1e-3f);

    hpprintf("Set up misc...");

    CUstream cuStream;
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    withGfx = false;
    initializeScreenRelatedBuffers(renderConfigs.imageWidth, renderConfigs.imageHeight);

    setUpPipelineLaunchParameters(renderConfigs.imageWidth, renderConfigs.imageHeight);

    perFramePlpOnHost.camera.aspect =
        static_cast<float>(renderConfigs.imageWidth) / renderConfigs.imageHeight;
    perFramePlpOnHost.outputBuffer = cudaOutputBuffer.getSurfaceObject(0);
    perFramePlpOnHost.mousePosition = int2(0, 0);
    perFramePlpOnHost.volumeDensity = 0.01f;
    perFramePlpOnHost.scatteringAlbedo = 0.1f;
    perFramePlpOnHost.enableVolume = true;
    perFramePlpOnHost.enableDebugPrint = false;

    uint32_t activeCamIdx = 0;
    const uint32_t numActiveCams = renderConfigs.activeCameraInfos.size();
    Ref<Camera> activeCamera;
    {
        const ActiveCameraInfo &activeCamInfo = renderConfigs.activeCameraInfos[activeCamIdx];
        activeCamera = renderConfigs.cameras.at(activeCamInfo.name);
    }

    g_scene.setUpDeviceDataBuffers(cuStream, renderConfigs.timeBegin);
    g_scene.setUpLightGeomDistributions(cuStream);
    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    StopWatchHiRes cpuTimer;

    SDRImageSaverConfig imageSaverConfig = {};
    imageSaverConfig.alphaForOverride = 1.0f;
    imageSaverConfig.applyToneMap = false;
    imageSaverConfig.apply_sRGB_gammaCorrection = true;
    imageSaverConfig.brightnessScale = 1.0f;
    imageSaverConfig.flipY = false;
    initImageSaverThread();

    constexpr bool printEnqueueSaveImageTime = false;

    std::mt19937 perFrameRng(72139121);

    const uint32_t numSamplesPerFrame = g_spp;

    hpprintf(" done. (tp: %.3f [s])\n",
             g_globalTimer.getElapsedFromRoot(StopWatchDurationType::Milliseconds) * 1e-3f);

    uint32_t timeStepIndex = g_beginTimeStepIndex;
    while (true) {
        if (timeStepIndex > g_endTimeStepIndex)
            break;
        float timePoint =
            renderConfigs.timeBegin
            + g_playbackSpeed * static_cast<float>(timeStepIndex) / renderConfigs.fps;
        if (timePoint > renderConfigs.timeEnd)
            break;

        hpprintf("Frame %u (Time: %.3f) (tp: %.3f [s])\n",
                 timeStepIndex, timePoint,
                 g_globalTimer.getElapsedFromRoot(StopWatchDurationType::Milliseconds) * 1e-3f);

        const ActiveCameraInfo &nextActiveCamInfo =
            renderConfigs.activeCameraInfos[std::min(activeCamIdx + 1, numActiveCams - 1)];
        if (timePoint >= nextActiveCamInfo.timePoint) {
            Ref<Camera> nextActiveCamera = renderConfigs.cameras.at(nextActiveCamInfo.name);
            if (nextActiveCamera != activeCamera)
                ++activeCamIdx;
            activeCamera = nextActiveCamera;
        }
        activeCamera->setUpDeviceData(&perFramePlpOnHost.camera, timePoint);

        perFramePlpOnHost.instances = g_scene.getInstancesOnDevice();

        g_scene.setUpDeviceDataBuffers(cuStream, timePoint);

        perFramePlpOnHost.travHandle = g_scene.buildASs(cuStream);

        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            perFramePlpOnDevice, &perFramePlpOnHost, sizeof(perFramePlpOnHost), cuStream));

        g_scene.setUpLightInstDistribution(
            cuStream,
            perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, worldDimInfo),
            perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, lightInstDist),
            perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, dirLightInstDist));
        //g_scene.checkLightInstDistribution(
        //    perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, lightInstDist));

        for (int i = 0; i < numSamplesPerFrame; ++i) {
            CUDADRV_CHECK(cuMemcpyHtoDAsync(
                perFramePlpOnDevice + offsetof(shared::PerFramePipelineLaunchParameters, numAccumFrames),
                &i, sizeof(i), cuStream));
            g_gpuEnv.pathTracing.setEntryPoint(PathTracingEntryPoint::pathTrace);
            g_gpuEnv.pathTracing.optixPipeline.launch(
                cuStream, plpOnDevice, renderConfigs.imageWidth, renderConfigs.imageHeight, 1);
            //g_gpuEnv.lightTracing.setEntryPoint(LightTracingEntryPoint::lightTrace);
            //g_gpuEnv.lightTracing.optixPipeline.launch(
            //    cuStream, plpOnDevice, numLightTracingPaths, 1, 1);
        }

        g_gpuEnv.applyToneMap.launchWithThreadDim(
            cuStream, cudau::dim3(renderConfigs.imageWidth, renderConfigs.imageHeight));

        CUDADRV_CHECK(cuStreamSynchronize(cuStream));

        if constexpr (printEnqueueSaveImageTime)
            cpuTimer.start();
        char filename[256];
        sprintf_s(filename, "%03u.png", timeStepIndex);
        enqueueSaveImage(filename, cudaOutputBuffer, imageSaverConfig);
        if constexpr (printEnqueueSaveImageTime) {
            uint64_t saveTime = cpuTimer.getElapsed(StopWatchDurationType::Milliseconds);
            hpprintf("Save (Enqueue) %s: %.3f [s]\n", filename, saveTime * 1e-3f);
        }

        ++timeStepIndex;
    }

    finishImageSaverThread();

    return 0;
}



static void parseCommandline(int32_t argc, const char* argv[]) {
    for (int i = 0; i < argc; ++i) {
        const char* arg = argv[i];

        if (strncmp(arg, "-", 1) != 0)
            continue;

        if (strncmp(arg, "-no-gui", 8) == 0) {
            g_guiMode = false;
        }
        else if (strncmp(arg, "-scene", 7) == 0) {
            if (i + 1 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_sceneFilePath = argv[i + 1];
            i += 1;
        }
        else if (strncmp(arg, "-t-mul", 7) == 0) {
            if (i + 1 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_playbackSpeed = std::atof(argv[i + 1]);
            i += 1;
        }
        else if (strncmp(arg, "-spp", 5) == 0) {
            if (i + 1 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_spp = std::atoi(argv[i + 1]);
            i += 1;
        }
        else if (strncmp(arg, "-range", 7) == 0) {
            if (i + 2 >= argc) {
                hpprintf("Invalid option.\n");
                exit(EXIT_FAILURE);
            }
            g_beginTimeStepIndex = std::atoi(argv[i + 1]);
            g_endTimeStepIndex = std::atoi(argv[i + 2]);
            i += 2;
        }
        else {
            printf("Unknown option.\n");
            exit(EXIT_FAILURE);
        }
    }
}

int32_t mainFunc(int32_t argc, const char* argv[]) {
    g_globalTimer.start();

    parseCommandline(argc, argv);

    initializeColorSystem();

    g_gpuEnv.initialize();

    LambertianSurfaceMaterial::setBSDFProcedureSet();
    SpecularScatteringSurfaceMaterial::setBSDFProcedureSet();
    SimplePBRSurfaceMaterial::setBSDFProcedureSet();

    g_gpuEnv.setupDeviceData();

    g_scene.initialize();

    for (uint32_t i = 0; i < lengthof(wavelengthRandoms); ++i) {
        wavelengthRandoms[i] = computeHaltonSequence(2, i);
        singleWavelengthRandoms[i] = computeHaltonSequence(3, i);
    }

    int32_t ret;
    if (g_guiMode)
        ret = runGuiApp();
    else
        ret = runApp();

    //g_scene.finalize();
    g_gpuEnv.finalize();

    finalizeColorSystem();

    hpprintf("End of App (tp: %.3f [s])\n",
             g_globalTimer.getElapsedFromRoot(StopWatchDurationType::Milliseconds) * 1e-3f);

    return ret;
}

} // namespace rtc10



int32_t main(int32_t argc, const char* argv[]) {
	try {
		return rtc10::mainFunc(argc, argv);
	}
	catch (const std::exception &ex) {
		hpprintf("Error: %s\n", ex.what());
		return -1;
	}
}