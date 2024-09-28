#pragma once

#include "renderer_shared.h"
#include "../common/common_renderer_host.h"
#include "../ext/cubd/cubd.h"

namespace rtc10 {

constexpr cudau::BufferType bufferType = cudau::BufferType::Device;

template <typename T>
using Ref = std::shared_ptr<T>;

template <typename T>
using TypedBufferRef = std::shared_ptr<cudau::TypedBuffer<T>>;



enum class PathTracingEntryPoint {
    pathTrace,
};

enum class LightTracingEntryPoint {
    lightTrace,
};

struct GPUEnvironment {
    CUcontext cuContext;
    optixu::Context optixContext;

    CUmodule postProcessKernelsModule;
    CUdeviceptr plpForPostProcessKernelsModule;
    cudau::Kernel applyToneMap;
    cudau::Kernel clearLtTargetBuffer;

    struct ComputeLightProbs {
        CUmodule cudaModule;
        cudau::Kernel initializeWorldDimInfo;
        cudau::Kernel finalizeWorldDimInfo;
        cudau::Kernel computeTriangleProbBuffer;
        cudau::Kernel computeGeomInstProbBuffer;
        cudau::Kernel computeInstProbBuffer;
        cudau::Kernel finalizeDiscreteDistribution1D;
        cudau::Kernel computeFirstMipOfEnvIBLImportanceMap;
        cudau::Kernel computeMipOfImportanceMap;
        cudau::Kernel testImportanceMap;
        CUdeviceptr debugPlp;
    } computeLightProbs;

    cudau::TypedBuffer<shared::BSDFProcedureSet> bsdfProcedureSetBuffer;
    std::vector<shared::BSDFProcedureSet> mappedBsdfProcedureSets;

    optixu::Material optixDefaultMaterial;

    template <typename EntryPointType>
    struct Pipeline {
        optixu::Pipeline optixPipeline;
        optixu::Module optixModule;
        std::unordered_map<EntryPointType, optixu::Program> entryPoints;
        std::unordered_map<std::string, optixu::HitProgramGroup> hitPrograms;
        std::unordered_map<std::string, optixu::Program> missPrograms;
        std::vector<optixu::CallableProgramGroup> callablePrograms;
        cudau::Buffer sbt;
        cudau::Buffer hitGroupSbt;

        void setEntryPoint(EntryPointType et) {
            optixPipeline.setRayGenerationProgram(entryPoints.at(et));
        }
    };

    Pipeline<PathTracingEntryPoint> pathTracing;
    Pipeline<LightTracingEntryPoint> lightTracing;

    void initialize();

    void finalize() {
    }

    void setupDeviceData() {
        bsdfProcedureSetBuffer.initialize(cuContext, bufferType, mappedBsdfProcedureSets);
    }

    uint32_t registerBSDFProcedureSet(const shared::BSDFProcedureSet &procSet) {
        uint32_t index = static_cast<uint32_t>(mappedBsdfProcedureSets.size());
        mappedBsdfProcedureSets.resize(index + 1);
        mappedBsdfProcedureSets[index] = procSet;
        return index;
    }
};

extern GPUEnvironment g_gpuEnv;



class Texture2D;
class SurfaceMaterial;
class Geometry;
class GeometryGroup;
class Instance;



class Scene {
    static constexpr uint32_t maxNumPrimitivesPerGeometry = 1 << 16;
    static constexpr uint32_t maxNumMaterials = 1 << 10;
    static constexpr uint32_t maxNumGeometryInstances = 1 << 16;
    static constexpr uint32_t maxNumGeometryGroups = 1 << 12;
    static constexpr uint32_t maxNumInstances = 1 << 16;

    optixu::Scene m_optixScene;

    SlotFinder m_surfaceMaterialSlotFinder;
    SlotFinder m_geometryInstanceSlotFinder;
    SlotFinder m_geometryGroupSlotFinder;
    SlotFinder m_instanceSlotFinder;
    cudau::TypedBuffer<shared::SurfaceMaterial> m_surfaceMaterialBuffer;
    cudau::TypedBuffer<shared::GeometryInstance> m_geometryInstanceBuffer;
    cudau::TypedBuffer<shared::GeometryGroup> m_geometryGroupBuffer;
    cudau::TypedBuffer<shared::Instance> m_instanceBuffer;
    std::unordered_map<uint32_t, Ref<SurfaceMaterial>> m_surfaceMaterialSlotOwners;
    std::unordered_map<uint32_t, Ref<Geometry>> m_geometryInstanceSlotOwners;
    std::unordered_map<uint32_t, Ref<GeometryGroup>> m_geometryGroupSlotOwners;
    std::unordered_map<uint32_t, Ref<Instance>> m_instanceSlotOwners;

    optixu::InstanceAccelerationStructure m_optixIas;
    cudau::Buffer m_optixAsScratchMem;
    cudau::Buffer m_optixAsMem;
    cudau::TypedBuffer<OptixInstance> m_optixInstanceBuffer;

    float m_densityCoeff;
    float m_majorant;

    LightDistribution lightInstDist;
    LightDistribution dirLightInstDist;
    cudau::Buffer m_scanScratchMem;

public:
    void initialize() {
        m_optixScene = g_gpuEnv.optixContext.createScene();

        m_surfaceMaterialSlotFinder.initialize(maxNumMaterials);
        m_surfaceMaterialBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumMaterials);
        m_geometryInstanceSlotFinder.initialize(maxNumGeometryInstances);
        m_geometryInstanceBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumGeometryInstances);
        m_geometryGroupSlotFinder.initialize(maxNumGeometryGroups);
        m_geometryGroupBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumGeometryGroups);
        m_instanceSlotFinder.initialize(maxNumInstances);
        m_instanceBuffer.initialize(g_gpuEnv.cuContext, bufferType, maxNumInstances);

        m_optixIas = m_optixScene.createInstanceAccelerationStructure();
        m_optixIas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::No);

        lightInstDist.initialize(g_gpuEnv.cuContext, bufferType, nullptr, maxNumInstances);
        dirLightInstDist.initialize(g_gpuEnv.cuContext, bufferType, nullptr, maxNumInstances);

        size_t scanScratchSize;
        constexpr int32_t maxScanSize = std::max<int32_t>({
            maxNumPrimitivesPerGeometry,
            maxNumMaterials,
            maxNumGeometryInstances,
            maxNumInstances });
        CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
            nullptr, scanScratchSize,
            static_cast<float*>(nullptr), static_cast<float*>(nullptr), maxScanSize));
        m_scanScratchMem.initialize(g_gpuEnv.cuContext, bufferType, scanScratchSize, 1u);
    }

    void finalize() {

    }



    void allocateSurfaceMaterial(const Ref<SurfaceMaterial> &surfMat);

    void allocateGeometryInstance(const Ref<Geometry> &geom);

    void allocateGeometryAccelerationStructure(const Ref<GeometryGroup> &geomGroup);

    void allocateInstance(const Ref<Instance> &inst);



    BoundingBox3D computeSceneAABB(float timePoint) const;

    shared::SurfaceMaterial* getSurfaceMaterialsOnDevice() const {
        return m_surfaceMaterialBuffer.getDevicePointer();
    }

    shared::GeometryInstance* getGeometryInstancesOnDevice() const {
        return m_geometryInstanceBuffer.getDevicePointer();
    }

    shared::GeometryGroup* getGeometryGroupsOnDevice() const {
        return m_geometryGroupBuffer.getDevicePointer();
    }

    shared::Instance* getInstancesOnDevice() const {
        return m_instanceBuffer.getDevicePointer();
    }



    void setUpDeviceDataBuffers(CUstream stream, float timePoint);
    OptixTraversableHandle buildASs(CUstream stream);



    const cudau::Buffer &getScanScratchMemory() const {
        return m_scanScratchMem;
    }
    
    void setUpLightGeomDistributions(CUstream stream);
    void checkLightGeomDistributions();
    void setUpLightInstDistribution(
        CUstream stream, CUdeviceptr worldDimInfoAddr,
        CUdeviceptr lightInstDistAddr, CUdeviceptr dirLightInstDistAddr);
    void checkLightInstDistribution(CUdeviceptr lightInstDistAddr);
};

extern Scene g_scene;



class Texture2D {
    Ref<cudau::Array> m_texArray;
    cudau::TextureSampler m_sampler;
    CUtexObject m_texObject;
    uint32_t m_dirty : 1;

public:
    Texture2D(const Ref<cudau::Array> &texArray) :
        m_texArray(texArray), m_texObject(0), m_dirty(true) {
        m_sampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
        m_sampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
        m_sampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
        m_sampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    }
    ~Texture2D() {
        if (m_texObject)
            /*CUDADRV_CHECK(*/cuTexObjectDestroy(m_texObject)/*)*/;
    }

    void setXyFilterMode(cudau::TextureFilterMode filterMode) {
        m_sampler.setXyFilterMode(filterMode);
        m_dirty = true;
    }
    void setReadMode(cudau::TextureReadMode readMode) {
        m_sampler.setReadMode(readMode);
        m_dirty = true;
    }

    uint32_t getWidth() const {
        return m_texArray->getWidth();
    }
    uint32_t getHeight() const {
        return m_texArray->getHeight();
    }

    Ref<cudau::Array> getCudaArray() const {
        return m_texArray;
    }

    CUtexObject getDeviceTexture() {
        if (m_dirty) {
            if (m_texObject)
                CUDADRV_CHECK(cuTexObjectDestroy(m_texObject));
            m_texObject = m_sampler.createTextureObject(*m_texArray);
            m_dirty = false;
        }
        return m_texObject;
    }

    shared::TexDimInfo getDimInfo() const {
        shared::TexDimInfo dimInfo = {};
        uint32_t w = m_texArray->getWidth();
        uint32_t h = m_texArray->getHeight();
        bool wIsPowerOfTwo = (w & (w - 1)) == 0;
        bool hIsPowerOfTwo = (h & (h - 1)) == 0;
        dimInfo.dimX = w;
        dimInfo.dimY = h;
        dimInfo.isNonPowerOfTwo = !wIsPowerOfTwo || !hIsPowerOfTwo;
        dimInfo.isBCTexture = m_texArray->isBCTexture();
        dimInfo.isLeftHanded = false;
        return dimInfo;
    }

    CUsurfObject getSurfaceObject(uint32_t mipmapLevel) {
        return m_texArray->getSurfaceObject(mipmapLevel);
    }
};



class SurfaceMaterial {
protected:
    uint32_t m_slot;
    Ref<Texture2D> m_emittance;
    uint32_t m_emitterType : 2;
    uint32_t m_dirty : 1;

public:
    SurfaceMaterial() : m_slot(0xFFFFFFFF), m_emitterType(static_cast<uint32_t>(shared::EmitterType::Diffuse)), m_dirty(false) {}
    ~SurfaceMaterial() {}

    void associateScene(uint32_t slot) {
        m_slot = slot;
        m_dirty = true;
    }

    void setEmittance(const Ref<Texture2D> &emittance, shared::EmitterType emitterType) {
        m_emittance = emittance;
        m_emitterType = static_cast<uint32_t>(emitterType);
    }

    bool isEmissive() const {
        return m_emittance != nullptr;
    }

    shared::EmitterType getEmitterType() const {
        return static_cast<shared::EmitterType>(m_emitterType);
    }

    uint32_t getSlot() const {
        return m_slot;
    }

    virtual bool setUpDeviceData(shared::SurfaceMaterial* deviceData)  {
        if (m_emittance)
            deviceData->emittance = m_emittance->getDeviceTexture();
        deviceData->emitterType = m_emitterType;
        return true;
    }
};



class LambertianSurfaceMaterial : public SurfaceMaterial {
    static uint32_t s_procSetSlot;

    Ref<Texture2D> m_reflectance;

public:
    static void setBSDFProcedureSet() {
        shared::BSDFProcedureSet procSet;
        procSet.setupBSDFBody = CallableProgram_setupLambertBRDF;
        procSet.getSurfaceParameters = CallableProgram_LambertBRDF_getSurfaceParameters;
        procSet.sampleF = CallableProgram_LambertBRDF_sampleF;
        procSet.evaluateF = CallableProgram_LambertBRDF_evaluateF;
        procSet.evaluatePDF = CallableProgram_LambertBRDF_evaluatePDF;
        procSet.evaluateDHReflectanceEstimate = CallableProgram_LambertBRDF_evaluateDHReflectanceEstimate;
        s_procSetSlot = g_gpuEnv.registerBSDFProcedureSet(procSet);
    }

    LambertianSurfaceMaterial() {}

    void set(
        const Ref<Texture2D> &reflectance) {
        m_reflectance = reflectance;

        m_dirty = true;
    }

    bool setUpDeviceData(shared::SurfaceMaterial* deviceData) override {
        if (!m_dirty)
            return false;

        *deviceData = {};
        SurfaceMaterial::setUpDeviceData(deviceData);
        auto &body = *reinterpret_cast<shared::LambertianSurfaceMaterial*>(deviceData->body);
        body.reflectance = m_reflectance->getDeviceTexture();
        body.reflectanceDimInfo = m_reflectance->getDimInfo();
        deviceData->bsdfProcSetSlot = s_procSetSlot;
        deviceData->setupBSDFBody = CallableProgram_setupLambertBRDF;

        m_dirty = false;
        return true;
    }
};



class SimplePBRSurfaceMaterial : public SurfaceMaterial {
    static uint32_t s_procSetSlot;

    Ref<Texture2D> m_baseColor_opacity;
    Ref<Texture2D> m_occlusion_roughness_metallic;

public:
    static void setBSDFProcedureSet() {
        shared::BSDFProcedureSet procSet;
        procSet.setupBSDFBody = CallableProgram_setupSimplePBR_BRDF;
        procSet.getSurfaceParameters = CallableProgram_DichromaticBRDF_getSurfaceParameters;
        procSet.sampleF = CallableProgram_DichromaticBRDF_sampleF;
        procSet.evaluateF = CallableProgram_DichromaticBRDF_evaluateF;
        procSet.evaluatePDF = CallableProgram_DichromaticBRDF_evaluatePDF;
        procSet.evaluateDHReflectanceEstimate = CallableProgram_DichromaticBRDF_evaluateDHReflectanceEstimate;
        s_procSetSlot = g_gpuEnv.registerBSDFProcedureSet(procSet);
    }

    SimplePBRSurfaceMaterial() {}

    void set(
        const Ref<Texture2D> &baseColor_opacity,
        const Ref<Texture2D> &occlusion_roughness_metallic) {
        m_baseColor_opacity = baseColor_opacity;
        m_occlusion_roughness_metallic = occlusion_roughness_metallic;

        m_dirty = true;
    }

    bool setUpDeviceData(shared::SurfaceMaterial* deviceData) override {
        if (!m_dirty)
            return false;

        *deviceData = {};
        SurfaceMaterial::setUpDeviceData(deviceData);
        auto &body = *reinterpret_cast<shared::SimplePBRSurfaceMaterial*>(deviceData->body);
        body.baseColor_opacity = m_baseColor_opacity->getDeviceTexture();
        body.baseColor_opacity_dimInfo = m_occlusion_roughness_metallic->getDimInfo();
        body.occlusion_roughness_metallic = m_occlusion_roughness_metallic->getDeviceTexture();
        body.occlusion_roughness_metallic_dimInfo = m_occlusion_roughness_metallic->getDimInfo();
        deviceData->bsdfProcSetSlot = s_procSetSlot;
        deviceData->setupBSDFBody = CallableProgram_setupSimplePBR_BRDF;

        m_dirty = false;
        return true;
    }
};



enum class BumpMapTextureType {
    NormalMap = 0,
    NormalMap_BC,
    NormalMap_BC_2ch,
    HeightMap,
    HeightMap_BC,
};



struct VertexBuffer {
    std::vector<shared::Vertex> onHost;
    cudau::TypedBuffer<shared::Vertex> onDevice;
};



class Geometry {
protected:
    optixu::GeometryInstance m_optixGeomInst;
    uint32_t m_slot;
    BoundingBox3D m_aabb;
    uint32_t m_dirty : 1;

public:
    Geometry() : m_slot(0xFFFFFFFF), m_dirty(false) {}
    virtual ~Geometry() {}

    virtual void associateScene(uint32_t slot, optixu::GeometryInstance optixGeomInst) {
        m_slot = slot;
        m_optixGeomInst = optixGeomInst;
        m_dirty = true;
    }

    const BoundingBox3D &getAABB() const {
        return m_aabb;
    }

    optixu::GeometryInstance getOptixGeometryInstance() const {
        return m_optixGeomInst;
    }

    uint32_t getSlot() const {
        return m_slot;
    }

    virtual bool setUpDeviceData(shared::GeometryInstance* deviceData) {
        return true;
    }

    virtual void setUpLightDistribution(
        CUstream stream,
        const shared::GeometryInstance* deviceData,
        const shared::SurfaceMaterial* surfMatBuffer) const = 0;
    virtual void scanLightDistribution(
        CUstream stream,
        const shared::GeometryInstance* deviceData) const = 0;
    virtual void finalizeLightDistribution(
        CUstream stream,
        const shared::GeometryInstance* deviceData) const = 0;
};



class TriangleMeshGeometry : public Geometry {
    Ref<VertexBuffer> m_vertices;
    cudau::TypedBuffer<shared::Triangle> m_triangles;
    LightDistribution m_emitterPrimDist;
    Ref<Texture2D> m_normalMap;
    Ref<SurfaceMaterial> m_surfMat;
    BumpMapTextureType m_bumpMapType;

public:
    TriangleMeshGeometry() : m_bumpMapType(BumpMapTextureType::NormalMap_BC_2ch) {}
    ~TriangleMeshGeometry() {
        m_triangles.finalize();
    }

    void set(
        const Ref<VertexBuffer> &vertices,
        const std::vector<shared::Triangle> &triangles,
        const Ref<Texture2D> &normalMap, BumpMapTextureType bumpMapType,
        const Ref<SurfaceMaterial> &surfMat) {
        m_vertices = vertices;
        m_triangles.initialize(g_gpuEnv.cuContext, bufferType, triangles);
        if (surfMat->isEmissive())
            m_emitterPrimDist.initialize(g_gpuEnv.cuContext, bufferType, nullptr, m_triangles.numElements());
        m_normalMap = normalMap;
        m_bumpMapType = bumpMapType;
        m_surfMat = surfMat;
        m_aabb = BoundingBox3D();
        for (int triIdx = 0; triIdx < triangles.size(); ++triIdx) {
            const shared::Triangle &tri = triangles[triIdx];
            const shared::Vertex(&vs)[3] = {
                vertices->onHost[tri.indices[0]],
                vertices->onHost[tri.indices[1]],
                vertices->onHost[tri.indices[2]],
            };
            m_aabb
                .unify(vs[0].position)
                .unify(vs[1].position)
                .unify(vs[2].position);
        }

        m_optixGeomInst.setVertexBuffer(m_vertices->onDevice);
        m_optixGeomInst.setTriangleBuffer(m_triangles);

        m_dirty = true;
    }

    bool setUpDeviceData(shared::GeometryInstance* deviceData) override {
        if (!m_dirty)
            return false;

        *deviceData = {};
        Geometry::setUpDeviceData(deviceData);
        deviceData->vertices = m_vertices->onDevice.getDevicePointer();
        deviceData->triangles = m_triangles.getDevicePointer();
        if (m_emitterPrimDist.isInitialized())
            m_emitterPrimDist.getDeviceType(&deviceData->emitterPrimDist);
        else
            std::memset(&deviceData->emitterPrimDist, 0, sizeof(deviceData->emitterPrimDist));
        deviceData->normal = m_normalMap->getDeviceTexture();
        deviceData->normalDimInfo = m_normalMap->getDimInfo();
        if (m_bumpMapType == BumpMapTextureType::NormalMap ||
            m_bumpMapType == BumpMapTextureType::NormalMap_BC)
            deviceData->readModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap;
        else if (m_bumpMapType == BumpMapTextureType::NormalMap_BC_2ch)
            deviceData->readModifiedNormal = CallableProgram_readModifiedNormalFromNormalMap2ch;
        else
            deviceData->readModifiedNormal = CallableProgram_readModifiedNormalFromHeightMap;
        deviceData->aabb = m_aabb;
        deviceData->surfMatSlot = m_surfMat->getSlot();

        m_dirty = false;
        return true;
    }

    void setUpLightDistribution(
        CUstream stream,
        const shared::GeometryInstance* deviceData,
        const shared::SurfaceMaterial* surfMatBuffer) const override {
        if (!m_emitterPrimDist.isInitialized())
            return;
        uint32_t numTriangles = m_triangles.numElements();
        g_gpuEnv.computeLightProbs.computeTriangleProbBuffer.launchWithThreadDim(
            stream, cudau::dim3(numTriangles),
            deviceData, numTriangles,
            surfMatBuffer);
    }
    void scanLightDistribution(
        CUstream stream,
        const shared::GeometryInstance* deviceData) const override {
        if (!m_emitterPrimDist.isInitialized())
            return;
        uint32_t numTriangles = m_triangles.numElements();
        const cudau::Buffer &scanScratchMem = g_scene.getScanScratchMemory();
        size_t scratchMemSize = scanScratchMem.sizeInBytes();
        CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
            scanScratchMem.getDevicePointer(), scratchMemSize,
            m_emitterPrimDist.weightsOnDevice(),
            m_emitterPrimDist.cdfOnDevice(),
            numTriangles, stream));
    }
    void finalizeLightDistribution(
        CUstream stream,
        const shared::GeometryInstance* deviceData) const override {
        if (!m_emitterPrimDist.isInitialized())
            return;
        g_gpuEnv.computeLightProbs.finalizeDiscreteDistribution1D.launchWithThreadDim(
            stream, cudau::dim3(1),
            &deviceData->emitterPrimDist);
    }
};



class GeometryGroup {
    optixu::GeometryAccelerationStructure m_optixGas;
    uint32_t m_slot;
    std::set<Ref<Geometry>> m_geometries;
    BoundingBox3D m_aabb;

    cudau::Buffer m_optixAsScratchMem;
    cudau::Buffer m_optixAsMem;
    cudau::TypedBuffer<uint32_t> m_geomInstSlotBuffer;
    LightDistribution m_lightGeomInstDist;
    LightDistribution m_dirLightGeomInstDist;
    uint32_t m_dirty : 1;
    uint32_t m_gasIsDirty : 1;

public:
    GeometryGroup() :
        m_slot(0xFFFFFFFF), m_dirty(false), m_gasIsDirty(false) {}

    void associateScene(uint32_t slot, optixu::GeometryAccelerationStructure optixGas) {
        m_slot = slot;
        m_optixGas = optixGas;
        m_dirty = true;
        m_gasIsDirty = true;
    }

    void set(const std::set<Ref<Geometry>> &geoms) {
        m_aabb = BoundingBox3D();
        std::vector<uint32_t> geomInstSlots(geoms.size());
        uint32_t nextIndex = 0;
        for (const Ref<Geometry> &geom : geoms) {
            m_geometries.insert(geom);
            m_optixGas.addChild(geom->getOptixGeometryInstance());
            m_aabb.unify(geom->getAABB());
            geomInstSlots[nextIndex] = geom->getSlot();
            ++nextIndex;
        }
        m_geomInstSlotBuffer.initialize(g_gpuEnv.cuContext, bufferType, geomInstSlots);
        m_lightGeomInstDist.initialize(g_gpuEnv.cuContext, bufferType, nullptr, geomInstSlots.size());
        m_dirLightGeomInstDist.initialize(g_gpuEnv.cuContext, bufferType, nullptr, geomInstSlots.size());
        m_dirty = true;
        m_gasIsDirty = true;
    }

    const BoundingBox3D &getAABB() const {
        return m_aabb;
    }

    optixu::GeometryAccelerationStructure getOptixGAS() const {
        return m_optixGas;
    }

    void buildAS(CUstream stream) {
        if (!m_gasIsDirty)
            return;

        OptixAccelBufferSizes sizes;
        m_optixGas.prepareForBuild(&sizes);

        if (!m_optixAsMem.isInitialized())
            m_optixAsMem.initialize(g_gpuEnv.cuContext, bufferType, sizes.outputSizeInBytes, 1);
        else if (m_optixAsMem.sizeInBytes() < sizes.outputSizeInBytes)
            m_optixAsMem.resize(sizes.outputSizeInBytes, 1);

        if (!m_optixAsScratchMem.isInitialized())
            m_optixAsScratchMem.initialize(g_gpuEnv.cuContext, bufferType, sizes.tempSizeInBytes, 1);
        else if (m_optixAsScratchMem.sizeInBytes() < sizes.tempSizeInBytes)
            m_optixAsScratchMem.resize(sizes.tempSizeInBytes, 1);

        m_optixGas.rebuild(stream, m_optixAsMem, m_optixAsScratchMem);

        m_gasIsDirty = false;
    }

    uint32_t getSlot() const {
        return m_slot;
    }

    bool setUpDeviceData(shared::GeometryGroup* deviceData) {
        if (!m_dirty)
            return false;

        *deviceData = {};
        deviceData->geomInstSlots = m_geomInstSlotBuffer.getDevicePointer();
        m_lightGeomInstDist.getDeviceType(&deviceData->lightGeomInstDist);
        m_dirLightGeomInstDist.getDeviceType(&deviceData->dirLightGeomInstDist);
        deviceData->aabb = m_aabb;

        m_dirty = false;
        return true;
    }

    void setUpLightDistribution(
        CUstream stream,
        const shared::GeometryGroup* deviceData,
        const shared::SurfaceMaterial* surfMatBuffer, const shared::GeometryInstance* geomInstBuffer) const {
        if (!m_lightGeomInstDist.isInitialized() && !m_dirLightGeomInstDist.isInitialized())
            return;
        uint32_t numGeomInsts = m_geomInstSlotBuffer.numElements();
        g_gpuEnv.computeLightProbs.computeGeomInstProbBuffer.launchWithThreadDim(
            stream, cudau::dim3(numGeomInsts),
            deviceData, numGeomInsts,
            surfMatBuffer,
            geomInstBuffer);
    }
    void scanLightDistribution(
        CUstream stream,
        const shared::GeometryGroup* deviceData) const {
        if (!m_lightGeomInstDist.isInitialized() && !m_dirLightGeomInstDist.isInitialized())
            return;
        uint32_t numGeomInsts = m_geomInstSlotBuffer.numElements();
        const cudau::Buffer &scanScratchMem = g_scene.getScanScratchMemory();
        size_t scratchMemSize = scanScratchMem.sizeInBytes();
        CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
            scanScratchMem.getDevicePointer(), scratchMemSize,
            m_lightGeomInstDist.weightsOnDevice(),
            m_lightGeomInstDist.cdfOnDevice(),
            numGeomInsts, stream));
        CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
            scanScratchMem.getDevicePointer(), scratchMemSize,
            m_dirLightGeomInstDist.weightsOnDevice(),
            m_dirLightGeomInstDist.cdfOnDevice(),
            numGeomInsts, stream));
    }
    void finalizeLightDistribution(
        CUstream stream,
        const shared::GeometryGroup* deviceData) const {
        if (!m_lightGeomInstDist.isInitialized() && !m_dirLightGeomInstDist.isInitialized())
            return;
        g_gpuEnv.computeLightProbs.finalizeDiscreteDistribution1D.launchWithThreadDim(
            stream, cudau::dim3(1),
            &deviceData->lightGeomInstDist);
        g_gpuEnv.computeLightProbs.finalizeDiscreteDistribution1D.launchWithThreadDim(
            stream, cudau::dim3(1),
            &deviceData->dirLightGeomInstDist);
    }
};



struct KeyInstanceState {
    float timePoint;
    float scale;
    Quaternion orientation;
    Point3D position;
};



class Instance {
    optixu::Instance m_optixInst;
    uint32_t m_slot;
    std::vector<KeyInstanceState> m_keyStates;
    Ref<GeometryGroup> m_geomGroup;
    Matrix4x4 m_staticTransform;
    float m_lastTimePoint;
    uint32_t m_hasCyclicAnim : 1;
    uint32_t m_dirty : 1;

    void interpolateStates(
        float timePoint,
        Point3D* position, Quaternion* orientation, float* scale) const {
        if (m_hasCyclicAnim) {
            float timeBegin = m_keyStates.front().timePoint;
            float timeEnd = m_keyStates.back().timePoint;
            timePoint = std::fmodf(timePoint - timeBegin, timeEnd - timeBegin) + timeBegin;
        }
        uint32_t numStates = static_cast<uint32_t>(m_keyStates.size());
        int idx = 0;
        for (int d = nextPowerOf2(numStates) >> 1; d >= 1; d >>= 1) {
            if (idx + d >= numStates)
                continue;
            const KeyInstanceState &keyState = m_keyStates[idx + d];
            if (keyState.timePoint <= timePoint)
                idx += d;
        }
        //KeyInstanceState states[2];
        //states[0] = m_keyStates[idx];
        //states[1] = m_keyStates[std::min(static_cast<uint32_t>(idx) + 1, numStates - 1)];
        //float t = safeDivide(timePoint - states[0].timePoint, states[1].timePoint - states[0].timePoint);
        //t = clamp(t, 0.0f, 1.0f);
        //*scale = lerp(states[0].scale, states[1].scale, t);
        //*orientation = slerp(states[0].orientation, states[1].orientation, t);
        //*position = lerp(states[0].position, states[1].position, t);

        KeyInstanceState states[4];
        states[1] = m_keyStates[idx];
        states[2] = m_keyStates[
            m_hasCyclicAnim ?
                (idx + 1) % numStates :
                std::min(idx + 1, static_cast<int32_t>(numStates - 1))];
        if (idx - 1 >= 0) {
            states[0] = m_keyStates[idx - 1];
        }
        else {
            // states[0] = states[1] - (states[2] - states[1])
            states[0].position = Point3D(2 * states[1].position - states[2].position);
            states[0].orientation = slerp(states[2].orientation, states[1].orientation, 2.0f, true);
            states[0].scale = 2 * states[1].scale - states[2].scale;
        }

        if (idx + 2 < numStates) {
            states[3] = m_keyStates[idx + 2];
        }
        else {
            // states[3] = states[2] + (states[2] - states[1])
            states[3].position = Point3D(2 * states[2].position - states[1].position);
            states[3].orientation = slerp(states[1].orientation, states[2].orientation, 2.0f, true);
            states[3].scale = 2 * states[2].scale - states[1].scale;
        }

        float t = safeDivide(timePoint - states[1].timePoint, states[2].timePoint - states[1].timePoint);
        t = clamp(t, 0.0f, 1.0f);
        Vector3D posA4(states[1].position);
        Vector3D posA3(0.5f * (states[2].position - states[0].position));
        Vector3D posA1(
            0.5f * (states[3].position - states[1].position)
            - 2 * Vector3D(states[2].position)
            + posA3 + 2 * posA4);
        Vector3D posA2(
            3 * states[2].position
            - 0.5f * (states[3].position - states[1].position)
            - 2 * posA3 - 3 * posA4);
        *position = Point3D(
            posA1 * pow3(t) + posA2 * pow2(t) + posA3 * t + posA4);
        *scale = lerp(states[1].scale, states[2].scale, t);
        *orientation = slerp(states[1].orientation, states[2].orientation, t, true);
    }

public:
    Instance() :
        m_slot(0xFFFFFFFF), m_lastTimePoint(NAN), m_hasCyclicAnim(false), m_dirty(false) {}

    void associateScene(uint32_t slot, optixu::Instance optixInst) {
        m_slot = slot;
        m_optixInst = optixInst;
        m_dirty = true;
    }

    void setGeometryGroup(const Ref<GeometryGroup> &geomGroup, const Matrix4x4 &staticTransform) {
        m_optixInst.setChild(geomGroup->getOptixGAS());
        m_geomGroup = geomGroup;
        m_staticTransform = staticTransform;
        m_dirty = true;
    }

    void setKeyStates(const std::vector<KeyInstanceState> &states, bool hasCyclicAnim) {
        m_keyStates = states;
        m_hasCyclicAnim = hasCyclicAnim;
        m_dirty = true;
    }

    BoundingBox3D computeAABB(float timePoint) {
        Point3D position;
        Quaternion orientation;
        float scale;
        interpolateStates(timePoint, &position, &orientation, &scale);
        Matrix4x4 transform =
            translate4x4(position) * orientation.toMatrix4x4() * scale4x4(scale) *
            m_staticTransform;
        BoundingBox3D geomGroupAABB = m_geomGroup->getAABB();
        BoundingBox3D ret = transform * geomGroupAABB;
        return ret;
    }

    bool setUpDeviceData(shared::Instance* deviceData, float timePoint) {
        *deviceData = {};
        Point3D position;
        Quaternion orientation;
        float scale;
        interpolateStates(timePoint, &position, &orientation, &scale);
        Matrix4x4 transform =
            translate4x4(position) * orientation.toMatrix4x4() * scale4x4(scale) *
            m_staticTransform;

        deviceData->geomGroupSlot = m_geomGroup->getSlot();
        deviceData->transform = transform;
        deviceData->prevTransform = transform; // TODO?
        deviceData->uniformScale = scale;

        const float xfmArray[12] = {
            transform.m00, transform.m01, transform.m02, transform.m03,
            transform.m10, transform.m11, transform.m12, transform.m13,
            transform.m20, transform.m21, transform.m22, transform.m23,
        };
        m_optixInst.setTransform(xfmArray);

        return true;
    }
};



struct KeyCameraState {
    float timePoint;
    Point3D position;
    Point3D positionLookAt;
    Vector3D up;
    float fovY;
};



class Camera {
    std::vector<KeyCameraState> m_keyStates;

public:
    Camera(const std::vector<KeyCameraState> &keyStates) :
        m_keyStates(keyStates) {}

    void setUpDeviceData(shared::PerspectiveCamera* deviceData, float timePoint) {
        float timeBegin = m_keyStates.front().timePoint;
        float timeEnd = m_keyStates.back().timePoint;
        timePoint = clamp(timePoint, timeBegin, timeEnd);
        uint32_t numStates = static_cast<uint32_t>(m_keyStates.size());
        int idx = 0;
        for (int d = nextPowerOf2(numStates) >> 1; d >= 1; d >>= 1) {
            if (idx + d >= numStates)
                continue;
            const KeyCameraState &keyState = m_keyStates[idx + d];
            if (keyState.timePoint <= timePoint)
                idx += d;
        }

        KeyCameraState states[4];
        states[1] = m_keyStates[idx];
        states[2] = m_keyStates[std::min(idx + 1, static_cast<int32_t>(numStates - 1))];
        if (idx - 1 >= 0) {
            states[0] = m_keyStates[idx - 1];
        }
        else {
            // states[0] = states[1] - (states[2] - states[1])
            states[0].position = Point3D(2 * states[1].position - states[2].position);
        }

        if (idx + 2 < numStates) {
            states[3] = m_keyStates[idx + 2];
        }
        else {
            // states[3] = states[2] + (states[2] - states[1])
            states[3].position = Point3D(2 * states[2].position - states[1].position);
        }

        float t = safeDivide(timePoint - states[1].timePoint, states[2].timePoint - states[1].timePoint);
        t = clamp(t, 0.0f, 1.0f);
        Vector3D posA4(states[1].position);
        Vector3D posA3(0.5f * (states[2].position - states[0].position));
        Vector3D posA1(
            0.5f * (states[3].position - states[1].position)
            - 2 * Vector3D(states[2].position)
            + posA3 + 2 * posA4);
        Vector3D posA2(
            3 * states[2].position
            - 0.5f * (states[3].position - states[1].position)
            - 2 * posA3 - 3 * posA4);
        Point3D position = Point3D(
            posA1 * pow3(t) + posA2 * pow2(t) + posA3 * t + posA4);
        Quaternion ori0 = qLookAt(states[1].positionLookAt - states[1].position, states[1].up);
        Quaternion ori1 = qLookAt(states[2].positionLookAt - states[2].position, states[2].up);
        Quaternion orientation = slerp(ori0, ori1, t, true);
        float fovY = lerp(states[1].fovY, states[2].fovY, t);

        deviceData->position = position;
        deviceData->orientation = conjugate(orientation)/* * qRotateY(pi_v<float>)*/;
        deviceData->fovY = fovY * pi_v<float> / 180;
    }
};



static void computeImportanceMap(
    CUstream stream,
    const Ref<Texture2D> &image, const Ref<Texture2D> &importanceMap, CUdeviceptr impMapAddr) {
    CUsurfObject deviceTexture = image->getSurfaceObject(0);
    uint32_t width = image->getWidth();
    uint32_t height = image->getHeight();
    uint32_t dimX = nextPowerOf2(width);
    uint32_t dimY = std::max(nextPowerOf2(height), dimX >> 1);
    uint2 curDims = uint2(dimX, dimY);
    uint32_t numMipLevels = nextPowOf2Exponent(curDims.x) + 1;

    constexpr bool debugMap = false;

    g_gpuEnv.computeLightProbs.computeFirstMipOfEnvIBLImportanceMap.launchWithThreadDim(
        stream, cudau::dim3(curDims.x, curDims.y),
        deviceTexture, uint2(width, height),
        impMapAddr, importanceMap->getSurfaceObject(0));
    if constexpr (debugMap) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        Ref<cudau::Array> cudaArray = importanceMap->getCudaArray();
        float* data = cudaArray->map<float>();
        auto image = new float4[curDims.x * curDims.y];
        CompensatedSum<float> sum(0);
        for (int y = 0; y < curDims.y; ++y) {
            for (int x = 0; x < curDims.x; ++x) {
                uint32_t idx = y * curDims.x + x;
                float value = data[idx];
                image[idx] = float4(value, value, value, 1.0f);
                sum += value;
            }
        }
        SDRImageSaverConfig config = {};
        config.alphaForOverride = 1.0f;
        config.applyToneMap = true;
        config.apply_sRGB_gammaCorrection = false;
        config.brightnessScale = 1.0f;
        config.flipY = false;
        saveImage("vis00.png", curDims.x, curDims.y, 4, reinterpret_cast<float*>(image), config);
        delete[] image;
        cudaArray->unmap();
    }

    for (uint32_t mipLevel = 1; mipLevel < numMipLevels; ++mipLevel) {
        curDims = (curDims + uint2(1, 1)) / 2;
        g_gpuEnv.computeLightProbs.computeMipOfImportanceMap.launchWithThreadDim(
            stream, cudau::dim3(curDims.x, curDims.y),
            impMapAddr, mipLevel,
            importanceMap->getSurfaceObject(mipLevel - 1),
            importanceMap->getSurfaceObject(mipLevel));
        if constexpr (debugMap) {
            CUDADRV_CHECK(cuStreamSynchronize(stream));
            Ref<cudau::Array> cudaArray = importanceMap->getCudaArray();
            float* data = cudaArray->map<float>(mipLevel);
            auto image = new float4[curDims.x * curDims.y];
            for (int y = 0; y < curDims.y; ++y) {
                for (int x = 0; x < curDims.x; ++x) {
                    uint32_t idx = y * curDims.x + x;
                    float value = data[idx];
                    image[idx] = float4(value, value, value, 1.0f);
                }
            }
            SDRImageSaverConfig config = {};
            config.alphaForOverride = 1.0f;
            config.applyToneMap = true;
            config.apply_sRGB_gammaCorrection = false;
            config.brightnessScale = 1.0f;
            config.flipY = false;
            char filename[256];
            sprintf_s(filename, "vis%02u.png", mipLevel);
            saveImage(filename, curDims.x, curDims.y, 4, reinterpret_cast<float*>(image), config);
            delete[] image;
            cudaArray->unmap(mipLevel);

            if (mipLevel == numMipLevels - 1) {
                shared::HierarchicalImportanceMap impMap;
                CUDADRV_CHECK(cuMemcpyDtoH(&impMap, impMapAddr, sizeof(impMap)));
                printf("");
            }
        }
    }

    constexpr bool testMap = false;
    if constexpr (testMap) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        constexpr uint32_t numThreads = 1 << 24;
        cudau::TypedBuffer<shared::PCG32RNG> rngs;
        cudau::TypedBuffer<uint32_t> histogram;
        rngs.initialize(g_gpuEnv.cuContext, bufferType, numThreads);
        {
            std::mt19937_64 mt(7124223134912111);
            shared::PCG32RNG* rngsOnHost = rngs.map();
            for (int i = 0; i < numThreads; ++i)
                rngsOnHost[i].setState(mt());
            rngs.unmap();
        }
        histogram.initialize(
            g_gpuEnv.cuContext, bufferType,
            width * height, 0);

        uint2 imageSize(width, height);
        g_gpuEnv.computeLightProbs.testImportanceMap.launchWithThreadDim(
            stream, cudau::dim3(numThreads),
            imageSize, impMapAddr, rngs.getDevicePointer(), histogram.getDevicePointer());

        CUDADRV_CHECK(cuStreamSynchronize(stream));
        const uint32_t* histogramOnHost = histogram.map();
        auto image = new float4[width * height];
        uint32_t maxHistValue = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint32_t idx = y * width + x;
                maxHistValue = std::max(histogramOnHost[idx], maxHistValue);
            }
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint32_t idx = y * width + x;
                float value = histogramOnHost[idx] / static_cast<float>(maxHistValue);
                image[idx] = float4(value, value, value, 1.0f);
            }
        }
        SDRImageSaverConfig config = {};
        config.alphaForOverride = 1.0f;
        config.applyToneMap = false;
        config.apply_sRGB_gammaCorrection = false;
        config.brightnessScale = 1.0f;
        config.flipY = false;
        saveImage("histogram.png", width, height, 4, reinterpret_cast<float*>(image), config);
        delete[] image;
        histogram.unmap();
    }
}



struct KeyEnvironmentState {
    float timePoint;
    float coeff;
    float rotation;

    virtual ~KeyEnvironmentState() {}
};



class Environment {
protected:
    std::vector<Ref<KeyEnvironmentState>> m_keyStates;
    uint32_t m_dirty : 1;

    void getControlPoints(
        float timePoint,
        std::vector<Ref<KeyEnvironmentState>>* controlPoints,
        float* t) {
        uint32_t numStates = static_cast<uint32_t>(m_keyStates.size());
        int idx = 0;
        for (int d = nextPowerOf2(numStates) >> 1; d >= 1; d >>= 1) {
            if (idx + d >= numStates)
                continue;
            const Ref<KeyEnvironmentState> &keyState = m_keyStates[idx + d];
            if (keyState->timePoint <= timePoint)
                idx += d;
        }
        controlPoints->push_back(m_keyStates[idx]);
        controlPoints->push_back(m_keyStates[std::min(static_cast<uint32_t>(idx) + 1, numStates - 1)]);
        *t = safeDivide(timePoint - controlPoints->at(0)->timePoint,
                        controlPoints->at(1)->timePoint - controlPoints->at(0)->timePoint);
        *t = clamp(*t, 0.0f, 1.0f);
    }
    void interpolateStates(
        const std::vector<Ref<KeyEnvironmentState>> &states, float t,
        float* coeff, float* rotation) const {
        *coeff = lerp(states[0]->coeff, states[1]->coeff, t);
        *rotation = lerp(states[0]->rotation, states[1]->rotation, t);
    }

public:
    Environment() :
        m_dirty(false) {}
    virtual ~Environment() {}

    void setKeyStates(const std::vector<Ref<KeyEnvironmentState>> &states) {
        m_keyStates = states;
        m_dirty = true;
    }

    virtual void setUpDeviceData(shared::EnvironmentalLight* deviceData, float timePoint) = 0;

    virtual void computeDistribution(
        CUstream stream, CUdeviceptr envLightAddr, float /*timePoint*/) = 0;
};



class ImageBasedEnvironment : public Environment {
    Ref<Texture2D> m_image;
    Ref<Texture2D> m_importanceMap;

public:
    ImageBasedEnvironment(const Ref<Texture2D> &image) :
        m_image(image) {
        uint32_t width = m_image->getWidth();
        uint32_t height = m_image->getHeight();
        uint32_t dimX = nextPowerOf2(width);
        uint32_t dimY = std::max(nextPowerOf2(height), dimX >> 1);
        uint32_t numMipLevels = nextPowOf2Exponent(dimX) + 1;
        auto mapArray = std::make_shared<cudau::Array>();
        mapArray->initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::Float32, 1,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            dimX, dimY, numMipLevels);
        m_importanceMap = std::make_shared<Texture2D>(mapArray);
        m_importanceMap->setReadMode(cudau::TextureReadMode::ElementType);

        m_dirty = true;
    }

    void setUpDeviceData(shared::EnvironmentalLight* deviceData, float timePoint) override {
        std::vector<Ref<KeyEnvironmentState>> controlPoints;
        float t;
        getControlPoints(timePoint, &controlPoints, &t);

        float coeff;
        float rotation;
        interpolateStates(controlPoints, t, &coeff, &rotation);

        auto &body = *reinterpret_cast<shared::ImageBasedEnvironmentalLight*>(deviceData->body);
        body.texObj = m_image->getDeviceTexture();
        body.importanceMap.setTexObject(
            m_importanceMap->getDeviceTexture(),
            uint2(m_importanceMap->getWidth(), m_importanceMap->getHeight()));
        body.imageWidth = m_image->getWidth();
        body.imageHeight = m_image->getHeight();
        deviceData->envLightSample = CallableProgram_ImageBasedEnvironmentalLight_sample;
        deviceData->envLightEvaluate = CallableProgram_ImageBasedEnvironmentalLight_evaluate;
        deviceData->powerCoeff = coeff;
        deviceData->rotation = rotation * pi_v<float> / 180;
    }

    void computeDistribution(
        CUstream stream, CUdeviceptr envLightAddr, float /*timePoint*/) override {
        CUdeviceptr impMapAddr = envLightAddr + offsetof(shared::ImageBasedEnvironmentalLight, importanceMap);
        computeImportanceMap(stream, m_image, m_importanceMap, impMapAddr);
    }
};



struct KeyAnalyticSkyEnvironmentState : public KeyEnvironmentState {
    Vector3D solarDirection;
    float turbidity;
    //RGBSpectrum groundAlbedo;
};



class AnalyticSkyEnvironment : public Environment {
    Ref<Texture2D> m_image;
    Ref<Texture2D> m_importanceMap;

    void interpolateStates(
        const std::vector<Ref<KeyEnvironmentState>> &baseStates, float t,
        Vector3D* solarDirection, float* turbidity, RGBSpectrum* /*groundAlbedo*/) const {
        std::vector<Ref<KeyAnalyticSkyEnvironmentState>> states(baseStates.size());
        for (int i = 0; i < baseStates.size(); ++i)
            states[i] = std::dynamic_pointer_cast<KeyAnalyticSkyEnvironmentState>(baseStates[i]);

        float dotDir = clamp(dot(states[0]->solarDirection, states[1]->solarDirection), -1.0f, 1.0f);
        float angle = std::acos(dotDir);
        Vector3D axis;
        if (std::fabs(dotDir) < 0.999f) {
            axis = cross(states[0]->solarDirection, states[1]->solarDirection).normalize();
        }
        else {
            Vector3D b;
            states[0]->solarDirection.makeCoordinateSystem(&axis, &b);
        }

        *solarDirection = rotate3x3(angle * t, axis) * states[0]->solarDirection;
        *turbidity = lerp(states[0]->turbidity, states[1]->turbidity, t);
        //*groundAlbedo = lerp(states[0]->groundAlbedo, states[1]->groundAlbedo, t);
    }

public:
    AnalyticSkyEnvironment(uint2 imageSize) {
        auto imageArray = std::make_shared<cudau::Array>();
        imageArray->initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            imageSize.x, imageSize.y, 0);
        m_image = std::make_shared<Texture2D>(imageArray);
        m_image->setReadMode(cudau::TextureReadMode::ElementType);
        // JP: EnvMapのサンプリング時に、低いPDF値のピクセルでも隣に超高輝度ピクセルがあった場合、
        //     バイリニアフィルタリングで高い値をとってしまい分散が非常に大きくなるので、
        //     とりあえずバイリニアフィルタリングを切っておく。
        //     ちゃんとするにはPDF側をフィルタリングする必要がある。
        m_image->setXyFilterMode(cudau::TextureFilterMode::Point);

        uint32_t dimX = nextPowerOf2(imageSize.x);
        uint32_t dimY = std::max(nextPowerOf2(imageSize.y), dimX >> 1);
        uint32_t numMipLevels = nextPowOf2Exponent(dimX) + 1;
        auto mapArray = std::make_shared<cudau::Array>();
        mapArray->initialize2D(
            g_gpuEnv.cuContext, cudau::ArrayElementType::Float32, 1,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            dimX, dimY, numMipLevels);
        m_importanceMap = std::make_shared<Texture2D>(mapArray);
        m_importanceMap->setReadMode(cudau::TextureReadMode::ElementType);

        m_dirty = true;
    }

    void setUpDeviceData(shared::EnvironmentalLight* deviceData, float timePoint) override {
        std::vector<Ref<KeyEnvironmentState>> controlPoints;
        float t;
        getControlPoints(timePoint, &controlPoints, &t);

        float coeff;
        float rotation;
        Environment::interpolateStates(controlPoints, t, &coeff, &rotation);

        auto &body = *reinterpret_cast<shared::ImageBasedEnvironmentalLight*>(deviceData->body);
        body.texObj = m_image->getDeviceTexture();
        body.importanceMap.setTexObject(
            m_importanceMap->getDeviceTexture(),
            uint2(m_importanceMap->getWidth(), m_importanceMap->getHeight()));
        body.imageWidth = m_image->getWidth();
        body.imageHeight = m_image->getHeight();
        deviceData->envLightSample = CallableProgram_ImageBasedEnvironmentalLight_sample;
        deviceData->envLightEvaluate = CallableProgram_ImageBasedEnvironmentalLight_evaluate;
        deviceData->powerCoeff = coeff;
        deviceData->rotation = rotation * pi_v<float> / 180;
    }

    void computeDistribution(
        CUstream stream, CUdeviceptr envLightAddr, float timePoint) override;
};



struct ActiveCameraInfo {
    float timePoint;
    std::string name;
};

struct RenderConfigs {
    uint32_t imageWidth;
    uint32_t imageHeight;
    float timeBegin;
    float timeEnd;
    uint32_t fps;

    std::map<std::string, Ref<Camera>> cameras;
    std::vector<ActiveCameraInfo> activeCameraInfos;

    std::vector<Ref<SurfaceMaterial>> surfaceMaterials;
    std::vector<Ref<Geometry>> geometries;
    std::vector<Ref<GeometryGroup>> geometryGroups;
    std::vector<Ref<Instance>> instances;
};

void loadScene(const std::filesystem::path &filePath, RenderConfigs* renderConfigs);

} // namespace rtc10