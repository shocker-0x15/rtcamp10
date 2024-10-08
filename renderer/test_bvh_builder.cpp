#include "bvh_builder.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

using namespace rtc10;

#define USE_VDB 0

struct TriangleMesh {
    std::vector<shared::Vertex> vertices;
    std::vector<shared::Triangle> triangles;
};

struct FlattenedNode {
    Matrix4x4 transform;
    std::vector<uint32_t> meshIndices;
};

static void computeFlattenedNodes(
    const aiScene* scene, const Matrix4x4 &parentXfm, const aiNode* curNode,
    std::vector<FlattenedNode> &flattenedNodes) {
    aiMatrix4x4 curAiXfm = curNode->mTransformation;
    Matrix4x4 curXfm = Matrix4x4(
        Vector4D(curAiXfm.a1, curAiXfm.a2, curAiXfm.a3, curAiXfm.a4),
        Vector4D(curAiXfm.b1, curAiXfm.b2, curAiXfm.b3, curAiXfm.b4),
        Vector4D(curAiXfm.c1, curAiXfm.c2, curAiXfm.c3, curAiXfm.c4),
        Vector4D(curAiXfm.d1, curAiXfm.d2, curAiXfm.d3, curAiXfm.d4));
    FlattenedNode flattenedNode;
    flattenedNode.transform = parentXfm * transpose(curXfm);
    flattenedNode.meshIndices.resize(curNode->mNumMeshes);
    if (curNode->mNumMeshes > 0) {
        std::copy_n(curNode->mMeshes, curNode->mNumMeshes, flattenedNode.meshIndices.data());
        flattenedNodes.push_back(flattenedNode);
    }

    for (uint32_t cIdx = 0; cIdx < curNode->mNumChildren; ++cIdx)
        computeFlattenedNodes(scene, flattenedNode.transform, curNode->mChildren[cIdx], flattenedNodes);
}

static void calcTriangleVertices(
    const bvh::Geometry &geom, const uint32_t primIdx,
    Point3D* const pA, Point3D* const pB, Point3D* const pC) {
    uint32_t tri[3];
    const auto triAddr = reinterpret_cast<uintptr_t>(geom.triangles) + geom.triangleStride * primIdx;
    if (geom.triangleFormat == bvh::TriangleFormat::UI32x3) {
        tri[0] = reinterpret_cast<const uint32_t*>(triAddr)[0];
        tri[1] = reinterpret_cast<const uint32_t*>(triAddr)[1];
        tri[2] = reinterpret_cast<const uint32_t*>(triAddr)[2];
    }
    else {
        Assert(geom.triangleFormat == bvh::TriangleFormat::UI16x3, "Invalid triangle format.");
        tri[0] = reinterpret_cast<const uint16_t*>(triAddr)[0];
        tri[1] = reinterpret_cast<const uint16_t*>(triAddr)[1];
        tri[2] = reinterpret_cast<const uint16_t*>(triAddr)[2];
    }

    Point3D ps[3];
    const auto vertBaseAddr = reinterpret_cast<uintptr_t>(geom.vertices);
    Assert(geom.vertexFormat == bvh::VertexFormat::Fp32x3, "Invalid vertex format.");
    ps[0] = *reinterpret_cast<const Point3D*>(vertBaseAddr + geom.vertexStride * tri[0]);
    ps[1] = *reinterpret_cast<const Point3D*>(vertBaseAddr + geom.vertexStride * tri[1]);
    ps[2] = *reinterpret_cast<const Point3D*>(vertBaseAddr + geom.vertexStride * tri[2]);

    *pA = geom.preTransform * ps[0];
    *pB = geom.preTransform * ps[1];
    *pC = geom.preTransform * ps[2];
}

void testBvhBuilder() {
    struct TestScene {
        std::filesystem::path filePath;
        Matrix4x4 transform;
        Matrix4x4 cameraTransform;
    };

    std::map<std::string, TestScene> scenes = {
        {
            "bunny",
            {
                R"(E:/assets/McguireCGArchive/bunny/bunny.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(-0.299932f, 1.73252f, 2.4276f) *
                rotate3DY_4x4(178.68f * pi_v<float> / 180) *
                rotate3DX_4x4(22.2f * pi_v<float> / 180)
            }
        },
        {
            "dragon",
            {
                R"(E:/assets/McguireCGArchive/dragon/dragon.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(-1.08556f, 0.450182f, -0.473484f) *
                rotate3DY_4x4(68.0f * pi_v<float> / 180) *
                rotate3DX_4x4(19.0f * pi_v<float> / 180)
            }
        },
        {
            "buddha",
            {
                R"(E:/assets/McguireCGArchive/buddha/buddha.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(-0.004269f, 0.342561f, -1.34414f) *
                rotate3DY_4x4(0.0f * pi_v<float> / 180) *
                rotate3DX_4x4(12.5f * pi_v<float> / 180)
            }
        },
        {
            "white_oak",
            {
                R"(E:/assets/McguireCGArchive/white_oak/white_oak.obj)",
                scale3D_4x4(0.01f),
                translate3D_4x4(2.86811f, 4.87556f, 10.4772f) *
                rotate3DY_4x4(195.5f * pi_v<float> / 180) *
                rotate3DX_4x4(8.9f * pi_v<float> / 180)
            }
        },
        {
            "conference",
            {
                R"(E:/assets/McguireCGArchive/conference/conference.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(1579.2f, 493.793f, 321.98f) *
                rotate3DY_4x4(-120 * pi_v<float> / 180) *
                rotate3DX_4x4(20 * pi_v<float> / 180)
            }
        },
        {
            "breakfast_room",
            {
                R"(E:/assets/McguireCGArchive/breakfast_room/breakfast_room.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(4.37691f, 1.8413f, 6.35917f) *
                rotate3DY_4x4(210 * pi_v<float> / 180) *
                rotate3DX_4x4(2.8f * pi_v<float> / 180)
            }
        },
        {
            "salle_de_bain",
            {
                R"(E:/assets/McguireCGArchive/salle_de_bain/salle_de_bain.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(2.56843f, 15.9865f, 45.3711f) *
                rotate3DY_4x4(191 * pi_v<float> / 180) *
                rotate3DX_4x4(6.2f * pi_v<float> / 180)
            }
        },
        {
            "crytek_sponza",
            {
                R"(E:/assets/McguireCGArchive/sponza/sponza.obj)",
                scale3D_4x4(0.01f),
                translate3D_4x4(10.0f, 2.0f, -0.5f) *
                rotate3DY_4x4(-pi_v<float> / 2)
            }
        },
        {
            "sibenik",
            {
                R"(E:/assets/McguireCGArchive/sibenik/sibenik.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(-15.0f, -3.0f, 0.0f) *
                rotate3DY_4x4(pi_v<float> / 2) *
                rotate3DX_4x4(20 * pi_v<float> / 180)
            }
        },
        {
            "hairball",
            {
                R"(E:/assets/McguireCGArchive/hairball/hairball.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(0.0f, 0.0f, 13.0f) *
                rotate3DY_4x4(pi_v<float>)
            }
        },
        {
            "rungholt",
            {
                R"(E:/assets/McguireCGArchive/rungholt/rungholt.obj)",
                scale3D_4x4(0.1f),
                translate3D_4x4(36.1611f, 5.56238f, -20.4327f) *
                rotate3DY_4x4(-53.0f * pi_v<float> / 180) *
                rotate3DX_4x4(14.2f * pi_v<float> / 180)
            }
        },
        {
            "san_miguel",
            {
                R"(E:/assets/McguireCGArchive/San_Miguel/san-miguel.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(6.2928f, 3.05034f, 7.49142f) *
                rotate3DY_4x4(125.8f * pi_v<float> / 180) *
                rotate3DX_4x4(9.3f * pi_v<float> / 180)
            }
        },
        {
            "powerplant",
            {
                R"(E:/assets/McguireCGArchive/powerplant/powerplant.obj)",
                scale3D_4x4(0.0001f),
                translate3D_4x4(-16.5697f, 5.66694f, 14.8665f) *
                rotate3DY_4x4(125.2f * pi_v<float> / 180) *
                rotate3DX_4x4(10.5f * pi_v<float> / 180)
            }
        },
        {
            "box",
            {
                R"(E:/assets/box/box.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(3.0f, 3.0f, 3.0f) *
                rotate3DY_4x4(225 * pi_v<float> / 180) *
                rotate3DX_4x4(35.264f * pi_v<float> / 180)
            }
        },
        {
            "lowpoly_bunny",
            {
                R"(E:/assets/lowpoly_bunny/stanford_bunny_309_faces.obj)",
                scale3D_4x4(0.1f),
                translate3D_4x4(-4.60892f, 9.15149f, 11.7878f) *
                rotate3DY_4x4(161.4f * pi_v<float> / 180) *
                rotate3DX_4x4(23.6f * pi_v<float> / 180)
            }
        },
        {
            "teapot",
            {
                R"(E:/assets/McguireCGArchive/teapot/teapot.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(0.0f, 133.3f, 200.0f) *
                rotate3DY_4x4(180 * pi_v<float> / 180) *
                rotate3DX_4x4(25 * pi_v<float> / 180)
            }
        },
        {
            "one_tri",
            {
                R"(E:/assets/one_tri.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(0.0f, 0.0f, 3.0f) *
                rotate3DY_4x4(180 * pi_v<float> / 180)
            }
        },
        {
            "two_tris",
            {
                R"(E:/assets/two_tris.obj)",
                Matrix4x4::Identity(),
                translate3D_4x4(0.0f, 0.0f, 3.0f) *
                rotate3DY_4x4(180 * pi_v<float> / 180)
            }
        },
    };

    const TestScene &scene = scenes.at("lowpoly_bunny");
    //const TestScene &scene = scenes.at("conference");
    //const TestScene &scene = scenes.at("hairball");
    //const TestScene &scene = scenes.at("breakfast_room");
    //const TestScene &scene = scenes.at("powerplant");
    //const TestScene &scene = scenes.at("san_miguel");
    constexpr uint32_t maxNumIntersections = 128;
    constexpr uint32_t singleCamIdx = 0;
    constexpr bool visStats = false;

    constexpr uint32_t arity = 8;

    bvh::GeometryBVHBuildConfig config = {};
    config.splittingBudget = 0.3f;
    config.intNodeTravCost = 1.2f;
    config.primIntersectCost = 1.0f;
    config.minNumPrimsPerLeaf = 1;
    config.maxNumPrimsPerLeaf = 128;

    hpprintf("Reading: %s ... ", scene.filePath.string().c_str());
    fflush(stdout);
    Assimp::Importer importer;
    const aiScene* aiscene = importer.ReadFile(
        scene.filePath.string(),
        aiProcess_Triangulate |
        aiProcess_GenNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_FlipUVs);
    if (!aiscene) {
        hpprintf("Failed to load %s.\n", scene.filePath.string().c_str());
        return;
    }
    hpprintf("done.\n");

    std::vector<TriangleMesh> meshes;
    for (uint32_t meshIdx = 0; meshIdx < aiscene->mNumMeshes; ++meshIdx) {
        const aiMesh* aiMesh = aiscene->mMeshes[meshIdx];

        std::vector<shared::Vertex> vertices(aiMesh->mNumVertices);
        for (int vIdx = 0; vIdx < vertices.size(); ++vIdx) {
            const aiVector3D &aip = aiMesh->mVertices[vIdx];
            const aiVector3D &ain = aiMesh->mNormals[vIdx];
            aiVector3D aitc0dir;
            if (aiMesh->mTangents)
                aitc0dir = aiMesh->mTangents[vIdx];
            if (!aiMesh->mTangents || !std::isfinite(aitc0dir.x)) {
                const auto makeCoordinateSystem = []
                (const Normal3D &normal, Vector3D* tangent, Vector3D* bitangent) {
                    float sign = normal.z >= 0 ? 1.0f : -1.0f;
                    const float a = -1 / (sign + normal.z);
                    const float b = normal.x * normal.y * a;
                    *tangent = Vector3D(1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
                    *bitangent = Vector3D(b, sign + normal.y * normal.y * a, -normal.y);
                };
                Vector3D tangent, bitangent;
                makeCoordinateSystem(Normal3D(ain.x, ain.y, ain.z), &tangent, &bitangent);
                aitc0dir = aiVector3D(tangent.x, tangent.y, tangent.z);
            }
            const aiVector3D ait = aiMesh->mTextureCoords[0] ?
                aiMesh->mTextureCoords[0][vIdx] :
                aiVector3D(0.0f, 0.0f, 0.0f);

            shared::Vertex v;
            v.position = Point3D(aip.x, aip.y, aip.z);
            v.normal = normalize(Normal3D(ain.x, ain.y, ain.z));
            v.tangent = normalize(Vector3D(aitc0dir.x, aitc0dir.y, aitc0dir.z));
            v.texCoord = TexCoord2D(ait.x, ait.y);
            vertices[vIdx] = v;
        }

        std::vector<shared::Triangle> triangles(aiMesh->mNumFaces);
        for (int fIdx = 0; fIdx < triangles.size(); ++fIdx) {
            const aiFace &aif = aiMesh->mFaces[fIdx];
            Assert(aif.mNumIndices == 3, "Number of face vertices must be 3 here.");
            shared::Triangle tri;
            tri.indices[0] = aif.mIndices[0];
            tri.indices[1] = aif.mIndices[1];
            tri.indices[2] = aif.mIndices[2];
            triangles[fIdx] = tri;
        }

        TriangleMesh mesh;
        mesh.vertices = std::move(vertices);
        mesh.triangles = std::move(triangles);
        meshes.push_back(std::move(mesh));
    }

    std::vector<FlattenedNode> flattenedNodes;
    computeFlattenedNodes(aiscene, scene.transform, aiscene->mRootNode, flattenedNodes);

    std::vector<bvh::Geometry> bvhGeoms;
    uint32_t numGlobalPrimitives = 0;
    for (uint32_t i = 0; i < flattenedNodes.size(); ++i) {
        const FlattenedNode &fNode = flattenedNodes[i];
        for (uint32_t mIdx = 0; mIdx < fNode.meshIndices.size(); ++mIdx) {
            const TriangleMesh &triMesh = meshes[fNode.meshIndices[mIdx]];

            bvh::Geometry geom = {};
            geom.vertices = triMesh.vertices.data();
            geom.vertexStride = sizeof(triMesh.vertices[0]);
            geom.vertexFormat = bvh::VertexFormat::Fp32x3;
            geom.numVertices = static_cast<uint32_t>(triMesh.vertices.size());
            geom.triangles = triMesh.triangles.data();
            geom.triangleStride = sizeof(triMesh.triangles[0]);
            geom.triangleFormat = bvh::TriangleFormat::UI32x3;
            geom.numTriangles = static_cast<uint32_t>(triMesh.triangles.size());
            geom.preTransform = fNode.transform;
            bvhGeoms.push_back(geom);

            numGlobalPrimitives += geom.numTriangles;
        }
    }

    bvh::GeometryBVH<arity> bvh;
    bvh::buildGeometryBVH(
        bvhGeoms.data(), static_cast<uint32_t>(bvhGeoms.size()),
        config, &bvh);

    static bool enableVdbViz = false;
    if (enableVdbViz) {
        struct StackEntry {
            uint32_t nodeIndex;
            uint32_t depth;
        };

        struct NodeChildAddress {
            uint32_t nodeIndex;
            uint32_t slot;
        };

        constexpr uint32_t maxDepth = 12;
        std::vector<StackEntry> stack;
        std::vector<std::vector<NodeChildAddress>> triToNodeChildMap(numGlobalPrimitives);

#if USE_VDB
        vdb_frame();
        drawAxes(10.0f);
        setColor(1.0f, 1.0f, 1.0f);
#endif

        stack.push_back(StackEntry{ 0, 0 });
        while (!stack.empty()) {
            const StackEntry entry = stack.back();
            stack.pop_back();

            const shared::InternalNode_T<arity> &intNode = bvh.intNodes[entry.nodeIndex];
            for (int32_t slot = 0; slot < arity; ++slot) {
                if (!intNode.getChildIsValid(slot))
                    break;

                const bool isLeaf = ((intNode.internalMask >> slot) & 0b1) == 0;
                const uint32_t lowerMask = (1 << slot) - 1;
                if (isLeaf) {
#if USE_VDB
                    setColor(0.1f, 0.1f, 0.1f);
                    drawAabb(intNode.getChildAabb(slot));
#endif
                    uint32_t leafOffset = intNode.leafBaseIndex + intNode.getLeafOffset(slot);
                    uint32_t chainLength = 0;
                    while (true) {
                        //hpprintf("%3u\n", leafOffset);
                        const shared::PrimitiveReference &primRef = bvh.primRefs[leafOffset++];
                        const shared::TriangleStorage &triStorage = bvh.triStorages[primRef.storageIndex];
                        ++chainLength;
#if USE_VDB
                        setColor(1.0f, 1.0f, 1.0f);
                        drawWiredTriangle(triStorage.pA, triStorage.pB, triStorage.pC);
#endif
                        triToNodeChildMap[primRef.storageIndex].push_back(
                            NodeChildAddress{ entry.nodeIndex, static_cast<uint32_t>(slot) });
                        if (primRef.isLeafEnd)
                            break;
                    }
                }
                else {
#if USE_VDB
                    setColor(0.0f, 0.3f * (entry.depth + 1) / maxDepth, 0.0f);
                    drawAabb(intNode.getChildAabb(slot));
#endif
                    if (entry.depth < maxDepth) {
                        const uint32_t childIdx = intNode.intNodeChildBaseIndex + intNode.getInternalChildNumber(slot);
                        stack.push_back(StackEntry{ childIdx, entry.depth + 1 });
                    }
                    else {
                        printf("");
                    }
                }
            }
        }

        if (false) {
            // Triangle to Node Children
            for (uint32_t globalPrimIdx = 0; globalPrimIdx < numGlobalPrimitives; ++globalPrimIdx) {
                const std::vector<NodeChildAddress> &refs = triToNodeChildMap[globalPrimIdx];

#if USE_VDB
                vdb_frame();
                drawAxes(10.0f);
#endif

                const shared::TriangleStorage &triStorage = bvh.triStorages[globalPrimIdx];
#if USE_VDB
                setColor(1.0f, 1.0f, 1.0f);
                drawWiredTriangle(triStorage.pA, triStorage.pB, triStorage.pC);
#endif

                for (uint32_t refIdx = 0; refIdx < refs.size(); ++refIdx) {
                    const NodeChildAddress &ref = refs[refIdx];
                    const shared::InternalNode_T<arity> &intNode = bvh.intNodes[ref.nodeIndex];
#if USE_VDB
                    setColor(0.1f, 0.1f, 0.1f);
                    drawAabb(intNode.getChildAabb(ref.slot));
#endif
                }
                printf("");
            }

            printf("");
        }
    }

    static bool enableTraversalTest = true;
    if (enableTraversalTest) {
        constexpr uint32_t width = 1024;
        constexpr uint32_t height = 1024;
        const float aspect = static_cast<float>(width) / height;
        const float fovY = 45 * pi_v<float> / 180;

        for (uint32_t camIdx = 0; camIdx < 30; ++camIdx) {
            if (camIdx != singleCamIdx && singleCamIdx != -1)
                continue;
            const Matrix4x4 camXfm =
                rotate3DY_4x4<float>(static_cast<float>(camIdx) / 30 * 2 * pi_v<float>) *
                scene.cameraTransform;

            std::vector<float4> image(width * height);
            double sumMaxStackDepth = 0;
            int32_t maxMaxStackDepth = -1;
            double sumAvgStackAccessDepth = 0;
            float maxAvgStackAccessDepth = -INFINITY;
            constexpr int32_t fastStackDepthLimit = 12 - 1;
            uint64_t stackMemoryAccessAmount = 0;
            for (uint32_t ipy = 0; ipy < height; ++ipy) {
                for (uint32_t ipx = 0; ipx < width; ++ipx) {
                    const float px = ipx + 0.5f;
                    const float py = ipy + 0.5f;

                    const Vector3D rayDirInLocal(
                        aspect * tan(fovY * 0.5f) * (1 - 2 * px / width),
                        tan(fovY * 0.5f) * (1 - 2 * py / height),
                        1);
                    const Point3D rayOrg = camXfm * Point3D(0, 0, 0);
                    const Vector3D rayDir = camXfm * rayDirInLocal;
                    bvh::TraversalStatistics stats = {};
                    stats.fastStackDepthLimit = fastStackDepthLimit;
                    const shared::HitObject hitObj = bvh::traverse(
                        bvh, rayOrg, rayDir, 0.0f, 1e+10f, &stats/*,
                        ipx == 691 && ipy == 458*/);

                    RGBSpectrum color = RGBSpectrum::Zero();
                    if (visStats) {
                        const float t = static_cast<float>(
                            stc::min(stats.numAabbTests + stats.numTriTests, maxNumIntersections)) /
                            maxNumIntersections;
                        const RGBSpectrum Red(1, 0, 0);
                        const RGBSpectrum Green(0, 1, 0);
                        const RGBSpectrum Blue(0, 0, 1);
                        color = t < 0.5f ? lerp(Blue, Green, 2.0f * t) : lerp(Green, Red, 2.0f * t - 1.0);
                    }
                    else {
                        if (hitObj.isHit()) {
                            const bvh::Geometry &geom = bvhGeoms[hitObj.geomIndex];
                            Point3D pA, pB, pC;
                            calcTriangleVertices(geom, hitObj.primIndex, &pA, &pB, &pC);
                            const Vector3D geomNormal = normalize(cross(pB - pA, pC - pA));
                            color.r = 0.5f + 0.5f * geomNormal.x;
                            color.g = 0.5f + 0.5f * geomNormal.y;
                            color.b = 0.5f + 0.5f * geomNormal.z;
                        }
                    }

                    image[width * ipy + ipx] = float4(color.r, color.g, color.b, 1.0f);
                    sumMaxStackDepth += stats.maxStackDepth;
                    maxMaxStackDepth = std::max(stats.maxStackDepth, maxMaxStackDepth);
                    sumAvgStackAccessDepth += stats.avgStackAccessDepth;
                    maxAvgStackAccessDepth = std::max(stats.avgStackAccessDepth, maxAvgStackAccessDepth);
                    stackMemoryAccessAmount += stats.stackMemoryAccessAmount;
                }
            }
            hpprintf("Avg Stack Access Depth - Avg: %.3f\n", sumAvgStackAccessDepth / (width * height));
            hpprintf("                       - Max: %.3f\n", maxAvgStackAccessDepth);
            hpprintf("Max Stack Depth - Avg: %.3f\n", sumMaxStackDepth / (width * height));
            hpprintf("                - Max: %d\n", maxMaxStackDepth);
            hpprintf(
                "Stack Memory Access: %llu [B] (#FastStackEntry: %d)",
                stackMemoryAccessAmount, fastStackDepthLimit + 1);

            SDRImageSaverConfig imageSaveConfig = {};
            imageSaveConfig.applyToneMap = false;
            imageSaveConfig.apply_sRGB_gammaCorrection = false;
            imageSaveConfig.brightnessScale = 1.0f;
            imageSaveConfig.flipY = false;
            char filename[256];
            sprintf_s(filename, "trav_test_%03u.png", camIdx);
            saveImage(
                filename,
                width, height, image.data(),
                imageSaveConfig);
        }
    }

    printf("");
}