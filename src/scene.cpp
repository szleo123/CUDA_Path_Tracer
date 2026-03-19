#include "scene.h"

#include "mesh.h"
#include "bvh.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cfloat>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

namespace
{
std::filesystem::path resolveScenePath(
    const std::filesystem::path& scenePath,
    const std::string& assetPath)
{
    const std::filesystem::path candidate(assetPath);
    if (candidate.is_absolute())
    {
        return candidate;
    }
    return scenePath.parent_path() / candidate;
}

std::filesystem::path findSceneFile(const std::string& inputPath)
{
    const std::filesystem::path input(inputPath);
    if (input.is_absolute() && std::filesystem::exists(input))
    {
        return std::filesystem::weakly_canonical(input);
    }

    const std::filesystem::path cwdCandidate = std::filesystem::current_path() / input;
    if (std::filesystem::exists(cwdCandidate))
    {
        return std::filesystem::weakly_canonical(cwdCandidate);
    }

    const std::filesystem::path sceneName = input.filename();
    std::filesystem::path probe = std::filesystem::current_path();
    for (int i = 0; i < 8; ++i)
    {
        const std::filesystem::path directCandidate = probe / input;
        if (std::filesystem::exists(directCandidate))
        {
            return std::filesystem::weakly_canonical(directCandidate);
        }

        const std::filesystem::path scenesCandidate = probe / "scenes" / sceneName;
        if (std::filesystem::exists(scenesCandidate))
        {
            return std::filesystem::weakly_canonical(scenesCandidate);
        }

        if (!probe.has_parent_path())
        {
            break;
        }
        probe = probe.parent_path();
    }

    return cwdCandidate;
}

SceneObjectType parseSceneObjectType(const std::string& type)
{
    if (type == "cube")
    {
        return SceneObjectType::Cube;
    }
    if (type == "mesh")
    {
        return SceneObjectType::Mesh;
    }
    return SceneObjectType::Sphere;
}

Geom buildGeomFromObject(const SceneObject& object)
{
    Geom geom{};
    geom.type = (object.type == SceneObjectType::Cube) ? CUBE : SPHERE;
    geom.materialid = object.materialId;
    geom.translation = object.translation;
    geom.rotation = object.rotation;
    geom.scale = object.scale;
    geom.transform = utilityCore::buildTransformationMatrix(object.translation, object.rotation, object.scale);
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::inverseTranspose(geom.transform);
    return geom;
}

void computeTransformedBounds(
    const glm::mat4& transform,
    const glm::vec3& localMin,
    const glm::vec3& localMax,
    glm::vec3& outMin,
    glm::vec3& outMax)
{
    outMin = glm::vec3(FLT_MAX);
    outMax = glm::vec3(-FLT_MAX);

    for (int x = 0; x < 2; ++x)
    {
        for (int y = 0; y < 2; ++y)
        {
            for (int z = 0; z < 2; ++z)
            {
                const glm::vec3 localPoint(
                    x ? localMax.x : localMin.x,
                    y ? localMax.y : localMin.y,
                    z ? localMax.z : localMin.z);
                const glm::vec3 worldPoint = glm::vec3(transform * glm::vec4(localPoint, 1.0f));
                outMin = glm::min(outMin, worldPoint);
                outMax = glm::max(outMax, worldPoint);
            }
        }
    }
}

void appendGeomPrimitive(
    const Geom& geom,
    int geomIndex,
    std::vector<ScenePrimitive>& scenePrimitives)
{
    ScenePrimitive primitive{};
    primitive.type = SCENE_PRIMITIVE_GEOM;
    primitive.index = geomIndex;
    computeTransformedBounds(
        geom.transform,
        glm::vec3(-0.5f),
        glm::vec3(0.5f),
        primitive.bboxMin,
        primitive.bboxMax);
    scenePrimitives.push_back(primitive);
}

void appendMeshInstance(
    const SceneObject& object,
    std::vector<MeshInstance>& meshInstances,
    std::vector<ScenePrimitive>& scenePrimitives)
{
    if (object.localTriangles.empty() || object.bvhRootIndex < 0)
    {
        return;
    }

    const glm::mat4 transform = utilityCore::buildTransformationMatrix(
        object.translation,
        object.rotation,
        object.scale);

    MeshInstance meshInstance{};
    meshInstance.materialId = object.materialId;
    meshInstance.triangleStart = object.triangleStart;
    meshInstance.triangleCount = object.triangleCount;
    meshInstance.bvhRootIndex = object.bvhRootIndex;
    meshInstance.transform = transform;
    meshInstance.inverseTransform = glm::inverse(transform);
    meshInstance.invTranspose = glm::inverseTranspose(transform);
    meshInstance.localBboxMin = object.localBboxMin;
    meshInstance.localBboxMax = object.localBboxMax;
    computeTransformedBounds(
        meshInstance.transform,
        meshInstance.localBboxMin,
        meshInstance.localBboxMax,
        meshInstance.bboxMin,
        meshInstance.bboxMax);

    const int meshIndex = static_cast<int>(meshInstances.size());
    meshInstances.push_back(meshInstance);

    ScenePrimitive primitive{};
    primitive.type = SCENE_PRIMITIVE_MESH_INSTANCE;
    primitive.index = meshIndex;
    primitive.bboxMin = meshInstance.bboxMin;
    primitive.bboxMax = meshInstance.bboxMax;
    scenePrimitives.push_back(primitive);
}

std::string defaultObjectName(SceneObjectType type, int index)
{
    switch (type)
    {
    case SceneObjectType::Cube:
        return "Cube " + std::to_string(index);
    case SceneObjectType::Mesh:
        return "Mesh " + std::to_string(index);
    default:
        return "Sphere " + std::to_string(index);
    }
}
}

Scene::Scene(string filename)
{
    const std::filesystem::path resolvedPath = findSceneFile(filename);
    cout << "Reading scene from " << resolvedPath.string() << " ..." << endl;
    cout << " " << endl;

    const auto ext = resolvedPath.extension().string();
    if (ext == ".json")
    {
        loadFromJSON(resolvedPath.string());
        return;
    }

    cout << "Couldn't read from " << resolvedPath.string() << endl;
    exit(-1);
}

void Scene::rebuildStaticMeshData()
{
    triangles.clear();
    triangleBvhNodes.clear();

    for (SceneObject& object : objects)
    {
        object.triangleStart = -1;
        object.triangleCount = 0;
        object.bvhRootIndex = -1;
        object.localBboxMin = glm::vec3(0.0f);
        object.localBboxMax = glm::vec3(0.0f);

        if (object.type != SceneObjectType::Mesh || object.localTriangles.empty())
        {
            continue;
        }

        if (object.localBvhNodes.empty())
        {
            std::vector<Triangle> localTriangles = object.localTriangles;
            std::vector<TriangleBvhNode> localNodes;
            buildTriangleBvh(localTriangles, localNodes);
            object.localTriangles.swap(localTriangles);
            object.localBvhNodes.swap(localNodes);
        }

        if (object.localBvhNodes.empty())
        {
            continue;
        }

        object.triangleStart = static_cast<int>(triangles.size());
        object.triangleCount = static_cast<int>(object.localTriangles.size());
        object.bvhRootIndex = static_cast<int>(triangleBvhNodes.size());
        object.localBboxMin = object.localBvhNodes[0].bboxMin;
        object.localBboxMax = object.localBvhNodes[0].bboxMax;

        triangles.insert(triangles.end(), object.localTriangles.begin(), object.localTriangles.end());

        for (TriangleBvhNode node : object.localBvhNodes)
        {
            if (node.triCount > 0)
            {
                node.leftFirst += object.triangleStart;
            }
            else
            {
                node.leftFirst += object.bvhRootIndex;
                node.rightChild += object.bvhRootIndex;
            }
            triangleBvhNodes.push_back(node);
        }
    }
}

void Scene::rebuildRenderData()
{
    geoms.clear();
    meshInstances.clear();
    scenePrimitives.clear();
    sceneBvhNodes.clear();

    for (const SceneObject& object : objects)
    {
        if (object.type == SceneObjectType::Mesh)
        {
            appendMeshInstance(object, meshInstances, scenePrimitives);
            continue;
        }

        const int geomIndex = static_cast<int>(geoms.size());
        geoms.push_back(buildGeomFromObject(object));
        appendGeomPrimitive(geoms.back(), geomIndex, scenePrimitives);
    }

    buildSceneBvh(scenePrimitives, sceneBvhNodes);
    gpuDynamicDataDirty = true;
}

void Scene::updateObjectTransform(
    size_t objectIndex,
    const glm::vec3& translation,
    const glm::vec3& rotation,
    const glm::vec3& scale)
{
    if (objectIndex >= objects.size())
    {
        return;
    }

    objects[objectIndex].translation = translation;
    objects[objectIndex].rotation = rotation;
    objects[objectIndex].scale = scale;
    rebuildRenderData();
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    const std::filesystem::path scenePath = std::filesystem::absolute(jsonName);
    std::ifstream f(scenePath);
    if (!f)
    {
        throw std::runtime_error("Failed to open scene file: " + scenePath.string());
    }

    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> matNameToId;
    std::unordered_map<std::string, uint32_t> texturePathToId;

    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        newMaterial.textureId = -1;

        glm::vec3 baseColor(1.0f);
        if (p.contains("RGB"))
        {
            const auto& col = p["RGB"];
            baseColor = glm::vec3(col[0], col[1], col[2]);
        }

        newMaterial.color = baseColor;
        newMaterial.specular.color = baseColor;
        newMaterial.specular.exponent = p.value("ROUGHNESS", 0.0f);
        newMaterial.indexOfRefraction = p.value("IOR", 1.5f);

        if (p.contains("TEXTURE"))
        {
            const std::filesystem::path texturePath = resolveScenePath(scenePath, p["TEXTURE"]);
            const std::string textureKey = texturePath.lexically_normal().string();
            auto existing = texturePathToId.find(textureKey);
            if (existing != texturePathToId.end())
            {
                newMaterial.textureId = static_cast<int>(existing->second);
            }
            else
            {
                TextureData texture{};
                std::string error;
                if (!loadTextureImage(texturePath, texture, texturePixels, error))
                {
                    throw std::runtime_error(error);
                }
                newMaterial.textureId = static_cast<int>(textures.size());
                texturePathToId[textureKey] = static_cast<uint32_t>(textures.size());
                textures.push_back(texture);
            }
        }

        const std::string matType = p["TYPE"];

        if (matType == "Diffuse")
        {
            newMaterial.hasReflective = p.value("REFLECTIVITY", 0.0f);
            newMaterial.hasRefractive = p.value("REFRACTIVITY", 0.0f);
        }
        else if (matType == "Emitting")
        {
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (matType == "Specular")
        {
            newMaterial.hasReflective = p.value("REFLECTIVITY", 1.0f);
        }
        else if (matType == "Refractive" || matType == "Glass")
        {
            newMaterial.hasRefractive = p.value("REFRACTIVITY", 1.0f);
        }
        else
        {
            newMaterial.hasReflective = p.value("REFLECTIVITY", 0.0f);
            newMaterial.hasRefractive = p.value("REFRACTIVITY", 0.0f);
        }
        matNameToId[name] = static_cast<uint32_t>(materials.size());
        materials.emplace_back(newMaterial);
    }

    const auto& objectsData = data["Objects"];
    int objectIndex = 0;
    for (const auto& p : objectsData)
    {
        SceneObject object{};
        object.type = parseSceneObjectType(p["TYPE"]);
        object.materialId = static_cast<int>(matNameToId[p["MATERIAL"]]);
        object.name = p.value("NAME", defaultObjectName(object.type, objectIndex));
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        object.translation = glm::vec3(trans[0], trans[1], trans[2]);
        object.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        object.scale = glm::vec3(scale[0], scale[1], scale[2]);

        if (object.type == SceneObjectType::Mesh)
        {
            const std::filesystem::path meshPath = resolveScenePath(scenePath, p["FILE"]);
            object.meshPath = meshPath.string();
            std::string error;
            if (!loadObjMesh(meshPath, glm::mat4(1.0f), glm::mat4(1.0f), object.materialId, object.localTriangles, error))
            {
                throw std::runtime_error(error);
            }
            std::vector<Triangle> localTriangles = object.localTriangles;
            if (!buildTriangleBvh(localTriangles, object.localBvhNodes) || object.localBvhNodes.empty())
            {
                throw std::runtime_error("Failed to build mesh BVH: " + meshPath.string());
            }
            object.localTriangles.swap(localTriangles);
            object.localBboxMin = object.localBvhNodes[0].bboxMin;
            object.localBboxMax = object.localBvhNodes[0].bboxMax;
        }

        objects.push_back(object);
        ++objectIndex;
    }

    rebuildStaticMeshData();
    rebuildRenderData();

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

