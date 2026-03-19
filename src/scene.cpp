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
struct SceneImportContext
{
    const std::filesystem::path& scenePath;
    std::unordered_map<std::string, uint32_t>& materialNameToId;
    std::unordered_map<std::string, uint32_t>& texturePathToId;
    std::unordered_map<std::string, uint32_t>& importedMaterialKeyToId;
    std::vector<Material>& materials;
    std::vector<TextureData>& textures;
    std::vector<glm::vec3>& texturePixels;
};

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

Geom buildGeomFromObject(const SceneObject& object, int objectIndex)
{
    Geom geom{};
    geom.type = (object.type == SceneObjectType::Cube) ? CUBE : SPHERE;
    geom.materialid = object.materialId;
    geom.objectIndex = objectIndex;
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
    int objectIndex,
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
    meshInstance.objectIndex = objectIndex;
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

glm::vec3 parseVec3(const json& value)
{
    return glm::vec3(value[0], value[1], value[2]);
}

int ensureTextureLoaded(
    const std::filesystem::path& texturePath,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<TextureData>& textures,
    std::vector<glm::vec3>& texturePixels)
{
    if (texturePath.empty())
    {
        return -1;
    }

    const std::string textureKey = std::filesystem::weakly_canonical(texturePath).string();
    auto existing = texturePathToId.find(textureKey);
    if (existing != texturePathToId.end())
    {
        return static_cast<int>(existing->second);
    }

    TextureData texture{};
    std::string error;
    if (!loadTextureImage(texturePath, true, texture, texturePixels, error))
    {
        throw std::runtime_error(error);
    }
    texture.wrapS = 10497;
    texture.wrapT = 10497;

    const int textureId = static_cast<int>(textures.size());
    texturePathToId[textureKey] = static_cast<uint32_t>(textureId);
    textures.push_back(texture);
    return textureId;
}

std::string importedMaterialKey(
    const std::filesystem::path& meshPath,
    const MeshMaterialDefinition& material)
{
    const std::string textureKey = !material.diffuseTextureKey.empty()
        ? material.diffuseTextureKey
        : (material.diffuseTexturePath.empty()
            ? std::string()
            : std::filesystem::weakly_canonical(material.diffuseTexturePath).string());
    return meshPath.string()
        + "|"
        + material.name
        + "|"
        + std::to_string(material.diffuseColor.r)
        + "|"
        + std::to_string(material.diffuseColor.g)
        + "|"
        + std::to_string(material.diffuseColor.b)
        + "|"
        + textureKey
        + "|"
        + std::to_string(material.diffuseTexcoordSet)
        + "|"
        + std::to_string(material.flipV ? 1 : 0)
        + "|"
        + std::to_string(material.wrapS)
        + "|"
        + std::to_string(material.wrapT);
}

int ensureImportedTextureLoaded(
    const MeshMaterialDefinition& importedMaterial,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<TextureData>& textures,
    std::vector<glm::vec3>& texturePixels)
{
    const std::string textureKey = !importedMaterial.diffuseTextureKey.empty()
        ? importedMaterial.diffuseTextureKey
        : (importedMaterial.diffuseTexturePath.empty()
            ? std::string()
            : std::filesystem::weakly_canonical(importedMaterial.diffuseTexturePath).string());
    const std::string texturedSamplerKey = textureKey
        + "|"
        + std::to_string(importedMaterial.flipV ? 1 : 0)
        + "|"
        + std::to_string(importedMaterial.wrapS)
        + "|"
        + std::to_string(importedMaterial.wrapT);

    if (textureKey.empty())
    {
        return -1;
    }

    auto existing = texturePathToId.find(texturedSamplerKey);
    if (existing != texturePathToId.end())
    {
        return static_cast<int>(existing->second);
    }

    TextureData texture{};
    std::string error;
    bool loaded = false;
    if (!importedMaterial.diffuseTextureBytes.empty())
    {
        loaded = loadTextureImageFromMemory(
            importedMaterial.diffuseTextureBytes.data(),
            importedMaterial.diffuseTextureBytes.size(),
            importedMaterial.flipV,
            texture,
            texturePixels,
            error);
    }
    else if (!importedMaterial.diffuseTexturePath.empty())
    {
        loaded = loadTextureImage(importedMaterial.diffuseTexturePath, importedMaterial.flipV, texture, texturePixels, error);
    }

    if (!loaded)
    {
        throw std::runtime_error(error.empty() ? ("Failed to load imported texture: " + textureKey) : error);
    }
    texture.wrapS = importedMaterial.wrapS;
    texture.wrapT = importedMaterial.wrapT;

    const int textureId = static_cast<int>(textures.size());
    texturePathToId[texturedSamplerKey] = static_cast<uint32_t>(textureId);
    textures.push_back(texture);
    return textureId;
}

int registerImportedMaterial(
    const std::filesystem::path& meshPath,
    const Material& baseMaterial,
    const MeshMaterialDefinition& importedMaterial,
    std::unordered_map<std::string, uint32_t>& importedMaterialKeyToId,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<Material>& materials,
    std::vector<TextureData>& textures,
    std::vector<glm::vec3>& texturePixels)
{
    const std::string key = importedMaterialKey(meshPath, importedMaterial);
    auto existing = importedMaterialKeyToId.find(key);
    if (existing != importedMaterialKeyToId.end())
    {
        return static_cast<int>(existing->second);
    }

    Material material = baseMaterial;
    material.color = importedMaterial.diffuseColor;
    material.specular.color = importedMaterial.diffuseColor;
    material.textureId = ensureImportedTextureLoaded(importedMaterial, texturePathToId, textures, texturePixels);

    const int materialId = static_cast<int>(materials.size());
    importedMaterialKeyToId[key] = static_cast<uint32_t>(materialId);
    materials.push_back(material);
    return materialId;
}

Material parseMaterialDefinition(
    const json& materialJson,
    const std::filesystem::path& scenePath,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<TextureData>& textures,
    std::vector<glm::vec3>& texturePixels)
{
    Material material{};
    material.textureId = -1;

    glm::vec3 baseColor(1.0f);
    if (materialJson.contains("RGB"))
    {
        const auto& col = materialJson["RGB"];
        baseColor = glm::vec3(col[0], col[1], col[2]);
    }

    material.color = baseColor;
    material.specular.color = baseColor;
    material.specular.exponent = materialJson.value("ROUGHNESS", 0.0f);
    material.indexOfRefraction = materialJson.value("IOR", 1.5f);

    if (materialJson.contains("TEXTURE"))
    {
        const std::filesystem::path texturePath = resolveScenePath(scenePath, materialJson["TEXTURE"]);
        material.textureId = ensureTextureLoaded(texturePath, texturePathToId, textures, texturePixels);
    }

    const std::string matType = materialJson["TYPE"];
    if (matType == "Diffuse")
    {
        material.hasReflective = materialJson.value("REFLECTIVITY", 0.0f);
        material.hasRefractive = materialJson.value("REFRACTIVITY", 0.0f);
    }
    else if (matType == "Emitting")
    {
        material.emittance = materialJson["EMITTANCE"];
    }
    else if (matType == "Specular")
    {
        material.hasReflective = materialJson.value("REFLECTIVITY", 1.0f);
    }
    else if (matType == "Refractive" || matType == "Glass")
    {
        material.hasRefractive = materialJson.value("REFRACTIVITY", 1.0f);
    }
    else
    {
        material.hasReflective = materialJson.value("REFLECTIVITY", 0.0f);
        material.hasRefractive = materialJson.value("REFRACTIVITY", 0.0f);
    }

    return material;
}

void remapMeshTriangleMaterials(
    SceneObject& object,
    const std::filesystem::path& meshPath,
    const Material& baseMaterial,
    const std::vector<MeshMaterialDefinition>& importedMaterials,
    std::unordered_map<std::string, uint32_t>& importedMaterialKeyToId,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<Material>& materials,
    std::vector<TextureData>& textures,
    std::vector<glm::vec3>& texturePixels)
{
    if (importedMaterials.empty())
    {
        for (Triangle& triangle : object.localTriangles)
        {
            triangle.materialId = object.materialId;
        }
        return;
    }

    std::vector<int> localMaterialToSceneMaterial(importedMaterials.size(), object.materialId);
    for (size_t i = 0; i < importedMaterials.size(); ++i)
    {
        localMaterialToSceneMaterial[i] = registerImportedMaterial(
            meshPath,
            baseMaterial,
            importedMaterials[i],
            importedMaterialKeyToId,
            texturePathToId,
            materials,
            textures,
            texturePixels);
    }

    for (Triangle& triangle : object.localTriangles)
    {
        if (triangle.materialId >= 0 && triangle.materialId < static_cast<int>(localMaterialToSceneMaterial.size()))
        {
            triangle.materialId = localMaterialToSceneMaterial[triangle.materialId];
        }
        else
        {
            triangle.materialId = object.materialId;
        }
    }
}

void initializeMeshObject(
    SceneObject& object,
    const SceneImportContext& importContext,
    const Material& baseMaterial,
    std::vector<MeshMaterialDefinition>& importedMaterials)
{
    const std::filesystem::path meshPath = resolveScenePath(importContext.scenePath, object.meshPath);
    object.meshPath = meshPath.string();

    std::string error;
    if (!loadMeshAsset(meshPath, object.materialId, object.localTriangles, importedMaterials, error))
    {
        throw std::runtime_error(error);
    }

    remapMeshTriangleMaterials(
        object,
        meshPath,
        baseMaterial,
        importedMaterials,
        importContext.importedMaterialKeyToId,
        importContext.texturePathToId,
        importContext.materials,
        importContext.textures,
        importContext.texturePixels);

    std::vector<Triangle> localTriangles = object.localTriangles;
    if (!buildTriangleBvh(localTriangles, object.localBvhNodes) || object.localBvhNodes.empty())
    {
        throw std::runtime_error("Failed to build mesh BVH: " + meshPath.string());
    }
    object.localTriangles.swap(localTriangles);
    object.localBboxMin = object.localBvhNodes[0].bboxMin;
    object.localBboxMax = object.localBvhNodes[0].bboxMax;
}

void loadMaterialsFromJson(const json& materialsData, const SceneImportContext& importContext)
{
    for (const auto& item : materialsData.items())
    {
        const std::string& name = item.key();
        const Material material = parseMaterialDefinition(
            item.value(),
            importContext.scenePath,
            importContext.texturePathToId,
            importContext.textures,
            importContext.texturePixels);

        importContext.materialNameToId[name] = static_cast<uint32_t>(importContext.materials.size());
        importContext.materials.push_back(material);
    }
}

SceneObject parseSceneObjectDefinition(
    const json& objectJson,
    int objectIndex,
    const std::unordered_map<std::string, uint32_t>& materialNameToId)
{
    SceneObject object{};
    object.type = parseSceneObjectType(objectJson["TYPE"]);
    object.materialId = static_cast<int>(materialNameToId.at(objectJson["MATERIAL"]));
    object.name = objectJson.value("NAME", defaultObjectName(object.type, objectIndex));
    object.translation = parseVec3(objectJson["TRANS"]);
    object.rotation = parseVec3(objectJson["ROTAT"]);
    object.scale = parseVec3(objectJson["SCALE"]);

    if (object.type == SceneObjectType::Mesh)
    {
        object.meshPath = objectJson["FILE"];
    }

    return object;
}

void loadObjectsFromJson(
    const json& objectsData,
    const SceneImportContext& importContext,
    std::vector<SceneObject>& objects)
{
    for (size_t objectIndex = 0; objectIndex < objectsData.size(); ++objectIndex)
    {
        SceneObject object = parseSceneObjectDefinition(
            objectsData[objectIndex],
            static_cast<int>(objectIndex),
            importContext.materialNameToId);

        if (object.type == SceneObjectType::Mesh)
        {
            std::vector<MeshMaterialDefinition> importedMaterials;
            initializeMeshObject(object, importContext, importContext.materials[object.materialId], importedMaterials);
        }

        objects.push_back(std::move(object));
    }
}

void configureCameraFromJson(const json& cameraData, RenderState& renderState)
{
    Camera& camera = renderState.camera;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];

    const float fovy = cameraData["FOVY"];
    renderState.iterations = cameraData["ITERATIONS"];
    renderState.traceDepth = cameraData["DEPTH"];
    renderState.imageName = cameraData["FILE"];
    camera.position = parseVec3(cameraData["EYE"]);
    camera.lookAt = parseVec3(cameraData["LOOKAT"]);
    camera.up = parseVec3(cameraData["UP"]);

    const float yscaled = tan(fovy * (PI / 180.0f));
    const float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    const float fovx = (atan(xscaled) * 180.0f) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    camera.pixelLength = glm::vec2(
        2.0f * xscaled / static_cast<float>(camera.resolution.x),
        2.0f * yscaled / static_cast<float>(camera.resolution.y));

    const int pixelCount = camera.resolution.x * camera.resolution.y;
    renderState.image.resize(pixelCount);
    std::fill(renderState.image.begin(), renderState.image.end(), glm::vec3(0.0f));
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

    for (size_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex)
    {
        const SceneObject& object = objects[objectIndex];
        if (object.type == SceneObjectType::Mesh)
        {
            appendMeshInstance(object, static_cast<int>(objectIndex), meshInstances, scenePrimitives);
            continue;
        }

        const int geomIndex = static_cast<int>(geoms.size());
        geoms.push_back(buildGeomFromObject(object, static_cast<int>(objectIndex)));
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
    std::unordered_map<std::string, uint32_t> materialNameToId;
    std::unordered_map<std::string, uint32_t> texturePathToId;
    std::unordered_map<std::string, uint32_t> importedMaterialKeyToId;
    SceneImportContext importContext{
        scenePath,
        materialNameToId,
        texturePathToId,
        importedMaterialKeyToId,
        materials,
        textures,
        texturePixels
    };

    loadMaterialsFromJson(data["Materials"], importContext);
    loadObjectsFromJson(data["Objects"], importContext, objects);

    rebuildStaticMeshData();
    rebuildRenderData();
    configureCameraFromJson(data["Camera"], state);
}

