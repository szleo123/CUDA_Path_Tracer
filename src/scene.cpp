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
#include <algorithm>
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
    std::vector<std::string>& materialNames;
    std::vector<TextureData>& textures;
    std::vector<glm::vec4>& texturePixels;
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
    bool decodeSrgb,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<TextureData>& textures,
    std::vector<glm::vec4>& texturePixels)
{
    if (texturePath.empty())
    {
        return -1;
    }

    const std::string textureKey = std::filesystem::weakly_canonical(texturePath).string()
        + "|"
        + (decodeSrgb ? "srgb" : "linear");
    auto existing = texturePathToId.find(textureKey);
    if (existing != texturePathToId.end())
    {
        return static_cast<int>(existing->second);
    }

    TextureData texture{};
    std::string error;
    if (!loadTextureImage(texturePath, true, decodeSrgb, texture, texturePixels, error))
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

int ensureHdrTextureLoaded(
    const std::filesystem::path& texturePath,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<TextureData>& textures,
    std::vector<glm::vec4>& texturePixels)
{
    if (texturePath.empty())
    {
        return -1;
    }

    const std::string textureKey = std::filesystem::weakly_canonical(texturePath).string() + "|hdr";
    auto existing = texturePathToId.find(textureKey);
    if (existing != texturePathToId.end())
    {
        return static_cast<int>(existing->second);
    }

    TextureData texture{};
    std::string error;
    if (!loadHdrImage(texturePath, false, texture, texturePixels, error))
    {
        throw std::runtime_error(error);
    }
    texture.wrapS = 10497;
    texture.wrapT = 33071;

    const int textureId = static_cast<int>(textures.size());
    texturePathToId[textureKey] = static_cast<uint32_t>(textureId);
    textures.push_back(texture);
    return textureId;
}

std::string importedMaterialKey(
    const std::filesystem::path& meshPath,
    const MeshMaterialDefinition& material)
{
    const std::string diffuseTextureKey = !material.diffuseTextureKey.empty()
        ? material.diffuseTextureKey
        : (material.diffuseTexturePath.empty()
            ? std::string()
            : std::filesystem::weakly_canonical(material.diffuseTexturePath).string());
    const std::string metallicRoughnessTextureKey = !material.metallicRoughnessTextureKey.empty()
        ? material.metallicRoughnessTextureKey
        : (material.metallicRoughnessTexturePath.empty()
            ? std::string()
            : std::filesystem::weakly_canonical(material.metallicRoughnessTexturePath).string());
    const std::string normalTextureKey = !material.normalTextureKey.empty()
        ? material.normalTextureKey
        : (material.normalTexturePath.empty()
            ? std::string()
            : std::filesystem::weakly_canonical(material.normalTexturePath).string());
    const std::string emissiveTextureKey = !material.emissiveTextureKey.empty()
        ? material.emissiveTextureKey
        : (material.emissiveTexturePath.empty()
            ? std::string()
            : std::filesystem::weakly_canonical(material.emissiveTexturePath).string());
    const std::string occlusionTextureKey = !material.occlusionTextureKey.empty()
        ? material.occlusionTextureKey
        : (material.occlusionTexturePath.empty()
            ? std::string()
            : std::filesystem::weakly_canonical(material.occlusionTexturePath).string());
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
        + std::to_string(material.baseAlpha)
        + "|"
        + std::to_string(material.metallicFactor)
        + "|"
        + std::to_string(material.roughnessFactor)
        + "|"
        + std::to_string(material.indexOfRefraction)
        + "|"
        + std::to_string(material.emissiveFactor.r)
        + "|"
        + std::to_string(material.emissiveFactor.g)
        + "|"
        + std::to_string(material.emissiveFactor.b)
        + "|"
        + std::to_string(material.emissiveStrength)
        + "|"
        + std::to_string(material.transmissionFactor)
        + "|"
        + std::to_string(material.clearcoatFactor)
        + "|"
        + std::to_string(material.clearcoatRoughnessFactor)
        + "|"
        + std::to_string(material.specularFactor)
        + "|"
        + std::to_string(material.specularFactorColor.r)
        + "|"
        + std::to_string(material.specularFactorColor.g)
        + "|"
        + std::to_string(material.specularFactorColor.b)
        + "|"
        + std::to_string(material.hasExplicitSpecularColor)
        + "|"
        + diffuseTextureKey
        + "|"
        + metallicRoughnessTextureKey
        + "|"
        + normalTextureKey
        + "|"
        + emissiveTextureKey
        + "|"
        + occlusionTextureKey
        + "|"
        + std::to_string(material.diffuseTexcoordSet)
        + "|"
        + std::to_string(material.metallicRoughnessTexcoordSet)
        + "|"
        + std::to_string(material.normalTexcoordSet)
        + "|"
        + std::to_string(material.emissiveTexcoordSet)
        + "|"
        + std::to_string(material.occlusionTexcoordSet)
        + "|"
        + std::to_string(material.normalTextureScale)
        + "|"
        + std::to_string(material.occlusionStrength)
        + "|"
        + std::to_string(material.thinWalled)
        + "|"
        + std::to_string(material.flipV ? 1 : 0)
        + "|"
        + std::to_string(material.wrapS)
        + "|"
        + std::to_string(material.wrapT)
        + "|"
        + std::to_string(material.doubleSided);
}

static glm::vec3 dielectricF0FromIor(
    float indexOfRefraction,
    float specularFactor,
    const glm::vec3& specularFactorColor)
{
    const float clampedIor = glm::max(indexOfRefraction, 1.0f);
    const float numerator = clampedIor - 1.0f;
    const float denominator = clampedIor + 1.0f;
    const float baseF0 = (denominator > EPSILON) ? ((numerator * numerator) / (denominator * denominator)) : 0.04f;
    return glm::clamp(glm::vec3(baseF0) * specularFactor * specularFactorColor, glm::vec3(0.0f), glm::vec3(1.0f));
}

int ensureImportedTextureLoaded(
    const std::filesystem::path& texturePath,
    const std::string& embeddedTextureKey,
    const std::vector<unsigned char>& textureBytes,
    bool flipV,
    bool decodeSrgb,
    int wrapS,
    int wrapT,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<TextureData>& textures,
    std::vector<glm::vec4>& texturePixels)
{
    const std::string textureKey = !embeddedTextureKey.empty()
        ? embeddedTextureKey
        : (texturePath.empty()
            ? std::string()
            : std::filesystem::weakly_canonical(texturePath).string());
    const std::string texturedSamplerKey = textureKey
        + "|"
        + std::to_string(flipV ? 1 : 0)
        + "|"
        + (decodeSrgb ? "srgb" : "linear")
        + "|"
        + std::to_string(wrapS)
        + "|"
        + std::to_string(wrapT);

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
    if (!textureBytes.empty())
    {
        loaded = loadTextureImageFromMemory(
            textureBytes.data(),
            textureBytes.size(),
            flipV,
            decodeSrgb,
            texture,
            texturePixels,
            error);
    }
    else if (!texturePath.empty())
    {
        loaded = loadTextureImage(texturePath, flipV, decodeSrgb, texture, texturePixels, error);
    }

    if (!loaded)
    {
        throw std::runtime_error(error.empty() ? ("Failed to load imported texture: " + textureKey) : error);
    }
    texture.wrapS = wrapS;
    texture.wrapT = wrapT;

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
    std::vector<std::string>& materialNames,
    std::vector<TextureData>& textures,
    std::vector<glm::vec4>& texturePixels)
{
    const std::string key = importedMaterialKey(meshPath, importedMaterial);
    auto existing = importedMaterialKeyToId.find(key);
    if (existing != importedMaterialKeyToId.end())
    {
        return static_cast<int>(existing->second);
    }

    Material material = baseMaterial;
    material.color = importedMaterial.diffuseColor;
    material.baseAlpha = importedMaterial.baseAlpha;
    material.indexOfRefraction = importedMaterial.indexOfRefraction;
    material.specularColor = importedMaterial.hasExplicitSpecularColor
        ? glm::clamp(
            importedMaterial.specularColor * importedMaterial.specularFactor * importedMaterial.specularFactorColor,
            glm::vec3(0.0f),
            glm::vec3(1.0f))
        : dielectricF0FromIor(
            importedMaterial.indexOfRefraction,
            importedMaterial.specularFactor,
            importedMaterial.specularFactorColor);
    material.emissiveColor = importedMaterial.emissiveFactor;
    material.roughness = importedMaterial.roughnessFactor;
    material.metallic = importedMaterial.metallicFactor;
    material.alphaMode = importedMaterial.alphaMode;
    material.alphaCutoff = importedMaterial.alphaCutoff;
    material.doubleSided = importedMaterial.doubleSided;
    material.thinWalled = importedMaterial.thinWalled;
    material.normalTextureScale = importedMaterial.normalTextureScale;
    material.baseColorTexcoordSet = importedMaterial.diffuseTexcoordSet;
    material.metallicRoughnessTexcoordSet = importedMaterial.metallicRoughnessTexcoordSet;
    material.normalTexcoordSet = importedMaterial.normalTexcoordSet;
    material.emissiveTexcoordSet = importedMaterial.emissiveTexcoordSet;
    material.occlusionTexcoordSet = importedMaterial.occlusionTexcoordSet;
    material.transmissionFactor = importedMaterial.transmissionFactor;
    material.clearcoatFactor = importedMaterial.clearcoatFactor;
    material.clearcoatRoughness = importedMaterial.clearcoatRoughnessFactor;
    material.occlusionStrength = importedMaterial.occlusionStrength;
    material.hasReflective = fmaxf(material.hasReflective, importedMaterial.metallicFactor);
    material.hasRefractive = fmaxf(material.hasRefractive, importedMaterial.transmissionFactor);
    if (glm::length(importedMaterial.emissiveFactor) > 0.0f
        || !importedMaterial.emissiveTextureBytes.empty()
        || !importedMaterial.emissiveTexturePath.empty())
    {
        material.emittance = importedMaterial.emissiveStrength;
    }
    material.textureId = ensureImportedTextureLoaded(
        importedMaterial.diffuseTexturePath,
        importedMaterial.diffuseTextureKey,
        importedMaterial.diffuseTextureBytes,
        importedMaterial.flipV,
        true,
        importedMaterial.wrapS,
        importedMaterial.wrapT,
        texturePathToId,
        textures,
        texturePixels);
    material.metallicRoughnessTextureId = ensureImportedTextureLoaded(
        importedMaterial.metallicRoughnessTexturePath,
        importedMaterial.metallicRoughnessTextureKey,
        importedMaterial.metallicRoughnessTextureBytes,
        importedMaterial.flipV,
        false,
        importedMaterial.wrapS,
        importedMaterial.wrapT,
        texturePathToId,
        textures,
        texturePixels);
    material.normalTextureId = ensureImportedTextureLoaded(
        importedMaterial.normalTexturePath,
        importedMaterial.normalTextureKey,
        importedMaterial.normalTextureBytes,
        importedMaterial.flipV,
        false,
        importedMaterial.wrapS,
        importedMaterial.wrapT,
        texturePathToId,
        textures,
        texturePixels);
    material.emissiveTextureId = ensureImportedTextureLoaded(
        importedMaterial.emissiveTexturePath,
        importedMaterial.emissiveTextureKey,
        importedMaterial.emissiveTextureBytes,
        importedMaterial.flipV,
        true,
        importedMaterial.wrapS,
        importedMaterial.wrapT,
        texturePathToId,
        textures,
        texturePixels);
    material.occlusionTextureId = ensureImportedTextureLoaded(
        importedMaterial.occlusionTexturePath,
        importedMaterial.occlusionTextureKey,
        importedMaterial.occlusionTextureBytes,
        importedMaterial.flipV,
        false,
        importedMaterial.wrapS,
        importedMaterial.wrapT,
        texturePathToId,
        textures,
        texturePixels);

    const int materialId = static_cast<int>(materials.size());
    importedMaterialKeyToId[key] = static_cast<uint32_t>(materialId);
    materials.push_back(material);
    materialNames.push_back(importedMaterial.name.empty() ? ("Imported Material " + std::to_string(materialId)) : importedMaterial.name);
    return materialId;
}

Material parseMaterialDefinition(
    const json& materialJson,
    const std::filesystem::path& scenePath,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<TextureData>& textures,
    std::vector<glm::vec4>& texturePixels)
{
    Material material{};
    material.textureId = -1;
    material.metallicRoughnessTextureId = -1;
    material.normalTextureId = -1;
    material.emissiveTextureId = -1;
    material.occlusionTextureId = -1;
    material.alphaMode = 0;
    material.doubleSided = 0;
    material.thinWalled = 0;
    material.baseAlpha = 1.0f;
    material.alphaCutoff = 0.5f;
    material.normalTextureScale = 1.0f;
    material.baseColorTexcoordSet = 0;
    material.metallicRoughnessTexcoordSet = 0;
    material.normalTexcoordSet = 0;
    material.emissiveTexcoordSet = 0;
    material.occlusionTexcoordSet = 0;
    material.occlusionStrength = 1.0f;

    glm::vec3 baseColor(1.0f);
    if (materialJson.contains("RGB"))
    {
        const auto& col = materialJson["RGB"];
        baseColor = glm::vec3(col[0], col[1], col[2]);
    }

    material.color = baseColor;
    material.emissiveColor = glm::vec3(0.0f);
    material.specularColor = glm::vec3(0.04f);
    material.roughness = materialJson.value("ROUGHNESS", 0.0f);
    material.metallic = materialJson.value("METALLIC", 0.0f);
    material.indexOfRefraction = materialJson.value("IOR", 1.5f);

    if (materialJson.contains("SPECULAR_RGB"))
    {
        const auto& col = materialJson["SPECULAR_RGB"];
        material.specularColor = glm::vec3(col[0], col[1], col[2]);
    }

    if (materialJson.contains("TEXTURE"))
    {
        const std::filesystem::path texturePath = resolveScenePath(scenePath, materialJson["TEXTURE"]);
        material.textureId = ensureTextureLoaded(texturePath, true, texturePathToId, textures, texturePixels);
    }
    if (materialJson.contains("METALLIC_ROUGHNESS_TEXTURE"))
    {
        const std::filesystem::path texturePath = resolveScenePath(scenePath, materialJson["METALLIC_ROUGHNESS_TEXTURE"]);
        material.metallicRoughnessTextureId = ensureTextureLoaded(texturePath, false, texturePathToId, textures, texturePixels);
    }
    if (materialJson.contains("NORMAL_TEXTURE"))
    {
        const std::filesystem::path texturePath = resolveScenePath(scenePath, materialJson["NORMAL_TEXTURE"]);
        material.normalTextureId = ensureTextureLoaded(texturePath, false, texturePathToId, textures, texturePixels);
        material.normalTextureScale = materialJson.value("NORMAL_SCALE", 1.0f);
    }

    const std::string matType = materialJson["TYPE"];
    if (matType == "Diffuse")
    {
        material.hasReflective = fmaxf(materialJson.value("REFLECTIVITY", 0.0f), material.metallic);
        material.hasRefractive = materialJson.value("REFRACTIVITY", 0.0f);
    }
    else if (matType == "Emitting")
    {
        material.emittance = materialJson["EMITTANCE"];
        material.emissiveColor = material.color;
    }
    else if (matType == "Specular")
    {
        material.metallic = materialJson.value("METALLIC", 1.0f);
        material.hasReflective = materialJson.value("REFLECTIVITY", 1.0f);
    }
    else if (matType == "Refractive" || matType == "Glass")
    {
        material.hasRefractive = materialJson.value("REFRACTIVITY", 1.0f);
    }
    else
    {
        material.hasReflective = fmaxf(materialJson.value("REFLECTIVITY", 0.0f), material.metallic);
        material.hasRefractive = materialJson.value("REFRACTIVITY", 0.0f);
    }

    return material;
}

void remapMeshTriangleMaterials(
    SceneObject& object,
    const std::filesystem::path& meshPath,
    Material baseMaterial,
    const std::vector<MeshMaterialDefinition>& importedMaterials,
    std::unordered_map<std::string, uint32_t>& importedMaterialKeyToId,
    std::unordered_map<std::string, uint32_t>& texturePathToId,
    std::vector<Material>& materials,
    std::vector<std::string>& materialNames,
    std::vector<TextureData>& textures,
    std::vector<glm::vec4>& texturePixels)
{
    object.usedMaterialIds.clear();

    if (importedMaterials.empty())
    {
        for (Triangle& triangle : object.localTriangles)
        {
            triangle.materialId = object.materialId;
        }
        object.usedMaterialIds.push_back(object.materialId);
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
            materialNames,
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

    object.usedMaterialIds.reserve(localMaterialToSceneMaterial.size());
    for (const int materialId : localMaterialToSceneMaterial)
    {
        if (std::find(object.usedMaterialIds.begin(), object.usedMaterialIds.end(), materialId) == object.usedMaterialIds.end())
        {
            object.usedMaterialIds.push_back(materialId);
        }
    }
}

void initializeMeshObject(
    SceneObject& object,
    const SceneImportContext& importContext,
    Material baseMaterial,
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
        importContext.materialNames,
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

    std::cout
        << "Loaded mesh object '" << object.name << "' from " << meshPath.string()
        << " with " << object.localTriangles.size() << " triangles"
        << " bboxMin=(" << object.localBboxMin.x << ", " << object.localBboxMin.y << ", " << object.localBboxMin.z << ")"
        << " bboxMax=(" << object.localBboxMax.x << ", " << object.localBboxMax.y << ", " << object.localBboxMax.z << ")"
        << std::endl;

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
        importContext.materialNames.push_back(name);
    }
}

SceneObject parseSceneObjectDefinition(
    const json& objectJson,
    int objectIndex,
    const std::unordered_map<std::string, uint32_t>& materialNameToId)
{
    SceneObject object{};
    object.type = parseSceneObjectType(objectJson["TYPE"]);
    const std::string materialName = objectJson["MATERIAL"];
    const auto materialIt = materialNameToId.find(materialName);
    if (materialIt == materialNameToId.end())
    {
        throw std::runtime_error("Unknown material referenced by object: " + materialName);
    }
    object.materialId = static_cast<int>(materialIt->second);
    object.name = objectJson.value("NAME", defaultObjectName(object.type, objectIndex));
    object.translation = parseVec3(objectJson["TRANS"]);
    object.rotation = parseVec3(objectJson["ROTAT"]);
    object.scale = parseVec3(objectJson["SCALE"]);
    object.usedMaterialIds.push_back(object.materialId);

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

void configureEnvironmentFromJson(
    const json& data,
    const SceneImportContext& importContext,
    RenderState& renderState)
{
    EnvironmentSettings environment{};

    if (data.contains("Environment"))
    {
        const json& environmentData = data["Environment"];
        environment.intensity = environmentData.value("INTENSITY", 1.0f);
        environment.rotation = environmentData.value("ROTATION", 0.0f);

        if (environmentData.contains("SKY_ZENITH"))
        {
            environment.zenithColor = parseVec3(environmentData["SKY_ZENITH"]);
        }
        if (environmentData.contains("SKY_HORIZON"))
        {
            environment.horizonColor = parseVec3(environmentData["SKY_HORIZON"]);
        }
        if (environmentData.contains("GROUND_COLOR"))
        {
            environment.groundColor = parseVec3(environmentData["GROUND_COLOR"]);
        }
        if (environmentData.contains("TYPE"))
        {
            const std::string type = environmentData["TYPE"];
            environment.useProceduralSky = (type != "HDR");
        }
        if (environmentData.contains("FILE"))
        {
            const std::filesystem::path hdrPath = resolveScenePath(importContext.scenePath, environmentData["FILE"]);
            environment.textureId = ensureHdrTextureLoaded(
                hdrPath,
                importContext.texturePathToId,
                importContext.textures,
                importContext.texturePixels);
            environment.useProceduralSky = 0;
        }
    }

    renderState.environment = environment;
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

void Scene::updateMaterial(
    size_t materialIndex,
    const Material& material)
{
    if (materialIndex >= materials.size())
    {
        return;
    }

    materials[materialIndex] = material;
    gpuDynamicDataDirty = true;
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
        materialNames,
        textures,
        texturePixels
    };

    loadMaterialsFromJson(data["Materials"], importContext);
    loadObjectsFromJson(data["Objects"], importContext, objects);

    rebuildStaticMeshData();
    rebuildRenderData();
    configureCameraFromJson(data["Camera"], state);
    configureEnvironmentFromJson(data, importContext, state);
}

