#pragma once

#include "sceneStructs.h"

#include <filesystem>
#include <string>
#include <vector>

struct MeshMaterialDefinition
{
    std::string name;
    glm::vec3 diffuseColor = glm::vec3(1.0f);
    glm::vec3 specularColor = glm::vec3(0.04f);
    float metallicFactor = 0.0f;
    float roughnessFactor = 1.0f;
    float alphaCutoff = 0.5f;
    int alphaMode = 0;
    int doubleSided = 0;
    std::filesystem::path diffuseTexturePath;
    std::string diffuseTextureKey;
    std::vector<unsigned char> diffuseTextureBytes;
    int diffuseTexcoordSet = 0;
    std::filesystem::path metallicRoughnessTexturePath;
    std::string metallicRoughnessTextureKey;
    std::vector<unsigned char> metallicRoughnessTextureBytes;
    int metallicRoughnessTexcoordSet = 0;
    std::filesystem::path normalTexturePath;
    std::string normalTextureKey;
    std::vector<unsigned char> normalTextureBytes;
    int normalTexcoordSet = 0;
    float normalTextureScale = 1.0f;
    int wrapS = 10497;
    int wrapT = 10497;
    bool flipV = true;
};

bool loadTextureImage(
    const std::filesystem::path& texturePath,
    bool flipV,
    bool decodeSrgb,
    TextureData& outTexture,
    std::vector<glm::vec4>& outPixels,
    std::string& outError);

bool loadTextureImageFromMemory(
    const unsigned char* textureBytes,
    size_t textureByteCount,
    bool flipV,
    bool decodeSrgb,
    TextureData& outTexture,
    std::vector<glm::vec4>& outPixels,
    std::string& outError);

bool loadHdrImage(
    const std::filesystem::path& texturePath,
    bool flipV,
    TextureData& outTexture,
    std::vector<glm::vec4>& outPixels,
    std::string& outError);

bool loadMeshAsset(
    const std::filesystem::path& meshPath,
    int materialId,
    std::vector<Triangle>& outTriangles,
    std::vector<MeshMaterialDefinition>& outMaterials,
    std::string& outError);
