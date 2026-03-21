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
    float baseAlpha = 1.0f;
    float metallicFactor = 0.0f;
    float roughnessFactor = 1.0f;
    float indexOfRefraction = 1.5f;
    float alphaCutoff = 0.5f;
    int alphaMode = 0;
    int doubleSided = 0;
    int hasExplicitSpecularColor = 0;
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
    std::filesystem::path emissiveTexturePath;
    std::string emissiveTextureKey;
    std::vector<unsigned char> emissiveTextureBytes;
    int emissiveTexcoordSet = 0;
    std::filesystem::path occlusionTexturePath;
    std::string occlusionTextureKey;
    std::vector<unsigned char> occlusionTextureBytes;
    int occlusionTexcoordSet = 0;
    float normalTextureScale = 1.0f;
    glm::vec3 emissiveFactor = glm::vec3(0.0f);
    float emissiveStrength = 1.0f;
    float transmissionFactor = 0.0f;
    float clearcoatFactor = 0.0f;
    float clearcoatRoughnessFactor = 0.0f;
    float specularFactor = 1.0f;
    glm::vec3 specularFactorColor = glm::vec3(1.0f);
    float occlusionStrength = 1.0f;
    int thinWalled = 0;
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
