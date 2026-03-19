#pragma once

#include "sceneStructs.h"

#include <filesystem>
#include <string>
#include <vector>

struct MeshMaterialDefinition
{
    std::string name;
    glm::vec3 diffuseColor = glm::vec3(1.0f);
    std::filesystem::path diffuseTexturePath;
    std::string diffuseTextureKey;
    std::vector<unsigned char> diffuseTextureBytes;
    int diffuseTexcoordSet = 0;
    int wrapS = 10497;
    int wrapT = 10497;
    bool flipV = true;
};

bool loadTextureImage(
    const std::filesystem::path& texturePath,
    bool flipV,
    TextureData& outTexture,
    std::vector<glm::vec3>& outPixels,
    std::string& outError);

bool loadTextureImageFromMemory(
    const unsigned char* textureBytes,
    size_t textureByteCount,
    bool flipV,
    TextureData& outTexture,
    std::vector<glm::vec3>& outPixels,
    std::string& outError);

bool loadMeshAsset(
    const std::filesystem::path& meshPath,
    int materialId,
    std::vector<Triangle>& outTriangles,
    std::vector<MeshMaterialDefinition>& outMaterials,
    std::string& outError);
