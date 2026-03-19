#pragma once

#include "sceneStructs.h"

#include <filesystem>
#include <string>
#include <vector>

bool loadTextureImage(
    const std::filesystem::path& texturePath,
    TextureData& outTexture,
    std::vector<glm::vec3>& outPixels,
    std::string& outError);

bool loadObjMesh(
    const std::filesystem::path& meshPath,
    const glm::mat4& transform,
    const glm::mat4& invTranspose,
    int materialId,
    std::vector<Triangle>& outTriangles,
    std::string& outError);
