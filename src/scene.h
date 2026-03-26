#pragma once

#include "sceneStructs.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

enum class SceneObjectType
{
    Sphere,
    Cube,
    Water,
    Mesh
};

struct SceneObject
{
    std::string name;
    SceneObjectType type;
    int materialId;
    std::vector<int> usedMaterialIds;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    Geom::WaterSettings water;
    std::string meshPath;
    std::vector<Triangle> localTriangles;
    std::vector<TriangleBvhNode> localBvhNodes;
    glm::vec3 localBboxMin = glm::vec3(0.0f);
    glm::vec3 localBboxMax = glm::vec3(0.0f);
    int triangleStart = -1;
    int triangleCount = 0;
    int bvhRootIndex = -1;
};

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void rebuildStaticMeshData();
    void rebuildRenderData();
public:
    Scene(std::string filename);

    void updateObjectTransform(
        size_t objectIndex,
        const glm::vec3& translation,
        const glm::vec3& rotation,
        const glm::vec3& scale);
    void updateMaterial(
        size_t materialIndex,
        const Material& material);
    void updateWaterSettings(
        size_t objectIndex,
        const Geom::WaterSettings& water);
    bool updateEnvironment(
        const EnvironmentSettings& environment,
        const std::string& hdrPath,
        std::string& outError);

    std::vector<SceneObject> objects;
    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<TriangleBvhNode> triangleBvhNodes;
    std::vector<MeshInstance> meshInstances;
    std::vector<ScenePrimitive> scenePrimitives;
    std::vector<SceneBvhNode> sceneBvhNodes;
    std::vector<Material> materials;
    std::vector<std::string> materialNames;
    std::vector<TextureData> textures;
    std::vector<glm::vec4> texturePixels;
    RenderState state;
    std::filesystem::path sourceScenePath;
    std::string environmentTexturePath;
    std::unordered_map<std::string, uint32_t> texturePathToIdCache;
    bool gpuDynamicDataDirty = true;
};
