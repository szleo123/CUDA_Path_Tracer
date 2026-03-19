#pragma once

#include "sceneStructs.h"

#include <string>
#include <vector>

enum class SceneObjectType
{
    Sphere,
    Cube,
    Mesh
};

struct SceneObject
{
    std::string name;
    SceneObjectType type;
    int materialId;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
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

    std::vector<SceneObject> objects;
    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<TriangleBvhNode> triangleBvhNodes;
    std::vector<MeshInstance> meshInstances;
    std::vector<ScenePrimitive> scenePrimitives;
    std::vector<SceneBvhNode> sceneBvhNodes;
    std::vector<Material> materials;
    std::vector<TextureData> textures;
    std::vector<glm::vec3> texturePixels;
    RenderState state;
    bool gpuDynamicDataDirty = true;
};
