#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

enum GeomType
{
    SPHERE,
    CUBE
};

enum ScenePrimitiveType
{
    SCENE_PRIMITIVE_GEOM,
    SCENE_PRIMITIVE_MESH_INSTANCE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    int objectIndex;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Triangle
{
    glm::vec3 p0;
    glm::vec3 p1;
    glm::vec3 p2;
    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;
    glm::vec3 geometricNormal;
    glm::vec2 uv0;
    glm::vec2 uv1;
    glm::vec2 uv2;
    int materialId;
    int hasVertexNormals;
    int hasUVs;
};

struct TextureData
{
    int width;
    int height;
    int pixelOffset;
    int flipV;
    int wrapS;
    int wrapT;
};

struct TriangleBvhNode
{
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    int leftFirst;
    int rightChild;
    int triCount;
};

struct MeshInstance
{
    int materialId;
    int objectIndex;
    int triangleStart;
    int triangleCount;
    int bvhRootIndex;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    glm::vec3 localBboxMin;
    glm::vec3 localBboxMax;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
};

struct ScenePrimitive
{
    int type;
    int index;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
};

struct SceneBvhNode
{
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    int leftFirst;
    int rightChild;
    int primitiveCount;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    int textureId;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct EnvironmentSettings
{
    int textureId = -1;
    int useProceduralSky = 1;
    float intensity = 1.0f;
    float rotation = 0.0f;
    glm::vec3 zenithColor = glm::vec3(0.55f, 0.72f, 1.0f);
    glm::vec3 horizonColor = glm::vec3(0.95f, 0.96f, 1.0f);
    glm::vec3 groundColor = glm::vec3(0.30f, 0.28f, 0.26f);
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
    EnvironmentSettings environment;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    float lastBsdfPdf;
    int lastBounceWasDelta;
};

struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    glm::vec3 geometricNormal;
    glm::vec2 uv;
    int materialId;
    int geomId;
};
