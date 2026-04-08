#include "scene.h"

#include "mesh.h"
#include "bvh.h"
#include "utilities.h"
#include "water.h"
#include "volume.h"

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
    if (type == "water")
    {
        return SceneObjectType::Water;
    }
    if (type == "volume")
    {
        return SceneObjectType::Volume;
    }
    if (type == "cloud")
    {
        return SceneObjectType::Cloud;
    }
    if (type == "mesh")
    {
        return SceneObjectType::Mesh;
    }
    return SceneObjectType::Sphere;
}

Geom::WaterSettings defaultWaterSettings()
{
    Geom::WaterSettings water{};
    water.uvScale = glm::vec2(6.0f);
    water.absorptionCoefficient = glm::vec3(0.10f, 0.04f, 0.02f);
    water.foamColor = glm::vec3(0.94f, 0.97f, 1.0f);
    water.shallowColor = glm::vec3(0.36f, 0.74f, 0.72f);
    water.fallbackAbsorptionDistance = 10.0f;
    water.foamIntensity = 0.35f;
    water.foamThreshold = 0.58f;
    water.foamSoftness = 0.18f;
    water.foamRoughness = 0.48f;
    water.shallowColorDistance = 2.5f;
    water.shallowColorStrength = 0.65f;
    water.shorelineFoamDistance = 0.90f;
    water.shorelineFoamIntensity = 0.75f;
    water.infinitePlane = 0;
    water.waveCount = 8;
    water.waves[0].direction = glm::vec2(1.0f, 0.18f);
    water.waves[0].amplitude = 0.050f;
    water.waves[0].wavelength = 1.60f;
    water.waves[0].speed = 0.90f;
    water.waves[0].steepness = 0.30f;
    water.waves[1].direction = glm::vec2(0.82f, 0.35f);
    water.waves[1].amplitude = 0.028f;
    water.waves[1].wavelength = 1.05f;
    water.waves[1].speed = 1.15f;
    water.waves[1].steepness = 0.28f;
    water.waves[2].direction = glm::vec2(-0.20f, 1.0f);
    water.waves[2].amplitude = 0.020f;
    water.waves[2].wavelength = 0.60f;
    water.waves[2].speed = 1.55f;
    water.waves[2].steepness = 0.42f;
    water.waves[3].direction = glm::vec2(0.45f, 1.0f);
    water.waves[3].amplitude = 0.015f;
    water.waves[3].wavelength = 0.44f;
    water.waves[3].speed = 1.95f;
    water.waves[3].steepness = 0.45f;
    water.waves[4].direction = glm::vec2(-0.85f, 0.35f);
    water.waves[4].amplitude = 0.011f;
    water.waves[4].wavelength = 0.34f;
    water.waves[4].speed = 2.20f;
    water.waves[4].steepness = 0.38f;
    water.waves[5].direction = glm::vec2(1.0f, -0.50f);
    water.waves[5].amplitude = 0.007f;
    water.waves[5].wavelength = 0.19f;
    water.waves[5].speed = 2.90f;
    water.waves[5].steepness = 0.50f;
    water.waves[6].direction = glm::vec2(-0.55f, -1.0f);
    water.waves[6].amplitude = 0.005f;
    water.waves[6].wavelength = 0.14f;
    water.waves[6].speed = 3.60f;
    water.waves[6].steepness = 0.52f;
    water.waves[7].direction = glm::vec2(0.18f, -1.0f);
    water.waves[7].amplitude = 0.004f;
    water.waves[7].wavelength = 0.10f;
    water.waves[7].speed = 4.20f;
    water.waves[7].steepness = 0.48f;
    water.maxVerticalDisplacement = computeWaterMaxVerticalDisplacement(water);
    return water;
}

Geom::VolumeSettings defaultVolumeSettings()
{
    Geom::VolumeSettings volume{};
    volume.model = Geom::VOLUME_MODEL_SDF;
    volume.sdfResolution = 64;
    volume.sdfPadding = 0.15f;
    return volume;
}

Geom::VolumeSettings defaultCloudSettings()
{
    Geom::VolumeSettings volume{};
    volume.model = Geom::VOLUME_MODEL_CLOUD;
    volume.albedo = glm::vec3(0.98f, 0.99f, 1.0f);
    volume.windDirection = glm::vec3(1.0f, 0.0f, 0.2f);
    volume.densityMultiplier = 0.95f;
    volume.noiseScale = 1.45f;
    volume.detailNoiseScale = 6.5f;
    volume.densityThreshold = 0.42f;
    volume.densitySoftness = 0.16f;
    volume.stepSize = 0.24f;
    volume.shadowStepSize = 0.30f;
    volume.phaseAnisotropy = 0.55f;
    volume.ambientIntensity = 0.10f;
    volume.windSpeed = 0.02f;
    volume.coverage = 0.52f;
    volume.bottomFade = 0.16f;
    volume.topFade = 0.28f;
    volume.erosionStrength = 0.22f;
    volume.detailErosionStrength = 0.40f;
    volume.sdfPadding = 0.0f;
    volume.sdfResolution = 0;
    return volume;
}

Geom::WaterSettings::Wave parseWaterWaveDefinition(const json& waveJson)
{
    Geom::WaterSettings::Wave wave{};
    if (waveJson.contains("DIR"))
    {
        wave.direction = glm::vec2(waveJson["DIR"][0], waveJson["DIR"][1]);
    }
    wave.amplitude = waveJson.value("AMPLITUDE", 0.0f);
    wave.wavelength = glm::max(waveJson.value("WAVELENGTH", 1.0f), 0.001f);
    wave.speed = waveJson.value("SPEED", 1.0f);
    wave.steepness = glm::clamp(waveJson.value("STEEPNESS", 0.0f), 0.0f, 1.5f);
    return wave;
}

Geom::WaterSettings parseWaterSettings(const json& objectJson)
{
    Geom::WaterSettings water = defaultWaterSettings();
    if (!objectJson.contains("WATER"))
    {
        return water;
    }

    const json& waterJson = objectJson["WATER"];
    if (waterJson.contains("UV_SCALE"))
    {
        water.uvScale = glm::vec2(waterJson["UV_SCALE"][0], waterJson["UV_SCALE"][1]);
    }
    if (waterJson.contains("ABSORPTION_COEFF"))
    {
        water.absorptionCoefficient = glm::vec3(
            waterJson["ABSORPTION_COEFF"][0],
            waterJson["ABSORPTION_COEFF"][1],
            waterJson["ABSORPTION_COEFF"][2]);
    }
    if (waterJson.contains("FOAM_COLOR"))
    {
        water.foamColor = glm::vec3(
            waterJson["FOAM_COLOR"][0],
            waterJson["FOAM_COLOR"][1],
            waterJson["FOAM_COLOR"][2]);
    }
    if (waterJson.contains("SHALLOW_COLOR"))
    {
        water.shallowColor = glm::vec3(
            waterJson["SHALLOW_COLOR"][0],
            waterJson["SHALLOW_COLOR"][1],
            waterJson["SHALLOW_COLOR"][2]);
    }
    water.fallbackAbsorptionDistance = glm::max(
        waterJson.value("FALLBACK_ABSORPTION_DISTANCE", water.fallbackAbsorptionDistance),
        0.0f);
    water.foamIntensity = glm::max(
        waterJson.value("FOAM_INTENSITY", water.foamIntensity),
        0.0f);
    water.foamThreshold = glm::clamp(
        waterJson.value("FOAM_THRESHOLD", water.foamThreshold),
        0.0f,
        2.0f);
    water.foamSoftness = glm::max(
        waterJson.value("FOAM_SOFTNESS", water.foamSoftness),
        0.001f);
    water.foamRoughness = glm::clamp(
        waterJson.value("FOAM_ROUGHNESS", water.foamRoughness),
        0.0f,
        1.0f);
    water.shallowColorDistance = glm::max(
        waterJson.value("SHALLOW_COLOR_DISTANCE", water.shallowColorDistance),
        0.0f);
    water.shallowColorStrength = glm::clamp(
        waterJson.value("SHALLOW_COLOR_STRENGTH", water.shallowColorStrength),
        0.0f,
        1.0f);
    water.shorelineFoamDistance = glm::max(
        waterJson.value("SHORELINE_FOAM_DISTANCE", water.shorelineFoamDistance),
        0.0f);
    water.shorelineFoamIntensity = glm::max(
        waterJson.value("SHORELINE_FOAM_INTENSITY", water.shorelineFoamIntensity),
        0.0f);
    water.infinitePlane = waterJson.value("INFINITE", water.infinitePlane != 0) ? 1 : 0;

    if (waterJson.contains("WAVES") && waterJson["WAVES"].is_array())
    {
        const size_t parsedWaveCount = std::min(
            waterJson["WAVES"].size(),
            static_cast<size_t>(RENDER_CONFIG_MAX_GERSTNER_WAVES));
        water.waveCount = static_cast<int>(parsedWaveCount);
        for (size_t waveIndex = 0; waveIndex < parsedWaveCount; ++waveIndex)
        {
            water.waves[waveIndex] = parseWaterWaveDefinition(waterJson["WAVES"][waveIndex]);
        }
        for (size_t waveIndex = parsedWaveCount; waveIndex < RENDER_CONFIG_MAX_GERSTNER_WAVES; ++waveIndex)
        {
            water.waves[waveIndex] = Geom::WaterSettings::Wave{};
        }
    }

    water.maxVerticalDisplacement = computeWaterMaxVerticalDisplacement(water);
    return water;
}

Geom::VolumeSettings parseVolumeSettings(const json& objectJson, SceneObjectType objectType)
{
    Geom::VolumeSettings volume = objectType == SceneObjectType::Cloud
        ? defaultCloudSettings()
        : defaultVolumeSettings();
    if (!objectJson.contains("VOLUME"))
    {
        return volume;
    }

    const json& volumeJson = objectJson["VOLUME"];
    if (volumeJson.contains("ALBEDO"))
    {
        volume.albedo = glm::vec3(
            volumeJson["ALBEDO"][0],
            volumeJson["ALBEDO"][1],
            volumeJson["ALBEDO"][2]);
    }
    if (volumeJson.contains("WIND_DIRECTION"))
    {
        volume.windDirection = glm::vec3(
            volumeJson["WIND_DIRECTION"][0],
            volumeJson["WIND_DIRECTION"][1],
            volumeJson["WIND_DIRECTION"][2]);
    }
    volume.densityMultiplier = glm::max(volumeJson.value("DENSITY", volume.densityMultiplier), 0.0f);
    const float legacyExtinction = glm::max(volumeJson.value("EXTINCTION", 1.0f), 0.0f);
    volume.densityMultiplier *= legacyExtinction;
    volume.noiseScale = glm::max(volumeJson.value("NOISE_SCALE", volume.noiseScale), 0.01f);
    volume.detailNoiseScale = glm::max(volumeJson.value("DETAIL_NOISE_SCALE", volume.detailNoiseScale), 0.01f);
    volume.densityThreshold = glm::clamp(volumeJson.value("DENSITY_THRESHOLD", volume.densityThreshold), 0.0f, 1.0f);
    volume.densitySoftness = glm::max(volumeJson.value("DENSITY_SOFTNESS", volume.densitySoftness), 0.001f);
    volume.stepSize = glm::max(volumeJson.value("STEP_SIZE", volume.stepSize), 0.005f);
    volume.shadowStepSize = glm::max(volumeJson.value("SHADOW_STEP_SIZE", volume.shadowStepSize), 0.005f);
    volume.phaseAnisotropy = glm::clamp(volumeJson.value("PHASE_G", volume.phaseAnisotropy), -0.95f, 0.95f);
    volume.ambientIntensity = glm::max(volumeJson.value("AMBIENT_INTENSITY", volume.ambientIntensity), 0.0f);
    volume.windSpeed = volumeJson.value("WIND_SPEED", volume.windSpeed);
    volume.coverage = glm::clamp(volumeJson.value("COVERAGE", volume.coverage), 0.0f, 1.0f);
    volume.bottomFade = glm::clamp(volumeJson.value("BOTTOM_FADE", volume.bottomFade), 0.001f, 0.95f);
    volume.topFade = glm::clamp(volumeJson.value("TOP_FADE", volume.topFade), 0.001f, 0.95f);
    volume.erosionStrength = glm::clamp(volumeJson.value("EROSION_STRENGTH", volume.erosionStrength), 0.0f, 1.0f);
    volume.detailErosionStrength = glm::clamp(volumeJson.value("DETAIL_EROSION_STRENGTH", volume.detailErosionStrength), 0.0f, 1.0f);
    volume.sdfPadding = glm::max(volumeJson.value("SDF_PADDING", volume.sdfPadding), 0.0f);
    volume.sdfResolution = glm::max(volumeJson.value("SDF_RESOLUTION", volume.sdfResolution), 8);

    volume.albedo = glm::clamp(volume.albedo, glm::vec3(0.0f), glm::vec3(1.0f));
    if (glm::dot(volume.windDirection, volume.windDirection) <= EPSILON)
    {
        volume.windDirection = glm::vec3(1.0f, 0.0f, 0.25f);
    }

    return volume;
}

float pointAabbDistanceSquaredHost(
    const glm::vec3& point,
    const glm::vec3& bboxMin,
    const glm::vec3& bboxMax)
{
    const glm::vec3 clampedPoint = glm::clamp(point, bboxMin, bboxMax);
    return glm::dot(point - clampedPoint, point - clampedPoint);
}

float pointTriangleDistanceSquaredHost(
    const glm::vec3& point,
    const Triangle& triangle)
{
    const glm::vec3 a = triangle.p0;
    const glm::vec3 b = triangle.p1;
    const glm::vec3 c = triangle.p2;
    const glm::vec3 ab = b - a;
    const glm::vec3 ac = c - a;
    const glm::vec3 ap = point - a;

    const float d1 = glm::dot(ab, ap);
    const float d2 = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        return glm::dot(point - a, point - a);
    }

    const glm::vec3 bp = point - b;
    const float d3 = glm::dot(ab, bp);
    const float d4 = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        return glm::dot(point - b, point - b);
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / std::max(d1 - d3, EPSILON);
        const glm::vec3 projection = a + v * ab;
        return glm::dot(point - projection, point - projection);
    }

    const glm::vec3 cp = point - c;
    const float d5 = glm::dot(ab, cp);
    const float d6 = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        return glm::dot(point - c, point - c);
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / std::max(d2 - d6, EPSILON);
        const glm::vec3 projection = a + w * ac;
        return glm::dot(point - projection, point - projection);
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const glm::vec3 bc = c - b;
        const float w = (d4 - d3) / std::max((d4 - d3) + (d5 - d6), EPSILON);
        const glm::vec3 projection = b + w * bc;
        return glm::dot(point - projection, point - projection);
    }

    const glm::vec3 normal = glm::normalize(glm::cross(ab, ac));
    const float distanceToPlane = glm::dot(point - a, normal);
    return distanceToPlane * distanceToPlane;
}

float queryNearestDistanceToTrianglesHost(
    const glm::vec3& point,
    const std::vector<Triangle>& triangles,
    const std::vector<TriangleBvhNode>& nodes)
{
    if (triangles.empty() || nodes.empty())
    {
        return FLT_MAX;
    }

    int stack[RENDER_CONFIG_MAX_PICKING_BVH_STACK_SIZE];
    int stackSize = 0;
    stack[stackSize++] = 0;
    float bestDistanceSquared = FLT_MAX;

    while (stackSize > 0)
    {
        const TriangleBvhNode& node = nodes[stack[--stackSize]];
        const float nodeDistanceSquared = pointAabbDistanceSquaredHost(point, node.bboxMin, node.bboxMax);
        if (nodeDistanceSquared >= bestDistanceSquared)
        {
            continue;
        }

        if (node.triCount > 0)
        {
            for (int i = 0; i < node.triCount; ++i)
            {
                const int triangleIndex = node.leftFirst + i;
                bestDistanceSquared = std::min(
                    bestDistanceSquared,
                    pointTriangleDistanceSquaredHost(point, triangles[triangleIndex]));
            }
            continue;
        }

        if (stackSize + 2 <= RENDER_CONFIG_MAX_PICKING_BVH_STACK_SIZE)
        {
            const int leftChild = node.leftFirst;
            const int rightChild = node.rightChild;
            const float leftDistanceSquared = pointAabbDistanceSquaredHost(point, nodes[leftChild].bboxMin, nodes[leftChild].bboxMax);
            const float rightDistanceSquared = pointAabbDistanceSquaredHost(point, nodes[rightChild].bboxMin, nodes[rightChild].bboxMax);
            if (leftDistanceSquared < rightDistanceSquared)
            {
                stack[stackSize++] = rightChild;
                stack[stackSize++] = leftChild;
            }
            else
            {
                stack[stackSize++] = leftChild;
                stack[stackSize++] = rightChild;
            }
        }
    }

    return sqrtf(bestDistanceSquared);
}

bool rayIntersectsTriangleHost(
    const Triangle& triangle,
    const Ray& ray,
    float& outT)
{
    const glm::vec3 edge1 = triangle.p1 - triangle.p0;
    const glm::vec3 edge2 = triangle.p2 - triangle.p0;
    const glm::vec3 pvec = glm::cross(ray.direction, edge2);
    const float det = glm::dot(edge1, pvec);
    if (fabsf(det) < RENDER_CONFIG_TRIANGLE_DET_EPSILON)
    {
        return false;
    }

    const float invDet = 1.0f / det;
    const glm::vec3 tvec = ray.origin - triangle.p0;
    const float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }

    const glm::vec3 qvec = glm::cross(tvec, edge1);
    const float v = glm::dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f)
    {
        return false;
    }

    outT = glm::dot(edge2, qvec) * invDet;
    return outT > RENDER_CONFIG_TRIANGLE_MIN_INTERSECTION_T;
}

bool intersectAabbHost(
    const Ray& ray,
    const glm::vec3& bboxMin,
    const glm::vec3& bboxMax,
    float& outEntry)
{
    float tMin = 0.0f;
    float tFar = FLT_MAX;

    for (int axis = 0; axis < 3; ++axis)
    {
        const float origin = ray.origin[axis];
        const float direction = ray.direction[axis];
        if (fabsf(direction) < EPSILON)
        {
            if (origin < bboxMin[axis] || origin > bboxMax[axis])
            {
                return false;
            }
            continue;
        }

        const float invDir = 1.0f / direction;
        float t0 = (bboxMin[axis] - origin) * invDir;
        float t1 = (bboxMax[axis] - origin) * invDir;
        if (t0 > t1)
        {
            std::swap(t0, t1);
        }

        tMin = std::max(tMin, t0);
        tFar = std::min(tFar, t1);
        if (tMin > tFar)
        {
            return false;
        }
    }

    outEntry = tMin;
    return true;
}

int countRayMeshIntersectionsHost(
    const Ray& ray,
    const std::vector<Triangle>& triangles,
    const std::vector<TriangleBvhNode>& nodes)
{
    if (triangles.empty() || nodes.empty())
    {
        return 0;
    }

    int stack[RENDER_CONFIG_MAX_PICKING_BVH_STACK_SIZE];
    int stackSize = 0;
    stack[stackSize++] = 0;
    int crossings = 0;

    while (stackSize > 0)
    {
        const TriangleBvhNode& node = nodes[stack[--stackSize]];
        float nodeEntry = 0.0f;
        if (!intersectAabbHost(ray, node.bboxMin, node.bboxMax, nodeEntry))
        {
            continue;
        }

        if (node.triCount > 0)
        {
            for (int i = 0; i < node.triCount; ++i)
            {
                float t = 0.0f;
                if (rayIntersectsTriangleHost(triangles[node.leftFirst + i], ray, t))
                {
                    ++crossings;
                }
            }
            continue;
        }

        if (stackSize + 2 <= RENDER_CONFIG_MAX_PICKING_BVH_STACK_SIZE)
        {
            stack[stackSize++] = node.rightChild;
            stack[stackSize++] = node.leftFirst;
        }
    }

    return crossings;
}

bool isPointInsideMeshHost(
    const glm::vec3& point,
    const std::vector<Triangle>& triangles,
    const std::vector<TriangleBvhNode>& nodes)
{
    Ray ray{};
    ray.origin = point + glm::normalize(glm::vec3(0.617f, 0.441f, 0.652f)) * (RAY_ORIGIN_BIAS * 8.0f);
    ray.direction = glm::normalize(glm::vec3(0.617f, 0.441f, 0.652f));
    return (countRayMeshIntersectionsHost(ray, triangles, nodes) & 1) == 1;
}

void buildVolumeMeshSdf(SceneObject& object)
{
    object.volumeSdfValues.clear();
    object.volumeSdfResolution = 0;
    object.volumeSdfBoundsMin = object.localBboxMin;
    object.volumeSdfBoundsMax = object.localBboxMax;

    if (object.localTriangles.empty() || object.localBvhNodes.empty())
    {
        return;
    }

    const int resolution = std::max(object.volume.sdfResolution, 8);
    const glm::vec3 localExtent = object.localBboxMax - object.localBboxMin;
    const float maxExtent = std::max(localExtent.x, std::max(localExtent.y, localExtent.z));
    const float padding = std::max(object.volume.sdfPadding * std::max(maxExtent, 0.001f), 0.01f);
    object.volumeSdfBoundsMin = object.localBboxMin - glm::vec3(padding);
    object.volumeSdfBoundsMax = object.localBboxMax + glm::vec3(padding);
    object.volumeSdfResolution = resolution;
    object.volumeSdfValues.resize(static_cast<size_t>(resolution) * resolution * resolution, 0.0f);

    const glm::vec3 sdfExtent = object.volumeSdfBoundsMax - object.volumeSdfBoundsMin;
    for (int z = 0; z < resolution; ++z)
    {
        for (int y = 0; y < resolution; ++y)
        {
            for (int x = 0; x < resolution; ++x)
            {
                const glm::vec3 uvw(
                    (static_cast<float>(x) + 0.5f) / static_cast<float>(resolution),
                    (static_cast<float>(y) + 0.5f) / static_cast<float>(resolution),
                    (static_cast<float>(z) + 0.5f) / static_cast<float>(resolution));
                const glm::vec3 samplePoint = object.volumeSdfBoundsMin + uvw * sdfExtent;
                const float distance = queryNearestDistanceToTrianglesHost(samplePoint, object.localTriangles, object.localBvhNodes);
                const bool inside = isPointInsideMeshHost(samplePoint, object.localTriangles, object.localBvhNodes);
                const float signedDistance = inside ? -distance : distance;
                const size_t linearIndex =
                    static_cast<size_t>(x)
                    + static_cast<size_t>(y) * static_cast<size_t>(resolution)
                    + static_cast<size_t>(z) * static_cast<size_t>(resolution) * static_cast<size_t>(resolution);
                object.volumeSdfValues[linearIndex] = signedDistance;
            }
        }
    }
}

Geom buildGeomFromObject(const SceneObject& object, int objectIndex)
{
    Geom geom{};
    if (object.type == SceneObjectType::Cube)
    {
        geom.type = CUBE;
    }
    else if (object.type == SceneObjectType::Water)
    {
        geom.type = WATER_PLANE;
    }
    else if (object.type == SceneObjectType::Volume || object.type == SceneObjectType::Cloud)
    {
        geom.type = VOLUME;
    }
    else
    {
        geom.type = SPHERE;
    }
    geom.materialid = object.materialId;
    geom.objectIndex = objectIndex;
    geom.translation = object.translation;
    geom.rotation = object.rotation;
    geom.scale = object.scale;
    geom.transform = utilityCore::buildTransformationMatrix(object.translation, object.rotation, object.scale);
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::inverseTranspose(geom.transform);
    geom.water = object.water;
    geom.volume = object.volume;
    geom.volumeMeshLocalBboxMin = object.localBboxMin;
    geom.volumeMeshLocalBboxMax = object.localBboxMax;
    geom.volumeSdfOffset = -1;
    geom.volumeSdfResolution = object.volumeSdfResolution;
    geom.volumeSdfBoundsMin = object.volumeSdfBoundsMin;
    geom.volumeSdfBoundsMax = object.volumeSdfBoundsMax;
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
    glm::vec3 localMin(-0.5f);
    glm::vec3 localMax(0.5f);
    if (geom.type == WATER_PLANE)
    {
        getWaterLocalBounds(geom.water, localMin, localMax);
    }
    else if (geom.type == VOLUME)
    {
        getVolumeLocalBounds(geom, localMin, localMax);
    }

    ScenePrimitive primitive{};
    primitive.type = SCENE_PRIMITIVE_GEOM;
    primitive.index = geomIndex;
    computeTransformedBounds(
        geom.transform,
        localMin,
        localMax,
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
    case SceneObjectType::Water:
        return "Water " + std::to_string(index);
    case SceneObjectType::Volume:
        return "Volume " + std::to_string(index);
    case SceneObjectType::Cloud:
        return "Cloud " + std::to_string(index);
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
    const std::filesystem::path& meshPath,
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
        ? (meshPath.string() + "|embedded|" + embeddedTextureKey)
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
    material.ambientOcclusion = 1.0f;
    const glm::vec3 specularFactorColorDelta = importedMaterial.specularFactorColor - glm::vec3(1.0f);
    const bool importedMaterialHasReflectiveSignals =
        importedMaterial.metallicFactor > EPSILON
        || !importedMaterial.metallicRoughnessTextureBytes.empty()
        || !importedMaterial.metallicRoughnessTexturePath.empty()
        || importedMaterial.hasExplicitSpecularColor != 0
        || fabsf(importedMaterial.specularFactor - 1.0f) > EPSILON
        || glm::dot(specularFactorColorDelta, specularFactorColorDelta) > EPSILON
        || importedMaterial.transmissionFactor > EPSILON
        || importedMaterial.clearcoatFactor > EPSILON;
    const float importedDielectricReflectivity = glm::clamp(
        glm::max(
            glm::max(material.specularColor.r, material.specularColor.g),
            material.specularColor.b) * 12.0f,
        0.0f,
        1.0f);
    material.hasReflective = fmaxf(
        material.hasReflective,
        importedMaterialHasReflectiveSignals
            ? fmaxf(importedMaterial.metallicFactor, importedDielectricReflectivity)
            : importedMaterial.metallicFactor);
    material.hasRefractive = fmaxf(material.hasRefractive, importedMaterial.transmissionFactor);
    if (glm::length(importedMaterial.emissiveFactor) > 0.0f
        || !importedMaterial.emissiveTextureBytes.empty()
        || !importedMaterial.emissiveTexturePath.empty())
    {
        material.emittance = importedMaterial.emissiveStrength;
    }
    material.textureId = ensureImportedTextureLoaded(
        meshPath,
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
        meshPath,
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
        meshPath,
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
        meshPath,
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
        meshPath,
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
    material.ambientOcclusion = 1.0f;

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
        material.hasReflective = materialJson.value("REFLECTIVITY", material.hasReflective);
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

void initializeVolumeMeshBoundaryObject(
    SceneObject& object,
    const SceneImportContext& importContext)
{
    const std::filesystem::path meshPath = resolveScenePath(importContext.scenePath, object.meshPath);
    object.meshPath = meshPath.string();

    std::vector<MeshMaterialDefinition> ignoredImportedMaterials;
    std::string error;
    if (!loadMeshAsset(meshPath, object.materialId, object.localTriangles, ignoredImportedMaterials, error))
    {
        throw std::runtime_error(error);
    }

    for (Triangle& triangle : object.localTriangles)
    {
        triangle.materialId = object.materialId;
    }
    object.usedMaterialIds.clear();
    object.usedMaterialIds.push_back(object.materialId);

    std::vector<Triangle> localTriangles = object.localTriangles;
    if (!buildTriangleBvh(localTriangles, object.localBvhNodes) || object.localBvhNodes.empty())
    {
        throw std::runtime_error("Failed to build volume mesh BVH: " + meshPath.string());
    }
    object.localTriangles.swap(localTriangles);
    object.localBboxMin = object.localBvhNodes[0].bboxMin;
    object.localBboxMax = object.localBvhNodes[0].bboxMax;
    buildVolumeMeshSdf(object);

    std::cout
        << "Loaded volume mesh '" << object.name << "' from " << meshPath.string()
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

    if (object.type == SceneObjectType::Water)
    {
        object.water = parseWaterSettings(objectJson);
    }
    else if (object.type == SceneObjectType::Volume || object.type == SceneObjectType::Cloud)
    {
        object.volume = parseVolumeSettings(objectJson, object.type);
        object.initialVolume = object.volume;
    }

    if (objectJson.contains("FILE"))
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
        else if (object.type == SceneObjectType::Volume)
        {
            if (object.meshPath.empty())
            {
                throw std::runtime_error("SDF volume requires FILE: " + object.name);
            }
            initializeVolumeMeshBoundaryObject(object, importContext);
        }
        else if (object.type == SceneObjectType::Cloud)
        {
            object.usedMaterialIds.clear();
            object.usedMaterialIds.push_back(object.materialId);
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

void configureAnimationFromJson(const json& data, RenderState& renderState)
{
    renderState.sceneTimeSeconds = 0.0f;
    renderState.frameDeltaTimeSeconds = 0.0f;
    renderState.stepDeltaTimeSeconds = 1.0f / 60.0f;
    renderState.playbackSpeed = 1.0f;
    renderState.playAnimation = 0;

    if (!data.contains("Animation"))
    {
        return;
    }

    const json& animationData = data["Animation"];
    renderState.sceneTimeSeconds = animationData.value("TIME", 0.0f);
    renderState.stepDeltaTimeSeconds = glm::max(animationData.value("STEP_DT", 1.0f / 60.0f), 0.0f);
    renderState.playbackSpeed = glm::max(animationData.value("SPEED", 1.0f), 0.0f);
    renderState.playAnimation = animationData.value("PLAY", false) ? 1 : 0;
}

std::string configureEnvironmentFromJson(
    const json& data,
    const SceneImportContext& importContext,
    RenderState& renderState)
{
    EnvironmentSettings environment{};
    environment.mode = ENVIRONMENT_NONE;
    environment.textureId = -1;
    std::string environmentTexturePath;

    if (data.contains("Environment"))
    {
        const json& environmentData = data["Environment"];
        environment.intensity = environmentData.value("INTENSITY", 1.0f);
        environment.rotation = environmentData.value("ROTATION", 0.0f);
        environment.rotationSpeed = environmentData.value("ROTATION_SPEED", 0.0f);
        environment.rotateCounterClockwise = environmentData.value("COUNTER_CLOCKWISE", false) ? 1 : 0;

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
            if (type == "HDR")
            {
                environment.mode = ENVIRONMENT_HDR;
            }
            else if (type == "NONE")
            {
                environment.mode = ENVIRONMENT_NONE;
            }
            else
            {
                environment.mode = ENVIRONMENT_PROCEDURAL_SKY;
            }
        }
        if (environmentData.contains("FILE"))
        {
            const std::filesystem::path hdrPath = resolveScenePath(importContext.scenePath, environmentData["FILE"]);
            environment.textureId = ensureHdrTextureLoaded(
                hdrPath,
                importContext.texturePathToId,
                importContext.textures,
                importContext.texturePixels);
            environment.mode = ENVIRONMENT_HDR;
            environmentTexturePath = hdrPath.string();
        }
    }

    renderState.environment = environment;
    if (environment.mode != ENVIRONMENT_HDR)
    {
        renderState.environment.textureId = -1;
    }
    return environmentTexturePath;
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
    volumeSdfData.clear();

    for (size_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex)
    {
        SceneObject& object = objects[objectIndex];
        if (object.type == SceneObjectType::Mesh)
        {
            appendMeshInstance(object, static_cast<int>(objectIndex), meshInstances, scenePrimitives);
            continue;
        }

        if (object.type == SceneObjectType::Volume)
        {
            object.volumeSdfResolution = object.volumeSdfResolution > 0 ? object.volumeSdfResolution : std::max(object.volume.sdfResolution, 8);
            if (!object.volumeSdfValues.empty())
            {
                const int sdfOffset = static_cast<int>(volumeSdfData.size());
                volumeSdfData.insert(volumeSdfData.end(), object.volumeSdfValues.begin(), object.volumeSdfValues.end());
                const int geomIndex = static_cast<int>(geoms.size());
                geoms.push_back(buildGeomFromObject(object, static_cast<int>(objectIndex)));
                geoms.back().volumeSdfOffset = sdfOffset;
                appendGeomPrimitive(geoms.back(), geomIndex, scenePrimitives);
                continue;
            }
        }
        else if (object.type == SceneObjectType::Cloud)
        {
            const int geomIndex = static_cast<int>(geoms.size());
            geoms.push_back(buildGeomFromObject(object, static_cast<int>(objectIndex)));
            appendGeomPrimitive(geoms.back(), geomIndex, scenePrimitives);
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

void Scene::updateWaterSettings(
    size_t objectIndex,
    const Geom::WaterSettings& water)
{
    if (objectIndex >= objects.size() || objects[objectIndex].type != SceneObjectType::Water)
    {
        return;
    }

    objects[objectIndex].water = water;
    objects[objectIndex].water.maxVerticalDisplacement = computeWaterMaxVerticalDisplacement(objects[objectIndex].water);
    rebuildRenderData();
}

void Scene::updateVolumeSettings(
    size_t objectIndex,
    const Geom::VolumeSettings& volume)
{
    if (objectIndex >= objects.size()
        || (objects[objectIndex].type != SceneObjectType::Volume
            && objects[objectIndex].type != SceneObjectType::Cloud))
    {
        return;
    }

    Geom::VolumeSettings& current = objects[objectIndex].volume;
    const bool sdfSettingsChanged =
        current.model != volume.model ||
        current.sdfResolution != volume.sdfResolution
        || fabsf(current.sdfPadding - volume.sdfPadding) > 1.0e-6f;

    current = volume;
    if (objects[objectIndex].type == SceneObjectType::Volume
        && sdfSettingsChanged
        && objects[objectIndex].localTriangles.empty()
        && !objects[objectIndex].meshPath.empty())
    {
        std::unordered_map<std::string, uint32_t> temporaryMaterialMap;
        std::unordered_map<std::string, uint32_t> temporaryImportedMaterialMap;
        SceneImportContext importContext{
            sourceScenePath,
            temporaryMaterialMap,
            texturePathToIdCache,
            temporaryImportedMaterialMap,
            materials,
            materialNames,
            textures,
            texturePixels
        };
        initializeVolumeMeshBoundaryObject(objects[objectIndex], importContext);
    }
    else if (sdfSettingsChanged && !objects[objectIndex].localTriangles.empty())
    {
        buildVolumeMeshSdf(objects[objectIndex]);
    }
    rebuildRenderData();
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    sourceScenePath = std::filesystem::absolute(jsonName);
    std::ifstream f(sourceScenePath);
    if (!f)
    {
        throw std::runtime_error("Failed to open scene file: " + sourceScenePath.string());
    }

    json data = json::parse(f);
    std::unordered_map<std::string, uint32_t> materialNameToId;
    std::unordered_map<std::string, uint32_t> texturePathToId;
    std::unordered_map<std::string, uint32_t> importedMaterialKeyToId;
    SceneImportContext importContext{
        sourceScenePath,
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
    texturePathToIdCache = texturePathToId;

    rebuildStaticMeshData();
    rebuildRenderData();
    configureCameraFromJson(data["Camera"], state);
    configureAnimationFromJson(data, state);
    environmentTexturePath = configureEnvironmentFromJson(data, importContext, state);
}

bool Scene::updateEnvironment(
    const EnvironmentSettings& environment,
    const std::string& hdrPath,
    std::string& outError)
{
    EnvironmentSettings updatedEnvironment = environment;

    if (updatedEnvironment.mode == ENVIRONMENT_HDR)
    {
        std::string resolvedPathString = hdrPath.empty() ? environmentTexturePath : hdrPath;
        if (resolvedPathString.empty())
        {
            outError = "HDR environment file path is empty.";
            return false;
        }

        try
        {
            std::filesystem::path resolvedPath(resolvedPathString);
            if (resolvedPath.is_relative())
            {
                const std::filesystem::path basePath = sourceScenePath.empty()
                    ? std::filesystem::current_path()
                    : sourceScenePath.parent_path();
                resolvedPath = basePath / resolvedPath;
            }

            resolvedPath = std::filesystem::weakly_canonical(resolvedPath);
            updatedEnvironment.textureId = ensureHdrTextureLoaded(
                resolvedPath,
                texturePathToIdCache,
                textures,
                texturePixels);
            environmentTexturePath = resolvedPath.string();
        }
        catch (const std::exception& e)
        {
            outError = e.what();
            return false;
        }
    }
    else
    {
        updatedEnvironment.textureId = -1;
    }

    state.environment = updatedEnvironment;
    gpuDynamicDataDirty = true;
    return true;
}

