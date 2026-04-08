#pragma once

#include "sceneStructs.h"
#include "utilities.h"

#include <cfloat>
#include <cmath>

CUDA_INLINE glm::vec3 volumeMultiplyMV(const glm::mat4& m, const glm::vec4& v)
{
    return glm::vec3(m * v);
}

CUDA_INLINE float volumeHash13(const glm::vec3& p)
{
    const float s = sinf(glm::dot(p, glm::vec3(127.1f, 311.7f, 74.7f))) * 43758.5453f;
    return s - floorf(s);
}

CUDA_INLINE float volumeValueNoise3D(const glm::vec3& p)
{
    const glm::vec3 cell = glm::floor(p);
    const glm::vec3 local = p - cell;
    const glm::vec3 smooth = local * local * (glm::vec3(3.0f) - 2.0f * local);

    const float n000 = volumeHash13(cell + glm::vec3(0.0f, 0.0f, 0.0f));
    const float n100 = volumeHash13(cell + glm::vec3(1.0f, 0.0f, 0.0f));
    const float n010 = volumeHash13(cell + glm::vec3(0.0f, 1.0f, 0.0f));
    const float n110 = volumeHash13(cell + glm::vec3(1.0f, 1.0f, 0.0f));
    const float n001 = volumeHash13(cell + glm::vec3(0.0f, 0.0f, 1.0f));
    const float n101 = volumeHash13(cell + glm::vec3(1.0f, 0.0f, 1.0f));
    const float n011 = volumeHash13(cell + glm::vec3(0.0f, 1.0f, 1.0f));
    const float n111 = volumeHash13(cell + glm::vec3(1.0f, 1.0f, 1.0f));

    const float nx00 = glm::mix(n000, n100, smooth.x);
    const float nx10 = glm::mix(n010, n110, smooth.x);
    const float nx01 = glm::mix(n001, n101, smooth.x);
    const float nx11 = glm::mix(n011, n111, smooth.x);
    const float nxy0 = glm::mix(nx00, nx10, smooth.y);
    const float nxy1 = glm::mix(nx01, nx11, smooth.y);
    return glm::mix(nxy0, nxy1, smooth.z);
}

CUDA_INLINE float volumeFbm(glm::vec3 p)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    for (int octave = 0; octave < 4; ++octave)
    {
        sum += amplitude * volumeValueNoise3D(p * frequency);
        p = glm::mat3(
            1.6f, -1.2f, 0.4f,
            1.1f, 1.3f, -0.6f,
            -0.5f, 0.8f, 1.7f) * p;
        frequency *= 1.85f;
        amplitude *= 0.5f;
    }
    return glm::clamp(sum / 0.9375f, 0.0f, 1.0f);
}

CUDA_INLINE float volumeSmoothstep(float edge0, float edge1, float x)
{
    const float denom = fmaxf(edge1 - edge0, EPSILON);
    const float t = glm::clamp((x - edge0) / denom, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

CUDA_INLINE void getVolumeLocalBounds(
    const Geom::VolumeSettings& volume,
    const glm::vec3& meshBboxMin,
    const glm::vec3& meshBboxMax,
    glm::vec3& outMin,
    glm::vec3& outMax)
{
    if (volume.model == Geom::VOLUME_MODEL_CLOUD)
    {
        outMin = glm::vec3(-0.5f);
        outMax = glm::vec3(0.5f);
        return;
    }
    outMin = meshBboxMin;
    outMax = meshBboxMax;
}

CUDA_INLINE void getVolumeLocalBounds(const Geom& volumeGeom, glm::vec3& outMin, glm::vec3& outMax)
{
    if (volumeGeom.volumeSdfResolution > 0)
    {
        outMin = volumeGeom.volumeSdfBoundsMin;
        outMax = volumeGeom.volumeSdfBoundsMax;
        return;
    }
    getVolumeLocalBounds(
        volumeGeom.volume,
        volumeGeom.volumeMeshLocalBboxMin,
        volumeGeom.volumeMeshLocalBboxMax,
        outMin,
        outMax);
}

 CUDA_INLINE bool intersectVolumeAnalyticBounds(
    const Ray& worldRay,
    const Geom& volumeGeom,
    float maxDistance,
    float& outEntry,
    float& outExit)
{
    const glm::vec3 localOrigin = volumeMultiplyMV(volumeGeom.inverseTransform, glm::vec4(worldRay.origin, 1.0f));
    const glm::vec3 localDirection = volumeMultiplyMV(volumeGeom.inverseTransform, glm::vec4(worldRay.direction, 0.0f));

    glm::vec3 bboxMin(0.0f);
    glm::vec3 bboxMax(0.0f);
    getVolumeLocalBounds(volumeGeom, bboxMin, bboxMax);

    float tMin = 0.0f;
    float tFar = maxDistance;

    for (int axis = 0; axis < 3; ++axis)
    {
        const float origin = localOrigin[axis];
        const float direction = localDirection[axis];
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
            const float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tMin = fmaxf(tMin, t0);
        tFar = fminf(tFar, t1);
        if (tMin > tFar)
        {
            return false;
        }
    }

    outEntry = tMin;
    outExit = tFar;
    return tFar > MIN_INTERSECTION_T;
}

CUDA_INLINE float evaluateVolumeDensityLocal(
    const Geom& volumeGeom,
    const glm::vec3& localPoint,
    float timeSeconds)
{
    const Geom::VolumeSettings& volume = volumeGeom.volume;
    glm::vec3 localMin(0.0f);
    glm::vec3 localMax(0.0f);
    getVolumeLocalBounds(volumeGeom, localMin, localMax);

    const glm::vec3 localExtent = glm::max(localMax - localMin, glm::vec3(0.001f));
    const glm::vec3 uvw = (localPoint - localMin) / localExtent;
    if (uvw.x < 0.0f || uvw.x > 1.0f
        || uvw.y < 0.0f || uvw.y > 1.0f
        || uvw.z < 0.0f || uvw.z > 1.0f)
    {
        return 0.0f;
    }

    const float boundsFade =
        volumeSmoothstep(0.00f, 0.10f, uvw.x) * volumeSmoothstep(0.00f, 0.10f, 1.0f - uvw.x) *
        volumeSmoothstep(0.00f, 0.10f, uvw.y) * volumeSmoothstep(0.00f, 0.10f, 1.0f - uvw.y) *
        volumeSmoothstep(0.00f, 0.10f, uvw.z) * volumeSmoothstep(0.00f, 0.10f, 1.0f - uvw.z);
    if (volume.model == Geom::VOLUME_MODEL_CLOUD)
    {
        const float bottomRamp = volumeSmoothstep(0.0f, volume.bottomFade, uvw.y);
        const float topRamp = volumeSmoothstep(0.0f, volume.topFade, 1.0f - uvw.y);
        const float heightProfile = glm::clamp(bottomRamp * topRamp, 0.0f, 1.0f);
        return glm::max(boundsFade * heightProfile, 0.0f);
    }

    const glm::vec3 windOffset = glm::vec3(volume.windDirection.x, volume.windDirection.y, volume.windDirection.z)
        * (volume.windSpeed * timeSeconds);
    const glm::vec3 normalizedLocalPoint = (uvw - glm::vec3(0.5f)) * 2.0f;
    const glm::vec3 noisePos = normalizedLocalPoint * volume.noiseScale + windOffset;
    const float baseNoise = volumeFbm(noisePos);
    const float detailNoise = volumeFbm(noisePos * (volume.detailNoiseScale / fmaxf(volume.noiseScale, 0.001f)));

    const float combinedNoise = glm::mix(baseNoise, detailNoise, 0.35f);
    const float thresholdMask = volumeSmoothstep(
        volume.densityThreshold,
        volume.densityThreshold + volume.densitySoftness,
        combinedNoise);
    const float detailModulation = glm::mix(0.82f, 1.18f, combinedNoise);
    const float carveModulation = glm::mix(1.0f, thresholdMask, 0.22f);
    const float modulation = glm::clamp(detailModulation * carveModulation, 0.55f, 1.20f);

    return glm::max(boundsFade * modulation, 0.0f);
}
