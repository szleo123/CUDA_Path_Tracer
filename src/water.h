#pragma once

#include "sceneStructs.h"
#include "utilities.h"

#include <cfloat>
#include <cmath>

CUDA_INLINE glm::vec3 waterMultiplyMV(const glm::mat4& m, const glm::vec4& v)
{
    return glm::vec3(m * v);
}

CUDA_INLINE glm::vec2 normalizeWaterDirection(glm::vec2 direction)
{
    const float lengthSquared = glm::dot(direction, direction);
    if (lengthSquared <= EPSILON)
    {
        return glm::vec2(1.0f, 0.0f);
    }
    return direction / sqrtf(lengthSquared);
}

CUDA_INLINE float waterHash11(float value)
{
    const float s = sinf(value * 127.1f) * 43758.5453f;
    return s - floorf(s);
}

CUDA_INLINE float waterHash21(const glm::vec2& p)
{
    const float s = sinf(glm::dot(p, glm::vec2(127.1f, 311.7f))) * 43758.5453f;
    return s - floorf(s);
}

CUDA_INLINE float waterValueNoise(const glm::vec2& p)
{
    const glm::vec2 cell = glm::floor(p);
    const glm::vec2 local = p - cell;
    const glm::vec2 smooth = local * local * (glm::vec2(3.0f) - 2.0f * local);

    const float n00 = waterHash21(cell + glm::vec2(0.0f, 0.0f));
    const float n10 = waterHash21(cell + glm::vec2(1.0f, 0.0f));
    const float n01 = waterHash21(cell + glm::vec2(0.0f, 1.0f));
    const float n11 = waterHash21(cell + glm::vec2(1.0f, 1.0f));

    const float nx0 = glm::mix(n00, n10, smooth.x);
    const float nx1 = glm::mix(n01, n11, smooth.x);
    return glm::mix(nx0, nx1, smooth.y);
}

CUDA_INLINE float waterFbm(glm::vec2 p)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    for (int octave = 0; octave < 3; ++octave)
    {
        sum += amplitude * waterValueNoise(p * frequency);
        p = glm::mat2(1.6f, -1.2f, 1.2f, 1.6f) * p;
        frequency *= 1.9f;
        amplitude *= 0.5f;
    }
    return glm::clamp(sum / 0.875f, 0.0f, 1.0f);
}

CUDA_INLINE float waterSmoothstep(float edge0, float edge1, float x)
{
    const float denom = fmaxf(edge1 - edge0, EPSILON);
    const float t = glm::clamp((x - edge0) / denom, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

CUDA_INLINE int clampWaterWaveCount(int waveCount)
{
    return waveCount < 0
        ? 0
        : (waveCount > RENDER_CONFIG_MAX_GERSTNER_WAVES ? RENDER_CONFIG_MAX_GERSTNER_WAVES : waveCount);
}

CUDA_INLINE float computeWaterMaxVerticalDisplacement(const Geom::WaterSettings& water)
{
    float displacement = 0.0f;
    const int waveCount = clampWaterWaveCount(water.waveCount);
    for (int waveIndex = 0; waveIndex < waveCount; ++waveIndex)
    {
        displacement += fabsf(water.waves[waveIndex].amplitude);
    }
    return displacement;
}

CUDA_INLINE void getWaterLocalBounds(
    const Geom::WaterSettings& water,
    glm::vec3& outMin,
    glm::vec3& outMax)
{
    const float halfExtent = water.infinitePlane != 0
        ? RENDER_CONFIG_WATER_INFINITE_HALF_EXTENT
        : 0.5f;
    const float maxDisplacement = computeWaterMaxVerticalDisplacement(water);
    const float halfThickness = fmaxf(
        maxDisplacement + RENDER_CONFIG_WATER_INTERSECTION_EPSILON,
        RENDER_CONFIG_WATER_MIN_HALF_THICKNESS);
    outMin = glm::vec3(-halfExtent, -halfThickness, -halfExtent);
    outMax = glm::vec3(halfExtent, halfThickness, halfExtent);
}

CUDA_INLINE void evaluateGerstnerSurfaceLocal(
    const Geom::WaterSettings& water,
    const glm::vec2& baseXZ,
    float timeSeconds,
    glm::vec3& outPosition,
    glm::vec3& outDpDx,
    glm::vec3& outDpDz)
{
    outPosition = glm::vec3(baseXZ.x, 0.0f, baseXZ.y);
    outDpDx = glm::vec3(1.0f, 0.0f, 0.0f);
    outDpDz = glm::vec3(0.0f, 0.0f, 1.0f);

    const int waveCount = clampWaterWaveCount(water.waveCount);
    for (int waveIndex = 0; waveIndex < waveCount; ++waveIndex)
    {
        const Geom::WaterSettings::Wave& wave = water.waves[waveIndex];
        if (fabsf(wave.amplitude) <= EPSILON || wave.wavelength <= EPSILON)
        {
            continue;
        }

        const glm::vec2 direction = normalizeWaterDirection(wave.direction);
        const float waveNumber = TWO_PI / wave.wavelength;
        const float phase = waveNumber * glm::dot(direction, baseXZ) + wave.speed * timeSeconds;
        const float sinPhase = sinf(phase);
        const float cosPhase = cosf(phase);
        const float heightDerivativeScale = wave.amplitude * waveNumber * cosPhase;

        outPosition.y += wave.amplitude * sinPhase;
        outDpDx += glm::vec3(0.0f, heightDerivativeScale * direction.x, 0.0f);
        outDpDz += glm::vec3(0.0f, heightDerivativeScale * direction.y, 0.0f);
    }
}

CUDA_INLINE float evaluateWaterFoamAmountLocal(
    const Geom::WaterSettings& water,
    const glm::vec2& baseXZ,
    float timeSeconds)
{
    if (water.foamIntensity <= EPSILON)
    {
        return 0.0f;
    }

    glm::vec3 surfacePosition(0.0f);
    glm::vec3 dpDx(1.0f, 0.0f, 0.0f);
    glm::vec3 dpDz(0.0f, 0.0f, 1.0f);
    evaluateGerstnerSurfaceLocal(water, baseXZ, timeSeconds, surfacePosition, dpDx, dpDz);

    glm::vec3 localNormal = glm::cross(dpDz, dpDx);
    if (glm::dot(localNormal, localNormal) <= EPSILON)
    {
        localNormal = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else
    {
        localNormal = glm::normalize(localNormal);
    }

    float crestSignal = 0.0f;
    float weightSum = 0.0f;
    glm::vec2 dominantDirection(0.0f);
    const int waveCount = clampWaterWaveCount(water.waveCount);
    for (int waveIndex = 0; waveIndex < waveCount; ++waveIndex)
    {
        const Geom::WaterSettings::Wave& wave = water.waves[waveIndex];
        if (fabsf(wave.amplitude) <= EPSILON || wave.wavelength <= EPSILON)
        {
            continue;
        }

        const glm::vec2 direction = normalizeWaterDirection(wave.direction);
        const float waveNumber = TWO_PI / wave.wavelength;
        const float phase = waveNumber * glm::dot(direction, baseXZ) + wave.speed * timeSeconds;
        const float crest = powf(fmaxf(sinf(phase), 0.0f), 6.0f);
        const float steepnessWeight = glm::clamp(wave.steepness / 1.5f, 0.0f, 1.0f);
        const float chopWeight = glm::clamp(0.42f / fmaxf(wave.wavelength, 0.05f), 0.0f, 1.0f);
        const float whitecapWeight = steepnessWeight * steepnessWeight * chopWeight * chopWeight * chopWeight;
        const float weight = whitecapWeight;
        if (weight <= EPSILON)
        {
            continue;
        }
        crestSignal += crest * weight;
        weightSum += weight;
        dominantDirection += direction * weight;
    }

    if (weightSum > EPSILON)
    {
        crestSignal /= weightSum;
        dominantDirection /= weightSum;
    }

    if (glm::dot(dominantDirection, dominantDirection) <= EPSILON)
    {
        dominantDirection = glm::vec2(1.0f, 0.0f);
    }
    else
    {
        dominantDirection = glm::normalize(dominantDirection);
    }

    const float slopeSignal = glm::clamp(1.0f - localNormal.y, 0.0f, 1.0f);
    const glm::vec2 drift = dominantDirection * (timeSeconds * 0.18f);
    const glm::vec2 warp(
        waterFbm(baseXZ * 0.22f + drift + glm::vec2(13.7f, -4.9f)),
        waterFbm(baseXZ * 0.19f - drift * 1.3f + glm::vec2(-8.3f, 21.4f)));
    const glm::vec2 warpedUv = baseXZ + (warp - glm::vec2(0.5f)) * 0.9f;
    const float breakupLarge = waterFbm(warpedUv * 0.55f + dominantDirection * timeSeconds * 0.11f);
    const glm::vec2 crossDrift(dominantDirection.y, -dominantDirection.x);
    const float breakupMedium = waterFbm(warpedUv * 1.45f - crossDrift * timeSeconds * 0.27f);
    const float breakupFine = waterFbm(warpedUv * 3.6f + glm::vec2(-timeSeconds * 0.43f, timeSeconds * 0.31f));
    const float streakNoise = 0.5f + 0.5f * sinf(
        glm::dot(warpedUv, dominantDirection) * 5.2f
        + breakupMedium * 3.7f
        + timeSeconds * 0.35f);
    const float patchMask = glm::clamp(
        waterSmoothstep(0.56f, 0.86f, breakupLarge)
        * (0.65f + 0.35f * waterSmoothstep(0.42f, 0.78f, breakupMedium)),
        0.0f,
        1.0f);
    const float detailMask = glm::clamp(
        waterSmoothstep(0.46f, 0.82f, breakupFine)
        * waterSmoothstep(0.38f, 0.78f, streakNoise),
        0.0f,
        1.0f);
    const float crestDrivenSignal = crestSignal * crestSignal * (0.25f + 0.75f * detailMask);
    const float slopeAssist = slopeSignal * crestSignal * 0.08f;
    const float rawFoamSignal = (crestDrivenSignal + slopeAssist) * patchMask;
    const float lower = water.foamThreshold;
    const float upper = water.foamThreshold + water.foamSoftness;
    float smoothMask = 0.0f;
    if (upper <= lower + EPSILON)
    {
        smoothMask = glm::clamp(rawFoamSignal, 0.0f, 1.0f);
    }
    else
    {
        const float t = glm::clamp((rawFoamSignal - lower) / (upper - lower), 0.0f, 1.0f);
        smoothMask = t * t * (3.0f - 2.0f * t);
    }

    const float sparseSuppression = waterSmoothstep(0.28f, 0.72f, patchMask * detailMask);
    return glm::clamp(smoothMask * sparseSuppression * water.foamIntensity, 0.0f, 1.0f);
}

CUDA_INLINE bool intersectWaterLocalBounds(
    const Ray& localRay,
    const glm::vec3& bboxMin,
    const glm::vec3& bboxMax,
    float& outEntry,
    float& outExit)
{
    float tEntry = 0.0f;
    float tExit = FLT_MAX;

    for (int axis = 0; axis < 3; ++axis)
    {
        const float origin = localRay.origin[axis];
        const float direction = localRay.direction[axis];
        if (fabsf(direction) < EPSILON)
        {
            if (origin < bboxMin[axis] || origin > bboxMax[axis])
            {
                return false;
            }
            continue;
        }

        const float inverseDirection = 1.0f / direction;
        float t0 = (bboxMin[axis] - origin) * inverseDirection;
        float t1 = (bboxMax[axis] - origin) * inverseDirection;
        if (t0 > t1)
        {
            const float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tEntry = fmaxf(tEntry, t0);
        tExit = fminf(tExit, t1);
        if (tEntry > tExit)
        {
            return false;
        }
    }

    outEntry = tEntry;
    outExit = tExit;
    return tExit > MIN_INTERSECTION_T;
}

CUDA_INLINE float waterIntersectionTest(
    const Geom& waterGeom,
    const Ray& worldRay,
    float timeSeconds,
    glm::vec3& intersectionPoint,
    glm::vec3& shadingNormal,
    glm::vec3& geometricNormal,
    glm::vec3& tangent,
    glm::vec2& uv,
    bool& outside)
{
    Ray localRay{};
    localRay.origin = waterMultiplyMV(waterGeom.inverseTransform, glm::vec4(worldRay.origin, 1.0f));
    localRay.direction = glm::normalize(waterMultiplyMV(waterGeom.inverseTransform, glm::vec4(worldRay.direction, 0.0f)));

    glm::vec3 bboxMin(0.0f);
    glm::vec3 bboxMax(0.0f);
    getWaterLocalBounds(waterGeom.water, bboxMin, bboxMax);

    float tEntry = 0.0f;
    float tExit = 0.0f;
    if (!intersectWaterLocalBounds(localRay, bboxMin, bboxMax, tEntry, tExit))
    {
        return -1.0f;
    }

    if (fabsf(localRay.direction.y) <= EPSILON)
    {
        return -1.0f;
    }

    const float planeT = -localRay.origin.y / localRay.direction.y;
    if (planeT < fmaxf(tEntry, MIN_INTERSECTION_T) || planeT > tExit)
    {
        return -1.0f;
    }

    auto evaluateHeightError = [&](float rayT, glm::vec3& outPosition, glm::vec3& outDpDx, glm::vec3& outDpDz, glm::vec2& outBaseXZ) {
        const glm::vec3 localRayPoint = localRay.origin + rayT * localRay.direction;
        outBaseXZ = glm::vec2(localRayPoint.x, localRayPoint.z);
        if (waterGeom.water.infinitePlane == 0)
        {
            if (outBaseXZ.x < -0.5f - RENDER_CONFIG_WATER_INTERSECTION_EPSILON
                || outBaseXZ.x > 0.5f + RENDER_CONFIG_WATER_INTERSECTION_EPSILON
                || outBaseXZ.y < -0.5f - RENDER_CONFIG_WATER_INTERSECTION_EPSILON
                || outBaseXZ.y > 0.5f + RENDER_CONFIG_WATER_INTERSECTION_EPSILON)
            {
                return FLT_MAX;
            }
        }

        evaluateGerstnerSurfaceLocal(waterGeom.water, outBaseXZ, timeSeconds, outPosition, outDpDx, outDpDz);
        return localRayPoint.y - outPosition.y;
    };

    const float bracketDelta = (waterGeom.water.maxVerticalDisplacement + RENDER_CONFIG_WATER_INTERSECTION_EPSILON)
        / fmaxf(fabsf(localRay.direction.y), EPSILON);
    float lowT = glm::max(fmaxf(tEntry, MIN_INTERSECTION_T), planeT - bracketDelta);
    float highT = glm::min(tExit, planeT + bracketDelta);
    if (highT - lowT <= RENDER_CONFIG_WATER_INTERSECTION_EPSILON)
    {
        return -1.0f;
    }

    glm::vec3 lowPosition(0.0f);
    glm::vec3 lowDpDx(1.0f, 0.0f, 0.0f);
    glm::vec3 lowDpDz(0.0f, 0.0f, 1.0f);
    glm::vec2 lowBaseXZ(0.0f);
    glm::vec3 highPosition(0.0f);
    glm::vec3 highDpDx(1.0f, 0.0f, 0.0f);
    glm::vec3 highDpDz(0.0f, 0.0f, 1.0f);
    glm::vec2 highBaseXZ(0.0f);
    float lowError = evaluateHeightError(lowT, lowPosition, lowDpDx, lowDpDz, lowBaseXZ);
    float highError = evaluateHeightError(highT, highPosition, highDpDx, highDpDz, highBaseXZ);
    if (lowError == FLT_MAX || highError == FLT_MAX)
    {
        return -1.0f;
    }
    if (lowError * highError > 0.0f)
    {
        return -1.0f;
    }

    glm::vec3 localPosition(0.0f);
    glm::vec3 dpDx(1.0f, 0.0f, 0.0f);
    glm::vec3 dpDz(0.0f, 0.0f, 1.0f);
    glm::vec2 baseXZ(0.0f);

    for (int iteration = 0; iteration < RENDER_CONFIG_WATER_INTERSECTION_MAX_STEPS + 8; ++iteration)
    {
        const float t = 0.5f * (lowT + highT);
        const float heightError = evaluateHeightError(t, localPosition, dpDx, dpDz, baseXZ);
        if (heightError == FLT_MAX)
        {
            return -1.0f;
        }

        if (fabsf(heightError) <= RENDER_CONFIG_WATER_INTERSECTION_EPSILON)
        {
            glm::vec3 localNormal = glm::cross(dpDz, dpDx);
            if (glm::dot(localNormal, localNormal) > EPSILON)
            {
                localNormal = glm::normalize(localNormal);
            }
            else
            {
                localNormal = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            glm::vec3 localTangent = dpDx;
            if (glm::dot(localTangent, localTangent) > EPSILON)
            {
                localTangent = glm::normalize(localTangent);
            }
            else
            {
                localTangent = glm::vec3(1.0f, 0.0f, 0.0f);
            }

            localPosition.x = baseXZ.x;
            localPosition.z = baseXZ.y;
            intersectionPoint = waterMultiplyMV(waterGeom.transform, glm::vec4(localPosition, 1.0f));
            shadingNormal = glm::normalize(waterMultiplyMV(waterGeom.invTranspose, glm::vec4(localNormal, 0.0f)));
            geometricNormal = shadingNormal;
            tangent = waterMultiplyMV(waterGeom.transform, glm::vec4(localTangent, 0.0f));
            if (glm::dot(tangent, tangent) > EPSILON)
            {
                tangent = glm::normalize(tangent);
            }
            else
            {
                tangent = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            uv = (baseXZ + glm::vec2(0.5f)) * waterGeom.water.uvScale;
            outside = glm::dot(worldRay.direction, shadingNormal) < 0.0f;
            return glm::length(worldRay.origin - intersectionPoint);
        }

        if (lowError * heightError <= 0.0f)
        {
            highT = t;
            highError = heightError;
        }
        else
        {
            lowT = t;
            lowError = heightError;
        }
    }

    return -1.0f;
}
