#pragma once

#include "renderConfig.h"
#include "glm/glm.hpp"

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           RENDER_CONFIG_EPSILON
#define MIN_INTERSECTION_T RENDER_CONFIG_MIN_INTERSECTION_T
#define RAY_ORIGIN_BIAS   RENDER_CONFIG_RAY_ORIGIN_BIAS

#ifdef __CUDACC__
#define CUDA_INLINE __host__ __device__ inline
#else
#define CUDA_INLINE inline
#endif

enum RenderDebugMode
{
    RENDER_DEBUG_NONE = 0,
    RENDER_DEBUG_MESH_UV_CHECKER = 1,
    RENDER_DEBUG_MESH_BASE_COLOR = 2,
    RENDER_DEBUG_MESH_TEXTURE_ONLY = 3
};

enum ToneMapMode
{
    TONEMAP_NONE = 0,
    TONEMAP_REINHARD = 1,
    TONEMAP_ACES = 2
};

CUDA_INLINE float saturatef(float value)
{
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

CUDA_INLINE glm::vec3 applyToneMapping(glm::vec3 color, int toneMapMode)
{
    if (toneMapMode == TONEMAP_REINHARD)
    {
        return color / (glm::vec3(1.0f) + color);
    }

    if (toneMapMode == TONEMAP_ACES)
    {
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        glm::vec3 mapped = (color * (a * color + glm::vec3(b)))
            / (color * (c * color + glm::vec3(d)) + glm::vec3(e));
        return glm::clamp(mapped, glm::vec3(0.0f), glm::vec3(1.0f));
    }

    return color;
}

CUDA_INLINE glm::vec3 applyDisplayTransform(glm::vec3 color, float exposureValue, int toneMapMode)
{
    const float exposureScale = powf(2.0f, exposureValue);
    color *= exposureScale;
    color = applyToneMapping(color, toneMapMode);
    color = glm::max(color, glm::vec3(0.0f));

    const float invGamma = 1.0f / 2.2f;
    return glm::vec3(
        powf(saturatef(color.x), invGamma),
        powf(saturatef(color.y), invGamma),
        powf(saturatef(color.z), invGamma));
}

class GuiDataContainer
{
public:
    GuiDataContainer()
        : TracedDepth(0)
        , UseMaterialSort(false)
        , EnableKernelTiming(false)
        , SortEveryNIterations(RENDER_CONFIG_DEFAULT_SORT_EVERY_N_ITERATIONS)
        , SortMaxBounce(RENDER_CONFIG_DEFAULT_SORT_MAX_BOUNCE)
        , SortMinPathCount(RENDER_CONFIG_DEFAULT_SORT_MIN_PATH_COUNT)
        , LastSortTimeMs(0.0f)
        , LastShadeTimeMs(0.0f)
        , LastNumShadedPaths(0)
        , ExposureValue(0.0f)
        , ToneMapModeValue(TONEMAP_REINHARD)
        , RenderDebugModeValue(RENDER_DEBUG_NONE)
    {}
    int TracedDepth;
    bool UseMaterialSort;
    bool EnableKernelTiming;
    int SortEveryNIterations;
    int SortMaxBounce;
    int SortMinPathCount;
    float LastSortTimeMs;
    float LastShadeTimeMs;
    int LastNumShadedPaths;
    float ExposureValue;
    int ToneMapModeValue;
    int RenderDebugModeValue;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
