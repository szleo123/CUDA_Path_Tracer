#pragma once

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
#define EPSILON           0.00001f
#define MIN_INTERSECTION_T 0.0001f
#define RAY_ORIGIN_BIAS   0.001f

enum RenderDebugMode
{
    RENDER_DEBUG_NONE = 0,
    RENDER_DEBUG_MESH_UV_CHECKER = 1,
    RENDER_DEBUG_MESH_BASE_COLOR = 2,
    RENDER_DEBUG_MESH_TEXTURE_ONLY = 3
};

class GuiDataContainer
{
public:
    GuiDataContainer()
        : TracedDepth(0)
        , UseMaterialSort(false)
        , SortEveryNIterations(4)
        , SortMaxBounce(2)
        , SortMinPathCount(32768)
        , LastSortTimeMs(0.0f)
        , LastShadeTimeMs(0.0f)
        , LastNumShadedPaths(0)
        , RenderDebugModeValue(RENDER_DEBUG_NONE)
    {}
    int TracedDepth;
    bool UseMaterialSort;
    int SortEveryNIterations;
    int SortMaxBounce;
    int SortMinPathCount;
    float LastSortTimeMs;
    float LastShadeTimeMs;
    int LastNumShadedPaths;
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
