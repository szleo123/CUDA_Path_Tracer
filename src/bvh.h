#pragma once

#include "sceneStructs.h"

#include <vector>

bool buildTriangleBvh(
    std::vector<Triangle>& triangles,
    std::vector<TriangleBvhNode>& outNodes);

bool buildSceneBvh(
    std::vector<ScenePrimitive>& primitives,
    std::vector<SceneBvhNode>& outNodes);
