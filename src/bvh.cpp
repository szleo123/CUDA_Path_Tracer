#include "bvh.h"

#include <algorithm>
#include <limits>

namespace
{
struct Bounds3
{
    glm::vec3 min;
    glm::vec3 max;
};

Bounds3 makeEmptyBounds()
{
    const float inf = std::numeric_limits<float>::infinity();
    return { glm::vec3(inf), glm::vec3(-inf) };
}

Bounds3 unionBounds(const Bounds3& a, const Bounds3& b)
{
    return { glm::min(a.min, b.min), glm::max(a.max, b.max) };
}

Bounds3 triangleBounds(const Triangle& triangle)
{
    Bounds3 bounds;
    bounds.min = glm::min(triangle.p0, glm::min(triangle.p1, triangle.p2));
    bounds.max = glm::max(triangle.p0, glm::max(triangle.p1, triangle.p2));
    return bounds;
}

glm::vec3 triangleCentroid(const Triangle& triangle)
{
    return (triangle.p0 + triangle.p1 + triangle.p2) / 3.0f;
}

Bounds3 scenePrimitiveBounds(const ScenePrimitive& primitive)
{
    return { primitive.bboxMin, primitive.bboxMax };
}

glm::vec3 scenePrimitiveCentroid(const ScenePrimitive& primitive)
{
    return 0.5f * (primitive.bboxMin + primitive.bboxMax);
}

template <typename Primitive, typename BoundsFn>
Bounds3 rangeBounds(const std::vector<Primitive>& primitives, int start, int end, BoundsFn getBounds)
{
    Bounds3 bounds = makeEmptyBounds();
    for (int i = start; i < end; ++i)
    {
        bounds = unionBounds(bounds, getBounds(primitives[i]));
    }
    return bounds;
}

template <typename Primitive, typename CentroidFn>
Bounds3 centroidBounds(const std::vector<Primitive>& primitives, int start, int end, CentroidFn getCentroid)
{
    Bounds3 bounds = makeEmptyBounds();
    for (int i = start; i < end; ++i)
    {
        const glm::vec3 centroid = getCentroid(primitives[i]);
        bounds.min = glm::min(bounds.min, centroid);
        bounds.max = glm::max(bounds.max, centroid);
    }
    return bounds;
}

int longestAxis(const glm::vec3& extent)
{
    if (extent.x >= extent.y && extent.x >= extent.z)
    {
        return 0;
    }
    if (extent.y >= extent.z)
    {
        return 1;
    }
    return 2;
}

int buildTriangleNode(
    std::vector<Triangle>& triangles,
    std::vector<TriangleBvhNode>& nodes,
    int start,
    int end)
{
    TriangleBvhNode node{};
    const Bounds3 bounds = rangeBounds(triangles, start, end, triangleBounds);
    node.bboxMin = bounds.min;
    node.bboxMax = bounds.max;
    node.leftFirst = -1;
    node.rightChild = -1;
    node.triCount = 0;

    const int nodeIndex = static_cast<int>(nodes.size());
    nodes.push_back(node);

    const int count = end - start;
    if (count <= 4)
    {
        nodes[nodeIndex].leftFirst = start;
        nodes[nodeIndex].rightChild = -1;
        nodes[nodeIndex].triCount = count;
        return nodeIndex;
    }

    const Bounds3 centers = centroidBounds(triangles, start, end, triangleCentroid);
    const glm::vec3 extent = centers.max - centers.min;
    const int axis = longestAxis(extent);

    int mid = (start + end) / 2;
    if (extent[axis] > 0.0f)
    {
        const float splitPos = 0.5f * (centers.min[axis] + centers.max[axis]);
        auto partitionIt = std::partition(
            triangles.begin() + start,
            triangles.begin() + end,
            [axis, splitPos](const Triangle& triangle)
            {
                return triangleCentroid(triangle)[axis] < splitPos;
            });
        mid = static_cast<int>(partitionIt - triangles.begin());
    }

    if (mid <= start || mid >= end)
    {
        std::nth_element(
            triangles.begin() + start,
            triangles.begin() + (start + end) / 2,
            triangles.begin() + end,
            [axis](const Triangle& a, const Triangle& b)
            {
                return triangleCentroid(a)[axis] < triangleCentroid(b)[axis];
            });
        mid = (start + end) / 2;
    }

    const int leftChild = buildTriangleNode(triangles, nodes, start, mid);
    const int rightChild = buildTriangleNode(triangles, nodes, mid, end);
    nodes[nodeIndex].leftFirst = leftChild;
    nodes[nodeIndex].rightChild = rightChild;
    nodes[nodeIndex].triCount = 0;
    return nodeIndex;
}

int buildSceneNode(
    std::vector<ScenePrimitive>& primitives,
    std::vector<SceneBvhNode>& nodes,
    int start,
    int end)
{
    SceneBvhNode node{};
    const Bounds3 bounds = rangeBounds(primitives, start, end, scenePrimitiveBounds);
    node.bboxMin = bounds.min;
    node.bboxMax = bounds.max;
    node.leftFirst = -1;
    node.rightChild = -1;
    node.primitiveCount = 0;

    const int nodeIndex = static_cast<int>(nodes.size());
    nodes.push_back(node);

    const int count = end - start;
    if (count <= 2)
    {
        nodes[nodeIndex].leftFirst = start;
        nodes[nodeIndex].rightChild = -1;
        nodes[nodeIndex].primitiveCount = count;
        return nodeIndex;
    }

    const Bounds3 centers = centroidBounds(primitives, start, end, scenePrimitiveCentroid);
    const glm::vec3 extent = centers.max - centers.min;
    const int axis = longestAxis(extent);

    int mid = (start + end) / 2;
    if (extent[axis] > 0.0f)
    {
        const float splitPos = 0.5f * (centers.min[axis] + centers.max[axis]);
        auto partitionIt = std::partition(
            primitives.begin() + start,
            primitives.begin() + end,
            [axis, splitPos](const ScenePrimitive& primitive)
            {
                return scenePrimitiveCentroid(primitive)[axis] < splitPos;
            });
        mid = static_cast<int>(partitionIt - primitives.begin());
    }

    if (mid <= start || mid >= end)
    {
        std::nth_element(
            primitives.begin() + start,
            primitives.begin() + (start + end) / 2,
            primitives.begin() + end,
            [axis](const ScenePrimitive& a, const ScenePrimitive& b)
            {
                return scenePrimitiveCentroid(a)[axis] < scenePrimitiveCentroid(b)[axis];
            });
        mid = (start + end) / 2;
    }

    const int leftChild = buildSceneNode(primitives, nodes, start, mid);
    const int rightChild = buildSceneNode(primitives, nodes, mid, end);
    nodes[nodeIndex].leftFirst = leftChild;
    nodes[nodeIndex].rightChild = rightChild;
    nodes[nodeIndex].primitiveCount = 0;
    return nodeIndex;
}
}

bool buildTriangleBvh(
    std::vector<Triangle>& triangles,
    std::vector<TriangleBvhNode>& outNodes)
{
    outNodes.clear();
    if (triangles.empty())
    {
        return true;
    }

    outNodes.reserve(triangles.size() * 2);
    buildTriangleNode(triangles, outNodes, 0, static_cast<int>(triangles.size()));
    return true;
}

bool buildSceneBvh(
    std::vector<ScenePrimitive>& primitives,
    std::vector<SceneBvhNode>& outNodes)
{
    outNodes.clear();
    if (primitives.empty())
    {
        return true;
    }

    outNodes.reserve(primitives.size() * 2);
    buildSceneNode(primitives, outNodes, 0, static_cast<int>(primitives.size()));
    return true;
}
