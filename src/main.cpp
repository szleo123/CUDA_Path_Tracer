#include "glslUtility.hpp"
#include "image.h"
#include "intersections.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX = 0.0;
static double lastY = 0.0;

static bool camchanged = true;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWindow = false;
static int selectedObjectIndex = -1;

enum class TransformMode
{
    Translate,
    Rotate,
    Scale
};

static TransformMode activeTransformMode = TransformMode::Translate;
static bool showAnalyticsWindow = true;
static bool showSceneObjectsWindow = true;
static bool showViewportGizmo = true;
static bool objectManipulationActive = false;
static bool cudaSceneInitialized = false;
static double manipulationStartX = 0.0;
static double manipulationStartY = 0.0;
static glm::vec3 manipulationStartTranslation(0.0f);
static glm::vec3 manipulationStartRotation(0.0f);
static glm::vec3 manipulationStartScale(1.0f);

// Forward declarations for window loop and interactivity
void runCuda();
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

namespace
{
void applyObjectTransform(int objectIndex, const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale);

const char* getRenderDebugModeLabel(int mode)
{
    switch (mode)
    {
    case RENDER_DEBUG_MESH_UV_CHECKER:
        return "Mesh UV Checker";
    case RENDER_DEBUG_MESH_BASE_COLOR:
        return "Mesh Base Color";
    case RENDER_DEBUG_MESH_TEXTURE_ONLY:
        return "Mesh Texture Only";
    default:
        return "None";
    }
}

const char* getToneMapModeLabel(int mode)
{
    switch (mode)
    {
    case TONEMAP_NONE:
        return "None";
    case TONEMAP_ACES:
        return "ACES";
    default:
        return "Reinhard";
    }
}

const char* getTransformModeLabel(TransformMode mode)
{
    switch (mode)
    {
    case TransformMode::Rotate:
        return "Rotate";
    case TransformMode::Scale:
        return "Scale";
    default:
        return "Translate";
    }
}

bool transformModeButton(const char* label, TransformMode mode)
{
    if (!ImGui::Button(label))
    {
        return false;
    }

    activeTransformMode = mode;
    return true;
}

void renderMainMenuBar()
{
    if (!ImGui::BeginMainMenuBar())
    {
        return;
    }

    if (ImGui::BeginMenu("View"))
    {
        ImGui::MenuItem("Analytics", nullptr, &showAnalyticsWindow);
        ImGui::MenuItem("Scene Objects", nullptr, &showSceneObjectsWindow);
        ImGui::MenuItem("Viewport Gizmo", nullptr, &showViewportGizmo);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Transform"))
    {
        if (ImGui::MenuItem("Translate", "W", activeTransformMode == TransformMode::Translate))
        {
            activeTransformMode = TransformMode::Translate;
        }
        if (ImGui::MenuItem("Rotate", "E", activeTransformMode == TransformMode::Rotate))
        {
            activeTransformMode = TransformMode::Rotate;
        }
        if (ImGui::MenuItem("Scale", "R", activeTransformMode == TransformMode::Scale))
        {
            activeTransformMode = TransformMode::Scale;
        }
        ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
}

void renderAnalyticsWindow()
{
    ImGui::Begin("Path Tracer Analytics", &showAnalyticsWindow);

    if (ImGui::Checkbox("Sort by Material", &imguiData->UseMaterialSort))
    {
        iteration = 0;
    }

    const char* toneMapLabels[] = { "None", "Reinhard", "ACES" };
    if (ImGui::Combo("Tone Mapping", &imguiData->ToneMapModeValue, toneMapLabels, IM_ARRAYSIZE(toneMapLabels)))
    {
        iteration = 0;
    }
    if (ImGui::SliderFloat("Exposure", &imguiData->ExposureValue, -5.0f, 5.0f, "%.2f EV"))
    {
        iteration = 0;
    }

    const char* renderDebugLabels[] = { "None", "Mesh UV Checker", "Mesh Base Color", "Mesh Texture Only" };
    if (ImGui::Combo("Render Debug", &imguiData->RenderDebugModeValue, renderDebugLabels, IM_ARRAYSIZE(renderDebugLabels)))
    {
        iteration = 0;
    }

    if (imguiData->UseMaterialSort)
    {
        if (ImGui::SliderInt("Sort Every N Iter", &imguiData->SortEveryNIterations, 1, 32))
        {
            iteration = 0;
        }
        if (ImGui::SliderInt("Sort Max Bounce", &imguiData->SortMaxBounce, 1, 16))
        {
            iteration = 0;
        }
        if (ImGui::SliderInt("Sort Min Paths", &imguiData->SortMinPathCount, 1024, 262144))
        {
            iteration = 0;
        }
    }

    ImGui::Text("Tone Mapping %s", getToneMapModeLabel(imguiData->ToneMapModeValue));
    ImGui::Text("Exposure %.2f EV", imguiData->ExposureValue);
    ImGui::Text("Render Debug %s", getRenderDebugModeLabel(imguiData->RenderDebugModeValue));
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Text("Last Sort Time %.3f ms", imguiData->LastSortTimeMs);
    ImGui::Text("Last Shade Time %.3f ms", imguiData->LastShadeTimeMs);
    ImGui::Text("Last Shaded Paths %d", imguiData->LastNumShadedPaths);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
}

void renderSceneObjectsWindow()
{
    ImGui::Begin("Scene Objects", &showSceneObjectsWindow);
    ImGui::Text("Mode: %s", getTransformModeLabel(activeTransformMode));
    transformModeButton("Translate", TransformMode::Translate);
    ImGui::SameLine();
    transformModeButton("Rotate", TransformMode::Rotate);
    ImGui::SameLine();
    transformModeButton("Scale", TransformMode::Scale);
    ImGui::Separator();
    ImGui::TextUnformatted("Viewport controls:");
    ImGui::BulletText("Left click selects and drags the picked object");
    ImGui::BulletText("W / E / R switch translate / rotate / scale");
    ImGui::BulletText("Alt + Left orbit, Alt + Middle pan, Alt + Right zoom");

    if (scene->objects.empty())
    {
        ImGui::Text("No editable objects in scene.");
        ImGui::End();
        return;
    }

    if (ImGui::BeginListBox("Objects"))
    {
        for (int i = 0; i < static_cast<int>(scene->objects.size()); ++i)
        {
            const bool isSelected = (selectedObjectIndex == i);
            if (ImGui::Selectable(scene->objects[i].name.c_str(), isSelected))
            {
                selectedObjectIndex = i;
            }
            if (isSelected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndListBox();
    }

    if (selectedObjectIndex < 0 || selectedObjectIndex >= static_cast<int>(scene->objects.size()))
    {
        ImGui::TextUnformatted("Click an object in the viewport to select it.");
        ImGui::End();
        return;
    }

    SceneObject& object = scene->objects[selectedObjectIndex];
    ImGui::Separator();
    ImGui::Text("Selected: %s", object.name.c_str());

    glm::vec3 translation = object.translation;
    glm::vec3 rotation = object.rotation;
    glm::vec3 scale = object.scale;
    bool changed = false;
    changed |= ImGui::DragFloat3("Translate", &translation[0], 0.05f);
    changed |= ImGui::DragFloat3("Rotate", &rotation[0], 1.0f);
    changed |= ImGui::DragFloat3("Scale", &scale[0], 0.05f, 0.01f, 100.0f);

    if (changed)
    {
        applyObjectTransform(selectedObjectIndex, translation, rotation, scale);
    }

    ImGui::End();
}

bool isAltPressed(GLFWwindow* window)
{
    return glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS
        || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
}

void syncCameraState()
{
    if (!camchanged)
    {
        return;
    }

    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;

    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.y = zoom * cos(theta);
    cameraPosition.z = zoom * cos(phi) * sin(theta);

    cam.view = -glm::normalize(cameraPosition);
    const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    cam.right = glm::normalize(glm::cross(cam.view, worldUp));
    cam.up = glm::normalize(glm::cross(cam.right, cam.view));
    cam.position = cameraPosition + cam.lookAt;

    camchanged = false;
}

Ray buildCameraRay(double xpos, double ypos)
{
    syncCameraState();

    const Camera& cam = renderState->camera;
    const float viewportX = static_cast<float>(cam.resolution.x) - 1.0f - static_cast<float>(xpos);
    const float sx = viewportX - static_cast<float>(cam.resolution.x) * 0.5f;
    const float sy = static_cast<float>(ypos) - static_cast<float>(cam.resolution.y) * 0.5f;

    Ray ray;
    ray.origin = cam.position;
    ray.direction = glm::normalize(
        cam.view
        - cam.right * cam.pixelLength.x * sx
        - cam.up * cam.pixelLength.y * sy);
    return ray;
}

bool intersectAabbForPicking(
    const Ray& ray,
    const glm::vec3& bboxMin,
    const glm::vec3& bboxMax,
    float tMax,
    float& tEntry)
{
    float tMin = 0.0f;
    float tFar = tMax;

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

    tEntry = tMin;
    return true;
}

Ray transformRayForPicking(const Ray& ray, const glm::mat4& transform)
{
    Ray transformedRay{};
    transformedRay.origin = multiplyMV(transform, glm::vec4(ray.origin, 1.0f));
    transformedRay.direction = glm::normalize(multiplyMV(transform, glm::vec4(ray.direction, 0.0f)));
    return transformedRay;
}

bool intersectGeomForPicking(const Geom& geom, const Ray& ray, float maxDistance, float& outT)
{
    glm::vec3 intersectionPoint;
    glm::vec3 normal;
    bool outside = false;
    const float t = (geom.type == CUBE)
        ? boxIntersectionTest(geom, ray, intersectionPoint, normal, outside)
        : sphereIntersectionTest(geom, ray, intersectionPoint, normal, outside);

    if (t > MIN_INTERSECTION_T && t < maxDistance)
    {
        outT = t;
        return true;
    }
    return false;
}

bool traverseTriangleBvhForPicking(
    const Scene& sceneRef,
    const MeshInstance& meshInstance,
    const Ray& worldRay,
    float maxDistance,
    float& outT)
{
    if (meshInstance.bvhRootIndex < 0)
    {
        return false;
    }

    const Ray localRay = transformRayForPicking(worldRay, meshInstance.inverseTransform);
    constexpr int kMaxBvhStackSize = 64;
    const float maxFloat = std::numeric_limits<float>::max();
    int stack[kMaxBvhStackSize];
    int stackSize = 0;
    stack[stackSize++] = meshInstance.bvhRootIndex;
    bool hit = false;
    outT = maxDistance;

    while (stackSize > 0)
    {
        const TriangleBvhNode& node = sceneRef.triangleBvhNodes[stack[--stackSize]];
        float nodeEntry = 0.0f;
        if (!intersectAabbForPicking(localRay, node.bboxMin, node.bboxMax, maxFloat, nodeEntry))
        {
            continue;
        }

        if (node.triCount > 0)
        {
            for (int i = 0; i < node.triCount; ++i)
            {
                const Triangle& triangle = sceneRef.triangles[node.leftFirst + i];
                glm::vec3 intersectionPoint;
                glm::vec3 shadingNormal;
                glm::vec3 geometricNormal;
                glm::vec2 uv;
                const float localT = triangleIntersectionTest(
                    triangle,
                    localRay,
                    intersectionPoint,
                    shadingNormal,
                    geometricNormal,
                    uv);
                if (localT <= MIN_INTERSECTION_T)
                {
                    continue;
                }

                const glm::vec3 worldPoint = multiplyMV(meshInstance.transform, glm::vec4(intersectionPoint, 1.0f));
                const float worldT = glm::length(worldPoint - worldRay.origin);
                if (worldT > MIN_INTERSECTION_T && worldT < outT)
                {
                    outT = worldT;
                    hit = true;
                }
            }
            continue;
        }

        const int leftChild = node.leftFirst;
        const int rightChild = node.rightChild;
        float leftEntry = 0.0f;
        float rightEntry = 0.0f;
        const bool hitLeft = intersectAabbForPicking(
            localRay,
            sceneRef.triangleBvhNodes[leftChild].bboxMin,
            sceneRef.triangleBvhNodes[leftChild].bboxMax,
            maxFloat,
            leftEntry);
        const bool hitRight = intersectAabbForPicking(
            localRay,
            sceneRef.triangleBvhNodes[rightChild].bboxMin,
            sceneRef.triangleBvhNodes[rightChild].bboxMax,
            maxFloat,
            rightEntry);

        if (hitLeft && hitRight)
        {
            const bool leftFirst = leftEntry < rightEntry;
            if (stackSize + 2 <= kMaxBvhStackSize)
            {
                stack[stackSize++] = leftFirst ? rightChild : leftChild;
                stack[stackSize++] = leftFirst ? leftChild : rightChild;
            }
        }
        else if (hitLeft)
        {
            if (stackSize + 1 <= kMaxBvhStackSize)
            {
                stack[stackSize++] = leftChild;
            }
        }
        else if (hitRight)
        {
            if (stackSize + 1 <= kMaxBvhStackSize)
            {
                stack[stackSize++] = rightChild;
            }
        }
    }

    return hit;
}

int pickSceneObject(double xpos, double ypos)
{
    if (scene == nullptr)
    {
        return -1;
    }

    const Ray ray = buildCameraRay(xpos, ypos);
    float closestT = std::numeric_limits<float>::max();
    int pickedIndex = -1;

    if (scene->sceneBvhNodes.empty() || scene->scenePrimitives.empty())
    {
        return -1;
    }

    constexpr int kMaxBvhStackSize = 64;
    int stack[kMaxBvhStackSize];
    int stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0)
    {
        const SceneBvhNode& node = scene->sceneBvhNodes[stack[--stackSize]];
        float nodeEntry = 0.0f;
        if (!intersectAabbForPicking(ray, node.bboxMin, node.bboxMax, closestT, nodeEntry))
        {
            continue;
        }

        if (node.primitiveCount > 0)
        {
            for (int i = 0; i < node.primitiveCount; ++i)
            {
                const ScenePrimitive& primitive = scene->scenePrimitives[node.leftFirst + i];
                float candidateT = closestT;
                bool hit = false;

                if (primitive.type == SCENE_PRIMITIVE_GEOM)
                {
                    const Geom& geom = scene->geoms[primitive.index];
                    hit = intersectGeomForPicking(geom, ray, closestT, candidateT);
                    if (hit)
                    {
                        pickedIndex = geom.objectIndex;
                    }
                }
                else if (primitive.type == SCENE_PRIMITIVE_MESH_INSTANCE)
                {
                    const MeshInstance& meshInstance = scene->meshInstances[primitive.index];
                    hit = traverseTriangleBvhForPicking(*scene, meshInstance, ray, closestT, candidateT);
                    if (hit)
                    {
                        pickedIndex = meshInstance.objectIndex;
                    }
                }

                if (hit && candidateT < closestT)
                {
                    closestT = candidateT;
                }
            }
            continue;
        }

        const int leftChild = node.leftFirst;
        const int rightChild = node.rightChild;
        float leftEntry = 0.0f;
        float rightEntry = 0.0f;
        const bool hitLeft = intersectAabbForPicking(
            ray,
            scene->sceneBvhNodes[leftChild].bboxMin,
            scene->sceneBvhNodes[leftChild].bboxMax,
            closestT,
            leftEntry);
        const bool hitRight = intersectAabbForPicking(
            ray,
            scene->sceneBvhNodes[rightChild].bboxMin,
            scene->sceneBvhNodes[rightChild].bboxMax,
            closestT,
            rightEntry);

        if (hitLeft && hitRight)
        {
            const bool leftFirst = leftEntry < rightEntry;
            if (stackSize + 2 <= kMaxBvhStackSize)
            {
                stack[stackSize++] = leftFirst ? rightChild : leftChild;
                stack[stackSize++] = leftFirst ? leftChild : rightChild;
            }
        }
        else if (hitLeft)
        {
            if (stackSize + 1 <= kMaxBvhStackSize)
            {
                stack[stackSize++] = leftChild;
            }
        }
        else if (hitRight)
        {
            if (stackSize + 1 <= kMaxBvhStackSize)
            {
                stack[stackSize++] = rightChild;
            }
        }
    }

    return pickedIndex;
}

void applyObjectTransform(int objectIndex, const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale)
{
    if (objectIndex < 0 || objectIndex >= static_cast<int>(scene->objects.size()))
    {
        return;
    }

    scene->updateObjectTransform(
        static_cast<size_t>(objectIndex),
        translation,
        rotation,
        glm::max(scale, glm::vec3(0.01f)));
    iteration = 0;
    renderState = &scene->state;
}

void applyViewportManipulation(double xpos, double ypos)
{
    if (!objectManipulationActive
        || selectedObjectIndex < 0
        || selectedObjectIndex >= static_cast<int>(scene->objects.size()))
    {
        return;
    }

    syncCameraState();
    const Camera& cam = renderState->camera;
    const SceneObject& object = scene->objects[selectedObjectIndex];
    const float dx = static_cast<float>(xpos - manipulationStartX);
    const float dy = static_cast<float>(ypos - manipulationStartY);

    glm::vec3 translation = manipulationStartTranslation;
    glm::vec3 rotation = manipulationStartRotation;
    glm::vec3 scale = manipulationStartScale;

    switch (activeTransformMode)
    {
    case TransformMode::Translate:
    {
        const float distanceToCamera = glm::length(object.translation - cam.position);
        const float xScale = cam.pixelLength.x * glm::max(distanceToCamera, 1.0f);
        const float yScale = cam.pixelLength.y * glm::max(distanceToCamera, 1.0f);
        translation += cam.right * (dx * xScale);
        translation -= cam.up * (dy * yScale);
        break;
    }
    case TransformMode::Rotate:
        rotation.y += dx * 0.35f;
        rotation.x += dy * 0.35f;
        break;
    case TransformMode::Scale:
    {
        const float uniformFactor = std::exp((dx - dy) * 0.01f);
        scale = manipulationStartScale * uniformFactor;
        break;
    }
    }

    applyObjectTransform(selectedObjectIndex, translation, rotation, scale);
}
bool projectWorldToViewport(const glm::vec3& worldPoint, ImVec2& screenPoint)
{
    syncCameraState();
    const Camera& cam = renderState->camera;
    const glm::vec3 toPoint = worldPoint - cam.position;
    const float depth = glm::dot(toPoint, cam.view);
    if (depth <= 0.001f)
    {
        return false;
    }

    const float sx = -glm::dot(toPoint, cam.right) / (cam.pixelLength.x * depth);
    const float sy = -glm::dot(toPoint, cam.up) / (cam.pixelLength.y * depth);
    const float unmirroredX = sx + static_cast<float>(cam.resolution.x) * 0.5f;
    const float mirroredX = static_cast<float>(cam.resolution.x) - 1.0f - unmirroredX;
    const float viewportY = sy + static_cast<float>(cam.resolution.y) * 0.5f;

    screenPoint = ImVec2(mirroredX, viewportY);
    return true;
}

void drawArrowHead(ImDrawList* drawList, const ImVec2& start, const ImVec2& end, ImU32 color)
{
    const ImVec2 dir(end.x - start.x, end.y - start.y);
    const float len = sqrtf(dir.x * dir.x + dir.y * dir.y);
    if (len < 1.0f)
    {
        return;
    }

    const ImVec2 n(dir.x / len, dir.y / len);
    const ImVec2 side(-n.y, n.x);
    const float headLength = 10.0f;
    const float headWidth = 5.0f;
    const ImVec2 base(end.x - n.x * headLength, end.y - n.y * headLength);
    drawList->AddTriangleFilled(
        end,
        ImVec2(base.x + side.x * headWidth, base.y + side.y * headWidth),
        ImVec2(base.x - side.x * headWidth, base.y - side.y * headWidth),
        color);
}

void drawViewportGizmoOverlay()
{
    if (!showViewportGizmo
        || selectedObjectIndex < 0
        || selectedObjectIndex >= static_cast<int>(scene->objects.size()))
    {
        return;
    }

    const SceneObject& object = scene->objects[selectedObjectIndex];
    ImVec2 center;
    if (!projectWorldToViewport(object.translation, center))
    {
        return;
    }

    ImDrawList* drawList = ImGui::GetForegroundDrawList();
    const ImU32 red = IM_COL32(235, 64, 52, 255);
    const ImU32 green = IM_COL32(52, 199, 89, 255);
    const ImU32 blue = IM_COL32(64, 156, 255, 255);
    const ImU32 white = IM_COL32(255, 255, 255, 230);

    const float objectSize = std::max(object.scale.x, std::max(object.scale.y, object.scale.z));
    const float axisLength = glm::max(0.75f, objectSize * 0.75f);

    if (activeTransformMode == TransformMode::Rotate)
    {
        drawList->AddCircle(center, 28.0f, red, 48, 2.0f);
        drawList->AddCircle(center, 36.0f, green, 48, 2.0f);
        drawList->AddCircle(center, 44.0f, blue, 48, 2.0f);
    }
    else
    {
        const glm::vec3 axes[3] = {
            glm::vec3(axisLength, 0.0f, 0.0f),
            glm::vec3(0.0f, axisLength, 0.0f),
            glm::vec3(0.0f, 0.0f, axisLength)
        };
        const ImU32 colors[3] = { red, green, blue };
        const char labels[3] = { 'X', 'Y', 'Z' };

        for (int axis = 0; axis < 3; ++axis)
        {
            ImVec2 endpoint;
            if (!projectWorldToViewport(object.translation + axes[axis], endpoint))
            {
                continue;
            }

            drawList->AddLine(center, endpoint, colors[axis], 2.5f);
            if (activeTransformMode == TransformMode::Translate)
            {
                drawArrowHead(drawList, center, endpoint, colors[axis]);
            }
            else
            {
                drawList->AddRectFilled(
                    ImVec2(endpoint.x - 4.0f, endpoint.y - 4.0f),
                    ImVec2(endpoint.x + 4.0f, endpoint.y + 4.0f),
                    colors[axis],
                    1.0f);
            }
            drawList->AddText(ImVec2(endpoint.x + 6.0f, endpoint.y - 8.0f), colors[axis], std::string(1, labels[axis]).c_str());
        }
    }

    drawList->AddCircleFilled(center, 4.5f, white);
    drawList->AddText(ImVec2(center.x + 10.0f, center.y + 10.0f), white, getTransformModeLabel(activeTransformMode));
}
}

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}

void RenderImGui()
{
    mouseOverImGuiWindow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    renderMainMenuBar();

    if (showAnalyticsWindow)
    {
        renderAnalyticsWindow();
    }

    if (showSceneObjectsWindow)
    {
        renderSceneObjectsWindow();
    }

    drawViewportGizmoOverlay();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWindow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    try
    {
        scene = new Scene(sceneFile);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Scene load failed: " << e.what() << std::endl;
        return 1;
    }

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage()
{
    const float samples = static_cast<float>(glm::max(iteration, 1));
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index] / samples;
            pix = applyDisplayTransform(pix, imguiData->ExposureValue, imguiData->ToneMapModeValue);
            img.setPixel(width - 1 - x, y, pix);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    img.savePNG(filename);
}

void runCuda()
{
    syncCameraState();

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (!cudaSceneInitialized)
    {
        pathtraceInit(scene);
        cudaSceneInitialized = true;
    }
    else if (scene->gpuDynamicDataDirty)
    {
        pathtraceUpdateScene(scene);
    }

    if (iteration == 0)
    {
        pathtraceResetAccumulation();
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    }
    else
    {
        saveImage();
        pathtraceFree();
        cudaSceneInitialized = false;
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS)
    {
        return;
    }

    switch (key)
    {
    case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
    case GLFW_KEY_S:
        saveImage();
        break;
    case GLFW_KEY_SPACE:
    {
        camchanged = true;
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        cam.lookAt = ogLookAt;
        break;
    }
    case GLFW_KEY_W:
        activeTransformMode = TransformMode::Translate;
        break;
    case GLFW_KEY_E:
        activeTransformMode = TransformMode::Rotate;
        break;
    case GLFW_KEY_R:
        activeTransformMode = TransformMode::Scale;
        break;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    glfwGetCursorPos(window, &lastX, &lastY);

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            if (isAltPressed(window))
            {
                leftMousePressed = true;
                objectManipulationActive = false;
                return;
            }

            selectedObjectIndex = pickSceneObject(lastX, lastY);
            if (selectedObjectIndex >= 0)
            {
                const SceneObject& object = scene->objects[selectedObjectIndex];
                objectManipulationActive = true;
                manipulationStartX = lastX;
                manipulationStartY = lastY;
                manipulationStartTranslation = object.translation;
                manipulationStartRotation = object.rotation;
                manipulationStartScale = object.scale;
            }
            else
            {
                objectManipulationActive = false;
            }
        }
        else if (action == GLFW_RELEASE)
        {
            leftMousePressed = false;
            objectManipulationActive = false;
        }
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        rightMousePressed = (action == GLFW_PRESS) && isAltPressed(window);
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE)
    {
        middleMousePressed = (action == GLFW_PRESS) && isAltPressed(window);
    }
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX && ypos == lastY)
    {
        return;
    }

    if (objectManipulationActive)
    {
        applyViewportManipulation(xpos, ypos);
    }
    else if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= static_cast<float>((xpos - lastX) / width);
        theta -= static_cast<float>((ypos - lastY) / height);
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += static_cast<float>((ypos - lastY) / height);
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        syncCameraState();
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= static_cast<float>(xpos - lastX) * right * 0.01f;
        cam.lookAt += static_cast<float>(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}







