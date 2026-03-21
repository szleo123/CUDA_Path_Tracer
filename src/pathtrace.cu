#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <vector>
#include <utility>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#define checkCUDAResult(err, msg) checkCUDAResultFn((err), (msg), FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaError_t err = cudaPeekAtLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

void checkCUDAResultFn(cudaError_t err, const char* msg, const char* file, int line)
{
#if ERRORCHECK
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    const unsigned int depthBits = (static_cast<unsigned int>(depth) & 0x1FFu) << 22;
    const unsigned int iterBits = static_cast<unsigned int>(iter) & 0x003FFFFFu;
    const unsigned int seed = utilhash(0x80000000u | depthBits | iterBits)
        ^ utilhash(static_cast<unsigned int>(index));
    return thrust::default_random_engine(seed);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(
    uchar4* pbo,
    glm::ivec2 resolution,
    int iter,
    glm::vec3* image,
    float exposureValue,
    int toneMapMode)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / static_cast<float>(iter);
        pix = applyDisplayTransform(pix, exposureValue, toneMapMode);

        glm::ivec3 color;
        color.x = glm::clamp(static_cast<int>(pix.x * 255.0f), 0, 255);
        color.y = glm::clamp(static_cast<int>(pix.y * 255.0f), 0, 255);
        color.z = glm::clamp(static_cast<int>(pix.z * 255.0f), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static Triangle* dev_triangles = NULL;
static TriangleBvhNode* dev_triangleBvhNodes = NULL;
static MeshInstance* dev_meshInstances = NULL;
static ScenePrimitive* dev_scenePrimitives = NULL;
static SceneBvhNode* dev_sceneBvhNodes = NULL;
static TextureData* dev_textures = NULL;
static glm::vec4* dev_texturePixels = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_materialSortKeys = NULL;
static int* dev_lightGeomIndices = NULL;
static float* dev_geomLightSelectionPmf = NULL;
static float* dev_environmentTexelPmf = NULL;
static float* dev_environmentAliasProb = NULL;
static int* dev_environmentAliasIndex = NULL;
static int hst_lightGeomCount = 0;
static size_t dev_geomsCount = 0;
static size_t dev_materialsCount = 0;
static size_t dev_trianglesCount = 0;
static size_t dev_triangleBvhNodeCount = 0;
static size_t dev_meshInstanceCount = 0;
static size_t dev_scenePrimitiveCount = 0;
static size_t dev_sceneBvhNodeCount = 0;
static size_t dev_textureCount = 0;
static size_t dev_texturePixelCount = 0;
static int dev_environmentWidth = 0;
static int dev_environmentHeight = 0;

__host__ __device__ inline float geomEmissiveArea(const Geom& geom, const Material& material);

template <typename T>
void uploadVectorToDevice(T*& devicePtr, size_t& deviceCount, const std::vector<T>& hostData)
{
    if (hostData.empty())
    {
        if (devicePtr != nullptr)
        {
            checkCUDAResult(cudaFree(devicePtr), "free empty device buffer");
            devicePtr = nullptr;
        }
        deviceCount = 0;
        return;
    }

    if (deviceCount != hostData.size())
    {
        if (devicePtr != nullptr)
        {
            checkCUDAResult(cudaFree(devicePtr), "resize device buffer");
            devicePtr = nullptr;
        }
        checkCUDAResult(cudaMalloc(&devicePtr, hostData.size() * sizeof(T)), "allocate device buffer");
        deviceCount = hostData.size();
    }

    checkCUDAResult(
        cudaMemcpy(devicePtr, hostData.data(), hostData.size() * sizeof(T), cudaMemcpyHostToDevice),
        "upload device buffer");
}

float computeEmissiveGeomWeight(const Geom& geom, const Material& material)
{
    if (material.emittance <= 0.0f)
    {
        return 0.0f;
    }

    const glm::vec3 emittedPower = material.color * material.emittance;
    const float luminance = 0.2126f * emittedPower.r + 0.7152f * emittedPower.g + 0.0722f * emittedPower.b;
    return fmaxf(geomEmissiveArea(geom, material) * luminance, 0.0f);
}

void uploadEnvironmentSamplingData(const Scene* scene)
{
    if (dev_environmentTexelPmf != nullptr)
    {
        checkCUDAResult(cudaFree(dev_environmentTexelPmf), "free environment texel pmf");
        dev_environmentTexelPmf = nullptr;
    }
    if (dev_environmentAliasProb != nullptr)
    {
        checkCUDAResult(cudaFree(dev_environmentAliasProb), "free environment alias probabilities");
        dev_environmentAliasProb = nullptr;
    }
    if (dev_environmentAliasIndex != nullptr)
    {
        checkCUDAResult(cudaFree(dev_environmentAliasIndex), "free environment alias indices");
        dev_environmentAliasIndex = nullptr;
    }
    dev_environmentWidth = 0;
    dev_environmentHeight = 0;

    const EnvironmentSettings& environment = scene->state.environment;
    if (environment.useProceduralSky || environment.textureId < 0)
    {
        return;
    }
    if (environment.textureId >= static_cast<int>(scene->textures.size()))
    {
        return;
    }

    const TextureData& texture = scene->textures[environment.textureId];
    if (texture.width <= 0 || texture.height <= 0)
    {
        return;
    }

    const int width = texture.width;
    const int height = texture.height;
    const size_t texelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<float> texelPmf(texelCount, 0.0f);

    float totalWeight = 0.0f;
    for (int y = 0; y < height; ++y)
    {
        const float theta = PI * ((static_cast<float>(y) + 0.5f) / static_cast<float>(height));
        const float sinTheta = fmaxf(sinf(theta), EPSILON);
        const int rowOffset = y * width;

        for (int x = 0; x < width; ++x)
        {
            const glm::vec3 radiance = glm::vec3(scene->texturePixels[texture.pixelOffset + rowOffset + x]);
            const float luminance = 0.2126f * radiance.r + 0.7152f * radiance.g + 0.0722f * radiance.b;
            const float weight = fmaxf(luminance, 0.0f) * sinTheta;
            texelPmf[rowOffset + x] = weight;
            totalWeight += weight;
        }
    }

    if (totalWeight <= EPSILON)
    {
        const float uniformTexelPmf = 1.0f / static_cast<float>(texelCount);
        for (float& pmf : texelPmf)
        {
            pmf = uniformTexelPmf;
        }
    }
    else
    {
        for (float& pmf : texelPmf)
        {
            pmf /= totalWeight;
        }
    }

    std::vector<float> aliasProb(texelCount, 1.0f);
    std::vector<int> aliasIndex(texelCount, 0);
    std::vector<float> scaledProb(texelCount, 0.0f);
    std::vector<int> smallIndices;
    std::vector<int> largeIndices;
    smallIndices.reserve(texelCount);
    largeIndices.reserve(texelCount);

    const float texelCountFloat = static_cast<float>(texelCount);
    for (size_t i = 0; i < texelCount; ++i)
    {
        scaledProb[i] = texelPmf[i] * texelCountFloat;
        aliasIndex[i] = static_cast<int>(i);
        if (scaledProb[i] < 1.0f)
        {
            smallIndices.push_back(static_cast<int>(i));
        }
        else
        {
            largeIndices.push_back(static_cast<int>(i));
        }
    }

    while (!smallIndices.empty() && !largeIndices.empty())
    {
        const int smallIndex = smallIndices.back();
        smallIndices.pop_back();
        const int largeIndex = largeIndices.back();
        largeIndices.pop_back();

        aliasProb[smallIndex] = scaledProb[smallIndex];
        aliasIndex[smallIndex] = largeIndex;

        scaledProb[largeIndex] = (scaledProb[largeIndex] + scaledProb[smallIndex]) - 1.0f;
        if (scaledProb[largeIndex] < 1.0f)
        {
            smallIndices.push_back(largeIndex);
        }
        else
        {
            largeIndices.push_back(largeIndex);
        }
    }

    for (const int index : largeIndices)
    {
        aliasProb[index] = 1.0f;
        aliasIndex[index] = index;
    }
    for (const int index : smallIndices)
    {
        aliasProb[index] = 1.0f;
        aliasIndex[index] = index;
    }

    checkCUDAResult(cudaMalloc(&dev_environmentTexelPmf, texelPmf.size() * sizeof(float)), "allocate environment texel pmf");
    checkCUDAResult(
        cudaMemcpy(dev_environmentTexelPmf, texelPmf.data(), texelPmf.size() * sizeof(float), cudaMemcpyHostToDevice),
        "upload environment texel pmf");
    checkCUDAResult(cudaMalloc(&dev_environmentAliasProb, aliasProb.size() * sizeof(float)), "allocate environment alias probabilities");
    checkCUDAResult(
        cudaMemcpy(dev_environmentAliasProb, aliasProb.data(), aliasProb.size() * sizeof(float), cudaMemcpyHostToDevice),
        "upload environment alias probabilities");
    checkCUDAResult(cudaMalloc(&dev_environmentAliasIndex, aliasIndex.size() * sizeof(int)), "allocate environment alias indices");
    checkCUDAResult(
        cudaMemcpy(dev_environmentAliasIndex, aliasIndex.data(), aliasIndex.size() * sizeof(int), cudaMemcpyHostToDevice),
        "upload environment alias indices");

    dev_environmentWidth = width;
    dev_environmentHeight = height;
}

void uploadLightGeomData(const Scene* scene)
{
    std::vector<int> lightGeomIndices;
    std::vector<float> geomLightSelectionPmf(scene->geoms.size(), 0.0f);
    std::vector<float> lightWeights;
    lightGeomIndices.reserve(scene->geoms.size());
    lightWeights.reserve(scene->geoms.size());

    float totalWeight = 0.0f;
    for (int i = 0; i < static_cast<int>(scene->geoms.size()); ++i)
    {
        const Geom& g = scene->geoms[i];
        const Material& material = scene->materials[g.materialid];
        const float weight = computeEmissiveGeomWeight(g, material);
        if (weight > 0.0f)
        {
            lightGeomIndices.push_back(i);
            lightWeights.push_back(weight);
            totalWeight += weight;
        }
    }

    if (dev_lightGeomIndices != nullptr)
    {
        checkCUDAResult(cudaFree(dev_lightGeomIndices), "free light geom indices");
        dev_lightGeomIndices = nullptr;
    }
    if (dev_geomLightSelectionPmf != nullptr)
    {
        checkCUDAResult(cudaFree(dev_geomLightSelectionPmf), "free geom light pmf");
        dev_geomLightSelectionPmf = nullptr;
    }

    hst_lightGeomCount = static_cast<int>(lightGeomIndices.size());
    if (hst_lightGeomCount > 0)
    {
        if (totalWeight <= EPSILON)
        {
            const float uniformWeight = 1.0f / static_cast<float>(hst_lightGeomCount);
            for (int lightGeomIdx : lightGeomIndices)
            {
                geomLightSelectionPmf[lightGeomIdx] = uniformWeight;
            }
        }
        else
        {
            for (int i = 0; i < hst_lightGeomCount; ++i)
            {
                geomLightSelectionPmf[lightGeomIndices[i]] = lightWeights[i] / totalWeight;
            }
        }

        checkCUDAResult(cudaMalloc(&dev_lightGeomIndices, hst_lightGeomCount * sizeof(int)), "allocate light geom indices");
        checkCUDAResult(
            cudaMemcpy(dev_lightGeomIndices, lightGeomIndices.data(), hst_lightGeomCount * sizeof(int), cudaMemcpyHostToDevice),
            "upload light geom indices");
    }

    if (!geomLightSelectionPmf.empty())
    {
        checkCUDAResult(cudaMalloc(&dev_geomLightSelectionPmf, geomLightSelectionPmf.size() * sizeof(float)), "allocate geom light pmf");
        checkCUDAResult(
            cudaMemcpy(
                dev_geomLightSelectionPmf,
                geomLightSelectionPmf.data(),
                geomLightSelectionPmf.size() * sizeof(float),
                cudaMemcpyHostToDevice),
            "upload geom light pmf");
    }
}

struct PathTerminated
{
    __host__ __device__ bool operator()(const PathSegment& p) const
    {
        return p.remainingBounces <= 0;
    }
};

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    uploadVectorToDevice(dev_geoms, dev_geomsCount, scene->geoms);
    uploadVectorToDevice(dev_materials, dev_materialsCount, scene->materials);
    uploadVectorToDevice(dev_triangles, dev_trianglesCount, scene->triangles);
    uploadVectorToDevice(dev_triangleBvhNodes, dev_triangleBvhNodeCount, scene->triangleBvhNodes);
    uploadVectorToDevice(dev_meshInstances, dev_meshInstanceCount, scene->meshInstances);
    uploadVectorToDevice(dev_scenePrimitives, dev_scenePrimitiveCount, scene->scenePrimitives);
    uploadVectorToDevice(dev_sceneBvhNodes, dev_sceneBvhNodeCount, scene->sceneBvhNodes);
    uploadVectorToDevice(dev_textures, dev_textureCount, scene->textures);
    uploadVectorToDevice(dev_texturePixels, dev_texturePixelCount, scene->texturePixels);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_materialSortKeys, pixelcount * sizeof(int));
    uploadLightGeomData(scene);
    uploadEnvironmentSamplingData(scene);
    scene->gpuDynamicDataDirty = false;

    checkCUDAError("pathtraceInit");
}

void pathtraceUpdateScene(Scene* scene)
{
    hst_scene = scene;
    uploadVectorToDevice(dev_geoms, dev_geomsCount, scene->geoms);
    uploadVectorToDevice(dev_materials, dev_materialsCount, scene->materials);
    uploadVectorToDevice(dev_meshInstances, dev_meshInstanceCount, scene->meshInstances);
    uploadVectorToDevice(dev_scenePrimitives, dev_scenePrimitiveCount, scene->scenePrimitives);
    uploadVectorToDevice(dev_sceneBvhNodes, dev_sceneBvhNodeCount, scene->sceneBvhNodes);
    uploadLightGeomData(scene);
    scene->gpuDynamicDataDirty = false;
    checkCUDAError("pathtraceUpdateScene");
}

void pathtraceResetAccumulation()
{
    if (hst_scene == nullptr || dev_image == nullptr)
    {
        return;
    }

    const Camera& cam = hst_scene->state.camera;
    const size_t pixelcount = static_cast<size_t>(cam.resolution.x) * static_cast<size_t>(cam.resolution.y);
    checkCUDAResult(cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3)), "reset accumulation");
    checkCUDAError("pathtraceResetAccumulation");
}

void pathtraceDownloadImage()
{
    if (hst_scene == nullptr || dev_image == nullptr)
    {
        return;
    }

    const Camera& cam = hst_scene->state.camera;
    const size_t pixelcount = static_cast<size_t>(cam.resolution.x) * static_cast<size_t>(cam.resolution.y);
    checkCUDAResult(
        cudaMemcpy(
            hst_scene->state.image.data(),
            dev_image,
            pixelcount * sizeof(glm::vec3),
            cudaMemcpyDeviceToHost),
        "copy image from device");
    checkCUDAError("pathtraceDownloadImage");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_triangles);
    cudaFree(dev_triangleBvhNodes);
    cudaFree(dev_meshInstances);
    cudaFree(dev_scenePrimitives);
    cudaFree(dev_sceneBvhNodes);
    cudaFree(dev_textures);
    cudaFree(dev_texturePixels);
    cudaFree(dev_intersections);
    cudaFree(dev_materialSortKeys);
    cudaFree(dev_lightGeomIndices);
    cudaFree(dev_geomLightSelectionPmf);
    cudaFree(dev_environmentTexelPmf);
    cudaFree(dev_environmentAliasProb);
    cudaFree(dev_environmentAliasIndex);
    dev_image = nullptr;
    dev_paths = nullptr;
    dev_geoms = nullptr;
    dev_materials = nullptr;
    dev_triangles = nullptr;
    dev_triangleBvhNodes = nullptr;
    dev_meshInstances = nullptr;
    dev_scenePrimitives = nullptr;
    dev_sceneBvhNodes = nullptr;
    dev_textures = nullptr;
    dev_texturePixels = nullptr;
    dev_intersections = nullptr;
    dev_materialSortKeys = nullptr;
    dev_lightGeomIndices = nullptr;
    dev_geomLightSelectionPmf = nullptr;
    dev_environmentTexelPmf = nullptr;
    dev_environmentAliasProb = nullptr;
    dev_environmentAliasIndex = nullptr;
    hst_lightGeomCount = 0;
    dev_geomsCount = 0;
    dev_materialsCount = 0;
    dev_trianglesCount = 0;
    dev_triangleBvhNodeCount = 0;
    dev_meshInstanceCount = 0;
    dev_scenePrimitiveCount = 0;
    dev_sceneBvhNodeCount = 0;
    dev_textureCount = 0;
    dev_texturePixelCount = 0;
    dev_environmentWidth = 0;
    dev_environmentHeight = 0;
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // Stochastic sampled antialiasing: jitter the ray origin on the image plane.
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jitterX = u01(rng);
        float jitterY = u01(rng);
        float sx = ((float)x + jitterX) - (float)cam.resolution.x * 0.5f;
        float sy = ((float)y + jitterY) - (float)cam.resolution.y * 0.5f;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * sx
            - cam.up * cam.pixelLength.y * sy
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.lastBsdfPdf = 1.0f;
        segment.lastBounceWasDelta = 1;
        segment.ignoreTriangleIndex = -1;
    }
}

struct TraceHit
{
    float t;
    glm::vec3 point;
    glm::vec3 shadingNormal;
    glm::vec3 geometricNormal;
    glm::vec3 tangents[MAX_TEXTURE_UV_SETS];
    float tangentSigns[MAX_TEXTURE_UV_SETS];
    int tangentSetMask;
    glm::vec2 uv[MAX_TEXTURE_UV_SETS];
    int uvSetMask;
    int geomId;
    int materialId;
    int triangleIndex;
};

__device__ inline bool intersectAabb(
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

    tEntry = tMin;
    return true;
}

__device__ inline Ray transformRay(const Ray& ray, const glm::mat4& transform)
{
    Ray transformedRay;
    transformedRay.origin = multiplyMV(transform, glm::vec4(ray.origin, 1.0f));
    transformedRay.direction = glm::normalize(multiplyMV(transform, glm::vec4(ray.direction, 0.0f)));
    return transformedRay;
}

__device__ inline void transformTraceHit(
    const MeshInstance& meshInstance,
    const Ray& worldRay,
    TraceHit& hit)
{
    hit.point = multiplyMV(meshInstance.transform, glm::vec4(hit.point, 1.0f));
    hit.shadingNormal = glm::normalize(multiplyMV(meshInstance.invTranspose, glm::vec4(hit.shadingNormal, 0.0f)));
    hit.geometricNormal = glm::normalize(multiplyMV(meshInstance.invTranspose, glm::vec4(hit.geometricNormal, 0.0f)));
    for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
    {
        if (glm::dot(hit.tangents[uvSet], hit.tangents[uvSet]) > EPSILON)
        {
            const glm::vec3 transformedTangent = multiplyMV(meshInstance.transform, glm::vec4(hit.tangents[uvSet], 0.0f));
            hit.tangents[uvSet] = glm::dot(transformedTangent, transformedTangent) > EPSILON
                ? glm::normalize(transformedTangent)
                : glm::vec3(0.0f);
        }
    }
    if (glm::dot(hit.shadingNormal, hit.geometricNormal) < 0.0f)
    {
        hit.shadingNormal = -hit.shadingNormal;
    }
    hit.t = glm::length(hit.point - worldRay.origin);
}

__device__ bool traverseTriangleBvhClosest(
    const Ray& ray,
    const Triangle* triangles,
    const TriangleBvhNode* nodes,
    int nodeCount,
    int rootNodeIndex,
    int ignoreTriangleIndex,
    float maxDistance,
    TraceHit& outHit)
{
    if (triangles == nullptr || nodes == nullptr || nodeCount <= 0 || rootNodeIndex < 0 || rootNodeIndex >= nodeCount)
    {
        return false;
    }

    int stack[RENDER_CONFIG_MAX_TRIANGLE_BVH_STACK_SIZE];
    int stackSize = 0;
    stack[stackSize++] = rootNodeIndex;
    bool hit = false;
    outHit.t = maxDistance;

    while (stackSize > 0)
    {
        const TriangleBvhNode& node = nodes[stack[--stackSize]];
        float nodeEntry = 0.0f;
        if (!intersectAabb(ray, node.bboxMin, node.bboxMax, outHit.t, nodeEntry))
        {
            continue;
        }

        if (node.triCount > 0)
        {
            for (int i = 0; i < node.triCount; ++i)
            {
                const int triangleIndex = node.leftFirst + i;
                if (triangleIndex == ignoreTriangleIndex)
                {
                    continue;
                }
                const Triangle& triangle = triangles[triangleIndex];
                glm::vec3 trianglePoint;
                glm::vec3 shadingNormal;
                glm::vec3 geometricNormal;
                glm::vec2 uv[MAX_TEXTURE_UV_SETS];
                int uvSetMask = 0;
                glm::vec3 tangents[MAX_TEXTURE_UV_SETS];
                float tangentSigns[MAX_TEXTURE_UV_SETS];
                int tangentSetMask = 0;
                const float t = triangleIntersectionTest(
                    triangle,
                    ray,
                    trianglePoint,
                    shadingNormal,
                    geometricNormal,
                    uv,
                    uvSetMask,
                    tangents,
                    tangentSigns,
                    tangentSetMask);
                if (t > MIN_INTERSECTION_T && t < outHit.t)
                {
                    outHit.t = t;
                    outHit.point = trianglePoint;
                    outHit.shadingNormal = shadingNormal;
                    outHit.geometricNormal = geometricNormal;
                    outHit.uvSetMask = uvSetMask;
                    outHit.tangentSetMask = tangentSetMask;
                    for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
                    {
                        outHit.uv[uvSet] = uv[uvSet];
                        outHit.tangents[uvSet] = tangents[uvSet];
                        outHit.tangentSigns[uvSet] = tangentSigns[uvSet];
                    }
                    outHit.geomId = -1;
                    outHit.materialId = triangle.materialId;
                    outHit.triangleIndex = triangleIndex;
                    hit = true;
                }
            }
            continue;
        }

        const int leftChild = node.leftFirst;
        const int rightChild = node.rightChild;
        float leftEntry = 0.0f;
        float rightEntry = 0.0f;
        const bool hitLeft = intersectAabb(ray, nodes[leftChild].bboxMin, nodes[leftChild].bboxMax, outHit.t, leftEntry);
        const bool hitRight = intersectAabb(ray, nodes[rightChild].bboxMin, nodes[rightChild].bboxMax, outHit.t, rightEntry);

        if (hitLeft && hitRight)
        {
            const bool leftFirst = leftEntry < rightEntry;
            if (stackSize + 2 <= RENDER_CONFIG_MAX_TRIANGLE_BVH_STACK_SIZE)
            {
                stack[stackSize++] = leftFirst ? rightChild : leftChild;
                stack[stackSize++] = leftFirst ? leftChild : rightChild;
            }
        }
        else if (hitLeft)
        {
            if (stackSize + 1 <= RENDER_CONFIG_MAX_TRIANGLE_BVH_STACK_SIZE)
            {
                stack[stackSize++] = leftChild;
            }
        }
        else if (hitRight)
        {
            if (stackSize + 1 <= RENDER_CONFIG_MAX_TRIANGLE_BVH_STACK_SIZE)
            {
                stack[stackSize++] = rightChild;
            }
        }
    }

    return hit;
}

__device__ bool intersectGeomDetailed(
    const Geom& geom,
    int geomIndex,
    const Ray& ray,
    float maxDistance,
    TraceHit& outHit)
{
    glm::vec3 point(0.0f);
    glm::vec3 normal(0.0f);
    bool outside = false;
    float hitT = -1.0f;

    if (geom.type == CUBE)
    {
        hitT = boxIntersectionTest(geom, ray, point, normal, outside);
    }
    else
    {
        hitT = sphereIntersectionTest(geom, ray, point, normal, outside);
    }

    if (hitT > MIN_INTERSECTION_T && hitT < maxDistance)
    {
        outHit.t = hitT;
        outHit.point = point;
        outHit.shadingNormal = normal;
        outHit.geometricNormal = normal;
        outHit.tangentSetMask = 0;
        outHit.uvSetMask = 0;
        for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
        {
            outHit.tangents[uvSet] = glm::vec3(0.0f);
            outHit.tangentSigns[uvSet] = 1.0f;
            outHit.uv[uvSet] = glm::vec2(0.0f);
        }
        outHit.geomId = geomIndex;
        outHit.materialId = geom.materialid;
        outHit.triangleIndex = -1;
        return true;
    }

    return false;
}

__device__ bool traceSceneClosest(
    const Ray& ray,
    float maxDistance,
    int ignoreGeomId,
    int ignoreTriangleIndex,
    const Geom* geoms,
    const MeshInstance* meshInstances,
    const ScenePrimitive* scenePrimitives,
    const SceneBvhNode* sceneBvhNodes,
    int sceneBvhNodeCount,
    const Triangle* triangles,
    const TriangleBvhNode* triangleBvhNodes,
    int triangleBvhNodeCount,
    TraceHit& outHit)
{
    outHit.t = maxDistance;
    outHit.geomId = -1;
    outHit.materialId = -1;
    outHit.uvSetMask = 0;
    outHit.tangentSetMask = 0;
    for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
    {
        outHit.uv[uvSet] = glm::vec2(0.0f);
        outHit.tangents[uvSet] = glm::vec3(0.0f);
        outHit.tangentSigns[uvSet] = 1.0f;
    }
    outHit.triangleIndex = -1;

    if (scenePrimitives == nullptr || sceneBvhNodes == nullptr || sceneBvhNodeCount <= 0)
    {
        return false;
    }

    int stack[RENDER_CONFIG_MAX_SCENE_BVH_STACK_SIZE];
    int stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0)
    {
        const SceneBvhNode& node = sceneBvhNodes[stack[--stackSize]];
        float nodeEntry = 0.0f;
        if (!intersectAabb(ray, node.bboxMin, node.bboxMax, outHit.t, nodeEntry))
        {
            continue;
        }

        if (node.primitiveCount > 0)
        {
            for (int i = 0; i < node.primitiveCount; ++i)
            {
                const ScenePrimitive& primitive = scenePrimitives[node.leftFirst + i];
                if (primitive.type == SCENE_PRIMITIVE_GEOM)
                {
                    if (primitive.index == ignoreGeomId)
                    {
                        continue;
                    }

                    TraceHit candidate{};
                    if (intersectGeomDetailed(geoms[primitive.index], primitive.index, ray, outHit.t, candidate))
                    {
                        outHit = candidate;
                    }
                }
                else if (primitive.type == SCENE_PRIMITIVE_MESH_INSTANCE)
                {
                    const MeshInstance& meshInstance = meshInstances[primitive.index];
                    TraceHit candidate{};
                    const Ray localRay = transformRay(ray, meshInstance.inverseTransform);
                    if (traverseTriangleBvhClosest(
                        localRay,
                        triangles,
                        triangleBvhNodes,
                        triangleBvhNodeCount,
                        meshInstance.bvhRootIndex,
                        ignoreTriangleIndex,
                        FLT_MAX,
                        candidate))
                    {
                        transformTraceHit(meshInstance, ray, candidate);
                        if (candidate.t > MIN_INTERSECTION_T && candidate.t < outHit.t)
                        {
                            outHit = candidate;
                        }
                    }
                }
            }
            continue;
        }

        const int leftChild = node.leftFirst;
        const int rightChild = node.rightChild;
        float leftEntry = 0.0f;
        float rightEntry = 0.0f;
        const bool hitLeft = intersectAabb(ray, sceneBvhNodes[leftChild].bboxMin, sceneBvhNodes[leftChild].bboxMax, outHit.t, leftEntry);
        const bool hitRight = intersectAabb(ray, sceneBvhNodes[rightChild].bboxMin, sceneBvhNodes[rightChild].bboxMax, outHit.t, rightEntry);

        if (hitLeft && hitRight)
        {
            const bool leftFirst = leftEntry < rightEntry;
            if (stackSize + 2 <= RENDER_CONFIG_MAX_SCENE_BVH_STACK_SIZE)
            {
                stack[stackSize++] = leftFirst ? rightChild : leftChild;
                stack[stackSize++] = leftFirst ? leftChild : rightChild;
            }
        }
        else if (hitLeft)
        {
            if (stackSize + 1 <= RENDER_CONFIG_MAX_SCENE_BVH_STACK_SIZE)
            {
                stack[stackSize++] = leftChild;
            }
        }
        else if (hitRight)
        {
            if (stackSize + 1 <= RENDER_CONFIG_MAX_SCENE_BVH_STACK_SIZE)
            {
                stack[stackSize++] = rightChild;
            }
        }
    }

    return outHit.materialId >= 0;
}

__global__ void computeIntersections(
    int num_paths,
    PathSegment* pathSegments,
    const Geom* geoms,
    const MeshInstance* meshInstances,
    const ScenePrimitive* scenePrimitives,
    const SceneBvhNode* sceneBvhNodes,
    int sceneBvhNodeCount,
    const Triangle* triangles,
    const TriangleBvhNode* triangleBvhNodes,
    int triangleBvhNodeCount,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
        if (pathSegment.remainingBounces <= 0)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = -1;
            intersections[path_index].tangentSetMask = 0;
            intersections[path_index].uvSetMask = 0;
            for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
            {
                intersections[path_index].tangents[uvSet] = glm::vec3(0.0f);
                intersections[path_index].tangentSigns[uvSet] = 1.0f;
                intersections[path_index].uv[uvSet] = glm::vec2(0.0f);
            }
            intersections[path_index].geomId = -1;
            intersections[path_index].triangleIndex = -1;
            return;
        }

        TraceHit hit{};
        if (!traceSceneClosest(
            pathSegment.ray,
            FLT_MAX,
            -1,
            pathSegment.ignoreTriangleIndex,
            geoms,
            meshInstances,
            scenePrimitives,
            sceneBvhNodes,
            sceneBvhNodeCount,
            triangles,
            triangleBvhNodes,
            triangleBvhNodeCount,
            hit))
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = -1;
            intersections[path_index].tangentSetMask = 0;
            intersections[path_index].uvSetMask = 0;
            for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
            {
                intersections[path_index].tangents[uvSet] = glm::vec3(0.0f);
                intersections[path_index].tangentSigns[uvSet] = 1.0f;
                intersections[path_index].uv[uvSet] = glm::vec2(0.0f);
            }
            intersections[path_index].geomId = -1;
            intersections[path_index].triangleIndex = -1;
        }
        else
        {
            intersections[path_index].t = hit.t;
            intersections[path_index].materialId = hit.materialId;
            intersections[path_index].surfaceNormal = hit.shadingNormal;
            intersections[path_index].geometricNormal = hit.geometricNormal;
            intersections[path_index].tangentSetMask = hit.tangentSetMask;
            intersections[path_index].uvSetMask = hit.uvSetMask;
            for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
            {
                intersections[path_index].tangents[uvSet] = hit.tangents[uvSet];
                intersections[path_index].tangentSigns[uvSet] = hit.tangentSigns[uvSet];
                intersections[path_index].uv[uvSet] = hit.uv[uvSet];
            }
            intersections[path_index].triangleIndex = hit.triangleIndex;
            intersections[path_index].geomId = hit.geomId;
        }
    }
}
__global__ void computeMaterialSortKeys(
    int num_paths,
    const ShadeableIntersection* intersections,
    int* sortKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        const ShadeableIntersection& isect = intersections[idx];
        sortKeys[idx] = (isect.t > 0.0f && isect.materialId >= 0) ? isect.materialId : INT_MAX;
    }
}

__host__ __device__ inline float powerHeuristic(float pdfA, float pdfB)
{
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, EPSILON);
}

__host__ __device__ inline float maxComponent(const glm::vec3& v)
{
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

__device__ inline glm::vec3 offsetRayOrigin(
    const glm::vec3& point,
    const glm::vec3& geometricNormal,
    const glm::vec3& direction)
{
    glm::vec3 offsetNormal = (glm::dot(direction, geometricNormal) < 0.0f) ? -geometricNormal : geometricNormal;
    return point + offsetNormal * RAY_ORIGIN_BIAS;
}

__device__ inline glm::vec3 offsetRayOriginAlongDirection(
    const glm::vec3& point,
    const glm::vec3& direction,
    float scale = 1.0f)
{
    return point + direction * (RAY_ORIGIN_BIAS * scale);
}

__device__ inline glm::vec4 sampleTexture(
    const TextureData& texture,
    const glm::vec2& uv,
    const glm::vec4* texturePixels)
{
    if (texture.width <= 0 || texture.height <= 0 || texturePixels == nullptr)
    {
        return glm::vec4(1.0f);
    }

    auto wrapCoord = [](float coord, int wrapMode) -> float
    {
        if (wrapMode == 33071)
        {
            return glm::clamp(coord, 0.0f, 1.0f);
        }
        if (wrapMode == 33648)
        {
            const float period = floorf(coord);
            const float frac = coord - period;
            return (static_cast<int>(period) & 1) ? (1.0f - frac) : frac;
        }
        return coord - floorf(coord);
    };

    float u = wrapCoord(uv.x, texture.wrapS);
    float v = wrapCoord(uv.y, texture.wrapT);
    float x = u * (texture.width - 1);
    float y = (texture.flipV ? (1.0f - v) : v) * (texture.height - 1);

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    if (texture.wrapS == 33071)
    {
        x0 = glm::clamp(x0, 0, texture.width - 1);
    }
    else if (texture.width > 0)
    {
        x0 %= texture.width;
        if (x0 < 0)
        {
            x0 += texture.width;
        }
    }
    if (texture.wrapT == 33071)
    {
        y0 = glm::clamp(y0, 0, texture.height - 1);
    }
    else if (texture.height > 0)
    {
        y0 %= texture.height;
        if (y0 < 0)
        {
            y0 += texture.height;
        }
    }
    int x1 = x0;
    if (texture.width > 1)
    {
        x1 = (texture.wrapS == 33071) ? glm::clamp(x0 + 1, 0, texture.width - 1) : ((x0 + 1) % texture.width);
    }
    int y1 = y0;
    if (texture.height > 1)
    {
        y1 = (texture.wrapT == 33071) ? glm::clamp(y0 + 1, 0, texture.height - 1) : ((y0 + 1) % texture.height);
    }
    float tx = x - floorf(x);
    float ty = y - floorf(y);

    const glm::vec4 c00 = texturePixels[texture.pixelOffset + y0 * texture.width + x0];
    const glm::vec4 c10 = texturePixels[texture.pixelOffset + y0 * texture.width + x1];
    const glm::vec4 c01 = texturePixels[texture.pixelOffset + y1 * texture.width + x0];
    const glm::vec4 c11 = texturePixels[texture.pixelOffset + y1 * texture.width + x1];
    const glm::vec4 c0 = glm::mix(c00, c10, tx);
    const glm::vec4 c1 = glm::mix(c01, c11, tx);
    return glm::mix(c0, c1, ty);
}

__device__ inline glm::vec3 sampleTextureRgb(
    const TextureData& texture,
    const glm::vec2& uv,
    const glm::vec4* texturePixels)
{
    return glm::vec3(sampleTexture(texture, uv, texturePixels));
}

__device__ inline glm::vec2 selectUvSet(
    const glm::vec2 uvSets[MAX_TEXTURE_UV_SETS],
    int uvSetMask,
    int requestedSet)
{
    const int clampedSet = glm::clamp(requestedSet, 0, MAX_TEXTURE_UV_SETS - 1);
    if (uvSetMask & (1 << clampedSet))
    {
        return uvSets[clampedSet];
    }
    for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
    {
        if (uvSetMask & (1 << uvSet))
        {
            return uvSets[uvSet];
        }
    }
    return glm::vec2(0.0f);
}

__device__ inline void selectTangentBasis(
    const ShadeableIntersection& intersection,
    int requestedSet,
    glm::vec3& tangent,
    float& tangentSign)
{
    const int clampedSet = glm::clamp(requestedSet, 0, MAX_TEXTURE_UV_SETS - 1);
    if (intersection.tangentSetMask & (1 << clampedSet))
    {
        tangent = intersection.tangents[clampedSet];
        tangentSign = intersection.tangentSigns[clampedSet];
        return;
    }
    if (intersection.tangentSetMask & 1)
    {
        tangent = intersection.tangents[0];
        tangentSign = intersection.tangentSigns[0];
        return;
    }
    tangent = glm::vec3(0.0f);
    tangentSign = 1.0f;
}

__device__ inline glm::vec3 evaluateMaterialColor(
    const Material& material,
    const ShadeableIntersection& intersection,
    const TextureData* textures,
    const glm::vec4* texturePixels,
    const glm::vec4* baseColorSample = nullptr)
{
    if (baseColorSample != nullptr)
    {
        return material.color * glm::vec3(*baseColorSample);
    }
    if (material.textureId < 0 || textures == nullptr || texturePixels == nullptr)
    {
        return material.color;
    }
    const glm::vec2 uv = selectUvSet(intersection.uv, intersection.uvSetMask, material.baseColorTexcoordSet);
    return material.color * sampleTextureRgb(textures[material.textureId], uv, texturePixels);
}

__device__ inline float evaluateMaterialAlpha(
    const Material& material,
    const ShadeableIntersection& intersection,
    const TextureData* textures,
    const glm::vec4* texturePixels,
    const glm::vec4* baseColorSample = nullptr)
{
    if (baseColorSample != nullptr)
    {
        return material.baseAlpha * baseColorSample->a;
    }
    if (material.textureId < 0 || textures == nullptr || texturePixels == nullptr)
    {
        return material.baseAlpha;
    }
    const TextureData& texture = textures[material.textureId];
    if (texture.isConstantOpaque)
    {
        return material.baseAlpha;
    }
    const glm::vec2 uv = selectUvSet(intersection.uv, intersection.uvSetMask, material.baseColorTexcoordSet);
    return material.baseAlpha * sampleTexture(texture, uv, texturePixels).a;
}

__device__ inline float evaluateMaterialAlphaFromUv(
    const Material& material,
    const glm::vec2& uv,
    const TextureData* textures,
    const glm::vec4* texturePixels)
{
    if (material.textureId < 0 || textures == nullptr || texturePixels == nullptr)
    {
        return material.baseAlpha;
    }
    const TextureData& texture = textures[material.textureId];
    if (texture.isConstantOpaque)
    {
        return material.baseAlpha;
    }
    return material.baseAlpha * sampleTexture(texture, uv, texturePixels).a;
}

__device__ inline glm::vec2 evaluateMetallicRoughness(
    const Material& material,
    const ShadeableIntersection& intersection,
    const TextureData* textures,
    const glm::vec4* texturePixels)
{
    float metallic = glm::clamp(material.metallic, 0.0f, 1.0f);
    float roughness = glm::clamp(material.roughness, 0.0f, 1.0f);
    if (material.metallicRoughnessTextureId >= 0 && textures != nullptr && texturePixels != nullptr)
    {
        const glm::vec2 uv = selectUvSet(intersection.uv, intersection.uvSetMask, material.metallicRoughnessTexcoordSet);
        const glm::vec3 mr = sampleTextureRgb(textures[material.metallicRoughnessTextureId], uv, texturePixels);
        roughness = glm::clamp(roughness * mr.g, 0.0f, 1.0f);
        metallic = glm::clamp(metallic * mr.b, 0.0f, 1.0f);
    }
    return glm::vec2(metallic, roughness);
}

__device__ inline glm::vec3 evaluateMaterialEmission(
    const Material& material,
    const ShadeableIntersection& intersection,
    const TextureData* textures,
    const glm::vec4* texturePixels)
{
    if (material.emittance <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 emission = material.emissiveColor;
    if (material.emissiveTextureId >= 0 && textures != nullptr && texturePixels != nullptr)
    {
        const glm::vec2 uv = selectUvSet(intersection.uv, intersection.uvSetMask, material.emissiveTexcoordSet);
        emission *= sampleTextureRgb(textures[material.emissiveTextureId], uv, texturePixels);
    }
    return emission * material.emittance;
}

__device__ inline float evaluateMaterialOcclusion(
    const Material& material,
    const ShadeableIntersection& intersection,
    const TextureData* textures,
    const glm::vec4* texturePixels)
{
    if (material.occlusionTextureId < 0 || textures == nullptr || texturePixels == nullptr)
    {
        return 1.0f;
    }
    const glm::vec2 uv = selectUvSet(intersection.uv, intersection.uvSetMask, material.occlusionTexcoordSet);
    const float ao = sampleTexture(textures[material.occlusionTextureId], uv, texturePixels).r;
    return glm::mix(1.0f, ao, glm::clamp(material.occlusionStrength, 0.0f, 1.0f));
}

__device__ inline glm::vec3 buildFallbackTangent(const glm::vec3& normal)
{
    if (fabsf(normal.x) < SQRT_OF_ONE_THIRD)
    {
        return glm::normalize(glm::cross(normal, glm::vec3(1, 0, 0)));
    }
    if (fabsf(normal.y) < SQRT_OF_ONE_THIRD)
    {
        return glm::normalize(glm::cross(normal, glm::vec3(0, 1, 0)));
    }
    return glm::normalize(glm::cross(normal, glm::vec3(0, 0, 1)));
}

__device__ inline glm::vec3 evaluateShadingNormal(
    const Material& material,
    const ShadeableIntersection& intersection,
    const TextureData* textures,
    const glm::vec4* texturePixels)
{
    glm::vec3 shadingNormal = glm::normalize(intersection.surfaceNormal);
    // Alpha-masked double-sided foliage cards do not have a stable tangent-space backface convention here.
    // Falling back to the interpolated vertex normal avoids the bright/white leaf failure mode.
    if (material.doubleSided && material.alphaMode == 1)
    {
        return shadingNormal;
    }
    if (material.normalTextureId < 0 || textures == nullptr || texturePixels == nullptr)
    {
        return shadingNormal;
    }

    glm::vec3 tangent(0.0f);
    float tangentSign = 1.0f;
    selectTangentBasis(intersection, material.normalTexcoordSet, tangent, tangentSign);
    if (glm::dot(tangent, tangent) > EPSILON)
    {
        tangent -= shadingNormal * glm::dot(shadingNormal, tangent);
    }
    if (glm::dot(tangent, tangent) <= EPSILON)
    {
        tangent = buildFallbackTangent(shadingNormal);
    }
    else
    {
        tangent = glm::normalize(tangent);
    }

    glm::vec3 bitangent = glm::cross(shadingNormal, tangent);
    if (tangentSign < 0.0f)
    {
        bitangent = -bitangent;
    }
    if (glm::dot(bitangent, bitangent) <= EPSILON)
    {
        return shadingNormal;
    }
    bitangent = glm::normalize(bitangent);

    const glm::vec2 uv = selectUvSet(intersection.uv, intersection.uvSetMask, material.normalTexcoordSet);
    glm::vec3 tangentSpaceNormal = sampleTextureRgb(textures[material.normalTextureId], uv, texturePixels) * 2.0f - glm::vec3(1.0f);
    tangentSpaceNormal.x *= material.normalTextureScale;
    tangentSpaceNormal.y *= material.normalTextureScale;
    if (glm::dot(tangentSpaceNormal, tangentSpaceNormal) <= EPSILON)
    {
        return shadingNormal;
    }
    tangentSpaceNormal = glm::normalize(tangentSpaceNormal);

    glm::vec3 mappedNormal = glm::normalize(
        tangent * tangentSpaceNormal.x
        + bitangent * tangentSpaceNormal.y
        + shadingNormal * tangentSpaceNormal.z);
    if (glm::dot(mappedNormal, intersection.geometricNormal) < 0.0f)
    {
        mappedNormal = -mappedNormal;
    }
    return mappedNormal;
}

__device__ inline Material evaluateShadingMaterial(
    const Material& material,
    const ShadeableIntersection& intersection,
    const TextureData* textures,
    const glm::vec4* texturePixels,
    const glm::vec4* baseColorSample = nullptr)
{
    Material shadingMaterial = material;
    shadingMaterial.color = evaluateMaterialColor(material, intersection, textures, texturePixels, baseColorSample);
    const glm::vec2 metallicRoughness = evaluateMetallicRoughness(material, intersection, textures, texturePixels);
    shadingMaterial.metallic = metallicRoughness.x;
    shadingMaterial.roughness = metallicRoughness.y;
    const float occlusion = evaluateMaterialOcclusion(material, intersection, textures, texturePixels);
    shadingMaterial.color *= occlusion;
    shadingMaterial.specularColor = glm::mix(material.specularColor, shadingMaterial.color, shadingMaterial.metallic);
    shadingMaterial.hasReflective = fmaxf(shadingMaterial.hasReflective, shadingMaterial.metallic);
    shadingMaterial.hasRefractive = fmaxf(shadingMaterial.hasRefractive, shadingMaterial.transmissionFactor);
    return shadingMaterial;
}

__device__ inline glm::vec3 evaluateTextureOnlyColor(
    const Material& material,
    const ShadeableIntersection& intersection,
    const TextureData* textures,
    const glm::vec4* texturePixels)
{
    if (material.textureId < 0 || textures == nullptr || texturePixels == nullptr)
    {
        return glm::vec3(1.0f, 0.0f, 1.0f);
    }
    const glm::vec2 uv = selectUvSet(intersection.uv, intersection.uvSetMask, material.baseColorTexcoordSet);
    return sampleTextureRgb(textures[material.textureId], uv, texturePixels);
}

__device__ inline glm::vec2 directionToEnvironmentUv(glm::vec3 direction, float rotationDegrees)
{
    const float rotationRadians = rotationDegrees * (PI / 180.0f);
    const float cosTheta = cosf(rotationRadians);
    const float sinTheta = sinf(rotationRadians);
    const glm::vec3 rotatedDirection(
        cosTheta * direction.x - sinTheta * direction.z,
        direction.y,
        sinTheta * direction.x + cosTheta * direction.z);

    const float phi = atan2f(rotatedDirection.z, rotatedDirection.x);
    const float theta = acosf(glm::clamp(rotatedDirection.y, -1.0f, 1.0f));
    const float u = phi / TWO_PI + 0.5f;
    const float v = theta / PI;
    return glm::vec2(u, v);
}

__device__ inline glm::vec3 environmentUvToDirection(glm::vec2 uv, float rotationDegrees)
{
    const float phi = (uv.x - 0.5f) * TWO_PI;
    const float theta = glm::clamp(uv.y, 0.0f, 1.0f) * PI;
    const float sinTheta = sinf(theta);
    const glm::vec3 rotatedDirection(
        cosf(phi) * sinTheta,
        cosf(theta),
        sinf(phi) * sinTheta);

    const float rotationRadians = rotationDegrees * (PI / 180.0f);
    const float cosRot = cosf(rotationRadians);
    const float sinRot = sinf(rotationRadians);
    return glm::vec3(
        cosRot * rotatedDirection.x + sinRot * rotatedDirection.z,
        rotatedDirection.y,
        -sinRot * rotatedDirection.x + cosRot * rotatedDirection.z);
}

__device__ inline glm::vec3 evaluateProceduralSky(
    const EnvironmentSettings& environment,
    glm::vec3 direction)
{
    const float upness = glm::clamp(direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
    const float horizonBlend = 1.0f - fabsf(direction.y);

    glm::vec3 skyColor = glm::mix(environment.horizonColor, environment.zenithColor, upness);
    skyColor = glm::mix(skyColor, environment.horizonColor, horizonBlend * horizonBlend * 0.35f);
    const glm::vec3 groundBlend = glm::mix(environment.groundColor, environment.horizonColor, upness);

    return direction.y >= 0.0f ? skyColor : groundBlend;
}

__device__ inline glm::vec3 sampleEnvironment(
    const EnvironmentSettings& environment,
    glm::vec3 direction,
    const TextureData* textures,
    const glm::vec4* texturePixels)
{
    glm::vec3 radiance(0.0f);
    if (!environment.useProceduralSky && environment.textureId >= 0)
    {
        const glm::vec2 uv = directionToEnvironmentUv(direction, environment.rotation);
        radiance = sampleTextureRgb(textures[environment.textureId], uv, texturePixels);
    }
    else
    {
        radiance = evaluateProceduralSky(environment, direction);
    }

    return radiance * environment.intensity;
}

__device__ inline bool hasEnvironmentImportanceSampling(
    const EnvironmentSettings& environment,
    const TextureData* textures,
    const float* environmentTexelPmf,
    const float* environmentAliasProb,
    const int* environmentAliasIndex,
    int environmentWidth,
    int environmentHeight)
{
    return !environment.useProceduralSky
        && environment.textureId >= 0
        && textures != nullptr
        && environmentTexelPmf != nullptr
        && environmentAliasProb != nullptr
        && environmentAliasIndex != nullptr
        && environmentWidth > 0
        && environmentHeight > 0;
}

__device__ inline float evaluateEnvironmentPdf(
    const EnvironmentSettings& environment,
    glm::vec3 direction,
    const TextureData* textures,
    const float* environmentTexelPmf,
    int environmentWidth,
    int environmentHeight)
{
    if (environment.useProceduralSky
        || environment.textureId < 0
        || textures == nullptr
        || environmentTexelPmf == nullptr
        || environmentWidth <= 0
        || environmentHeight <= 0)
    {
        return 0.0f;
    }

    const glm::vec2 uv = directionToEnvironmentUv(direction, environment.rotation);
    const float wrappedU = uv.x - floorf(uv.x);
    const float clampedV = glm::clamp(uv.y, 0.0f, 1.0f - 1e-6f);
    const int x = glm::clamp(static_cast<int>(wrappedU * environmentWidth), 0, environmentWidth - 1);
    const int y = glm::clamp(static_cast<int>(clampedV * environmentHeight), 0, environmentHeight - 1);
    const float texelPmf = environmentTexelPmf[y * environmentWidth + x];
    if (texelPmf <= 0.0f)
    {
        return 0.0f;
    }

    const float theta = PI * ((static_cast<float>(y) + 0.5f) / static_cast<float>(environmentHeight));
    const float sinTheta = fmaxf(sinf(theta), EPSILON);
    const float texelSolidAngle = (TWO_PI / static_cast<float>(environmentWidth))
        * (PI / static_cast<float>(environmentHeight))
        * sinTheta;
    return texelPmf / fmaxf(texelSolidAngle, EPSILON);
}

__device__ inline bool sampleEnvironmentLight(
    thrust::default_random_engine& rng,
    const EnvironmentSettings& environment,
    const TextureData* textures,
    const glm::vec4* texturePixels,
    const float* environmentTexelPmf,
    const float* environmentAliasProb,
    const int* environmentAliasIndex,
    int environmentWidth,
    int environmentHeight,
    glm::vec3& outDirection,
    glm::vec3& outRadiance,
    float& outPdf)
{
    if (!hasEnvironmentImportanceSampling(
            environment,
            textures,
            environmentTexelPmf,
            environmentAliasProb,
            environmentAliasIndex,
            environmentWidth,
            environmentHeight))
    {
        outPdf = 0.0f;
        return false;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    const int texelCount = environmentWidth * environmentHeight;
    const float texelXi = u01(rng) * static_cast<float>(texelCount);
    const int candidate = glm::min(static_cast<int>(texelXi), texelCount - 1);
    const int texelIndex = (u01(rng) < environmentAliasProb[candidate])
        ? candidate
        : environmentAliasIndex[candidate];
    const int row = texelIndex / environmentWidth;
    const int col = texelIndex - row * environmentWidth;
    const float jitterU = u01(rng);
    const float jitterV = u01(rng);
    const glm::vec2 uv(
        (static_cast<float>(col) + jitterU) / static_cast<float>(environmentWidth),
        (static_cast<float>(row) + jitterV) / static_cast<float>(environmentHeight));

    outDirection = environmentUvToDirection(uv, environment.rotation);
    outRadiance = sampleTextureRgb(textures[environment.textureId], uv, texturePixels) * environment.intensity;
    outPdf = evaluateEnvironmentPdf(
        environment,
        outDirection,
        textures,
        environmentTexelPmf,
        environmentWidth,
        environmentHeight);
    return outPdf > 0.0f;
}

__device__ inline glm::vec3 evaluateMeshUvCheckerColor(const glm::vec2& uv)
{
    const glm::vec2 wrapped = glm::vec2(
        uv.x - floorf(uv.x),
        uv.y - floorf(uv.y));

    const int checkX = static_cast<int>(floorf(wrapped.x * 16.0f));
    const int checkY = static_cast<int>(floorf(wrapped.y * 16.0f));
    const bool odd = ((checkX + checkY) & 1) != 0;
    return odd
        ? glm::vec3(wrapped.x, wrapped.y, 1.0f - wrapped.x)
        : glm::vec3(1.0f - wrapped.y, wrapped.x, wrapped.y);
}

__device__ inline glm::vec3 evaluateMeshDebugColor(
    int renderDebugMode,
    const Material& material,
    const Material& shadingMaterial,
    const ShadeableIntersection& intersection,
    const TextureData* textures,
    const glm::vec4* texturePixels)
{
    switch (renderDebugMode)
    {
    case RENDER_DEBUG_MESH_UV_CHECKER:
        return evaluateMeshUvCheckerColor(selectUvSet(intersection.uv, intersection.uvSetMask, material.baseColorTexcoordSet));
    case RENDER_DEBUG_MESH_BASE_COLOR:
        return shadingMaterial.color;
    case RENDER_DEBUG_MESH_TEXTURE_ONLY:
        return evaluateTextureOnlyColor(material, intersection, textures, texturePixels);
    default:
        return shadingMaterial.color;
    }
}

__host__ __device__ inline float geomEmissiveArea(const Geom& geom, const Material& material)
{
    (void)material;
    if (geom.type == SPHERE)
    {
        const float radius = 0.5f * glm::max(geom.scale.x, glm::max(geom.scale.y, geom.scale.z));
        return 4.0f * PI * radius * radius;
    }

    const float sx = geom.scale.x;
    const float sy = geom.scale.y;
    const float sz = geom.scale.z;
    return 2.0f * (sx * sy + sy * sz + sz * sx);
}

__device__ inline void samplePointOnSphere(
    const Geom& geom,
    thrust::default_random_engine& rng,
    glm::vec3& point,
    glm::vec3& normal)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const float z = 1.0f - 2.0f * u01(rng);
    const float phi = 2.0f * PI * u01(rng);
    const float r = sqrtf(glm::max(0.0f, 1.0f - z * z));
    const glm::vec3 localNormal(r * cosf(phi), z, r * sinf(phi));
    const glm::vec3 localPoint = 0.5f * localNormal;
    point = multiplyMV(geom.transform, glm::vec4(localPoint, 1.0f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(localNormal, 0.0f)));
}

__device__ inline void samplePointOnCube(
    const Geom& geom,
    thrust::default_random_engine& rng,
    glm::vec3& point,
    glm::vec3& normal)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const float sx = geom.scale.x;
    const float sy = geom.scale.y;
    const float sz = geom.scale.z;
    const float aXY = sx * sy;
    const float aYZ = sy * sz;
    const float aXZ = sx * sz;
    const float totalArea = 2.0f * (aXY + aYZ + aXZ);
    float sample = u01(rng) * totalArea;

    glm::vec3 localPoint(0.0f);
    glm::vec3 localNormal(0.0f);
    const float u = u01(rng) - 0.5f;
    const float v = u01(rng) - 0.5f;

    if ((sample -= aYZ) < 0.0f)
    {
        localPoint = glm::vec3(-0.5f, u, v);
        localNormal = glm::vec3(-1.0f, 0.0f, 0.0f);
    }
    else if ((sample -= aYZ) < 0.0f)
    {
        localPoint = glm::vec3(0.5f, u, v);
        localNormal = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    else if ((sample -= aXZ) < 0.0f)
    {
        localPoint = glm::vec3(u, -0.5f, v);
        localNormal = glm::vec3(0.0f, -1.0f, 0.0f);
    }
    else if ((sample -= aXZ) < 0.0f)
    {
        localPoint = glm::vec3(u, 0.5f, v);
        localNormal = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else if ((sample -= aXY) < 0.0f)
    {
        localPoint = glm::vec3(u, v, -0.5f);
        localNormal = glm::vec3(0.0f, 0.0f, -1.0f);
    }
    else
    {
        localPoint = glm::vec3(u, v, 0.5f);
        localNormal = glm::vec3(0.0f, 0.0f, 1.0f);
    }

    point = multiplyMV(geom.transform, glm::vec4(localPoint, 1.0f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(localNormal, 0.0f)));
}
__device__ float evaluateLightPdf(
    const Geom& lightGeom,
    const Material& lightMaterial,
    float lightSelectionPmf,
    const glm::vec3& shadingPoint,
    const glm::vec3& wi,
    const glm::vec3& lightPoint,
    const glm::vec3& lightNormal)
{
    if (lightSelectionPmf <= 0.0f)
    {
        return 0.0f;
    }

    const float lightArea = geomEmissiveArea(lightGeom, lightMaterial);
    if (lightArea <= EPSILON)
    {
        return 0.0f;
    }

    const glm::vec3 toLight = lightPoint - shadingPoint;
    const float dist2 = glm::dot(toLight, toLight);
    const float cosLight = glm::max(0.0f, glm::dot(lightNormal, -wi));
    if (dist2 <= EPSILON || cosLight <= 0.0f)
    {
        return 0.0f;
    }

    return lightSelectionPmf * (dist2 / (cosLight * lightArea));
}

__device__ int sampleLightIndex(
    thrust::default_random_engine& rng,
    const int* lightGeomIndices,
    const float* geomLightSelectionPmf,
    int lightGeomCount,
    float& outLightSelectionPmf)
{
    if (lightGeomCount <= 0)
    {
        outLightSelectionPmf = 0.0f;
        return -1;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    const float xi = u01(rng);
    float cumulative = 0.0f;

    for (int i = 0; i < lightGeomCount; ++i)
    {
        const int geomIndex = lightGeomIndices[i];
        const float pmf = geomLightSelectionPmf[geomIndex];
        cumulative += pmf;
        if (xi <= cumulative || i == lightGeomCount - 1)
        {
            outLightSelectionPmf = pmf;
            return geomIndex;
        }
    }

    outLightSelectionPmf = 0.0f;
    return -1;
}

__device__ glm::vec3 computeShadowTransmittance(
    const Ray& shadowRay,
    float maxDistance,
    int ignoreGeomId,
    int initialIgnoreTriangleIndex,
    const Geom* geoms,
    const MeshInstance* meshInstances,
    const ScenePrimitive* scenePrimitives,
    const SceneBvhNode* sceneBvhNodes,
    int sceneBvhNodeCount,
    const Material* materials,
    const TextureData* textures,
    const glm::vec4* texturePixels,
    const Triangle* triangles,
    const TriangleBvhNode* triangleBvhNodes,
    int triangleBvhNodeCount)
{
    Ray currentRay = shadowRay;
    float remainingDistance = maxDistance;
    int ignoreTriangleIndex = initialIgnoreTriangleIndex;
    glm::vec3 transmittance(1.0f);

    for (int step = 0; step < RENDER_CONFIG_SHADOW_TRANSMITTANCE_MAX_STEPS && remainingDistance > MIN_INTERSECTION_T; ++step)
    {
        TraceHit hit{};
        if (!traceSceneClosest(
            currentRay,
            remainingDistance,
            ignoreGeomId,
            ignoreTriangleIndex,
            geoms,
            meshInstances,
            scenePrimitives,
            sceneBvhNodes,
            sceneBvhNodeCount,
            triangles,
            triangleBvhNodes,
            triangleBvhNodeCount,
            hit))
        {
            return transmittance;
        }

        const Material& material = materials[hit.materialId];
        if (material.alphaMode == 1)
        {
            const glm::vec2 uv = selectUvSet(hit.uv, hit.uvSetMask, material.baseColorTexcoordSet);
            const float alpha = evaluateMaterialAlphaFromUv(material, uv, textures, texturePixels);
            if (alpha < material.alphaCutoff)
            {
                currentRay.origin = offsetRayOriginAlongDirection(
                    hit.point,
                    currentRay.direction,
                    RENDER_CONFIG_THIN_SURFACE_DIRECTION_OFFSET_SCALE);
                remainingDistance -= hit.t;
                ignoreTriangleIndex = hit.triangleIndex;
                continue;
            }
        }
        else if (material.alphaMode == 2)
        {
            const glm::vec2 uv = selectUvSet(hit.uv, hit.uvSetMask, material.baseColorTexcoordSet);
            const float alpha = glm::clamp(evaluateMaterialAlphaFromUv(material, uv, textures, texturePixels), 0.0f, 1.0f);
            const float transmission = glm::clamp(material.transmissionFactor, 0.0f, 1.0f);
            const float blockedFraction = alpha * (1.0f - transmission);
            transmittance *= glm::vec3(1.0f - blockedFraction);
            if (maxComponent(transmittance) <= EPSILON)
            {
                return glm::vec3(0.0f);
            }
            currentRay.origin = offsetRayOriginAlongDirection(
                hit.point,
                currentRay.direction,
                RENDER_CONFIG_THIN_SURFACE_DIRECTION_OFFSET_SCALE);
            remainingDistance -= hit.t;
            ignoreTriangleIndex = hit.triangleIndex;
            continue;
        }

        return glm::vec3(0.0f);
    }

    return remainingDistance > MIN_INTERSECTION_T ? transmittance : glm::vec3(1.0f);
}

__device__ bool applyRussianRoulette(
    PathSegment& pathSegment,
    thrust::default_random_engine& rng)
{
    float survivalProbability = glm::clamp(
        maxComponent(pathSegment.color),
        RENDER_CONFIG_MIN_RUSSIAN_ROULETTE_SURVIVAL,
        RENDER_CONFIG_MAX_RUSSIAN_ROULETTE_SURVIVAL);
    thrust::uniform_real_distribution<float> u01(0, 1);
    if (u01(rng) > survivalProbability)
    {
        pathSegment.remainingBounces = 0;
        return false;
    }

    pathSegment.color /= survivalProbability;
    return true;
}

__global__ void shadeMaterialPaths(
    int iter,
    int num_paths,
    int traceDepth,
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    const Material* materials,
    const Geom* geoms,
    const MeshInstance* meshInstances,
    const ScenePrimitive* scenePrimitives,
    const SceneBvhNode* sceneBvhNodes,
    int sceneBvhNodeCount,
    int geoms_size,
    const Triangle* triangles,
    const TriangleBvhNode* triangleBvhNodes,
    int triangleBvhNodeCount,
    const TextureData* textures,
    const glm::vec4* texturePixels,
    EnvironmentSettings environment,
    const float* environmentTexelPmf,
    const float* environmentAliasProb,
    const int* environmentAliasIndex,
    int environmentWidth,
    int environmentHeight,
    const int* lightGeomIndices,
    const float* geomLightSelectionPmf,
    int lightGeomCount,
    int renderDebugMode,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths){
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& pathSegment = pathSegments[idx];

        if (pathSegment.remainingBounces <= 0) {
            return;
        }

        if (intersection.t > 0.0f) {
            const Material& material = materials[intersection.materialId];
            glm::vec4 baseColorSample(1.0f);
            const glm::vec4* baseColorSamplePtr = nullptr;
            if (material.textureId >= 0 && textures != nullptr && texturePixels != nullptr)
            {
                const glm::vec2 baseUv = selectUvSet(intersection.uv, intersection.uvSetMask, material.baseColorTexcoordSet);
                baseColorSample = sampleTexture(textures[material.textureId], baseUv, texturePixels);
                baseColorSamplePtr = &baseColorSample;
            }

            thrust::default_random_engine rng = makeSeededRandomEngine(
                iter,
                pathSegment.pixelIndex,
                pathSegment.remainingBounces);

            if (material.alphaMode == 1
                && evaluateMaterialAlpha(material, intersection, textures, texturePixels, baseColorSamplePtr) < material.alphaCutoff)
            {
                const glm::vec3 hitPoint = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                pathSegment.ray.origin = offsetRayOriginAlongDirection(
                    hitPoint,
                    pathSegment.ray.direction,
                    RENDER_CONFIG_THIN_SURFACE_DIRECTION_OFFSET_SCALE);
                pathSegment.lastBounceWasDelta = 1;
                pathSegment.lastBsdfPdf = 1.0f;
                pathSegment.ignoreTriangleIndex = intersection.triangleIndex;
                return;
            }

            if (material.alphaMode == 2)
            {
                const float alpha = glm::clamp(
                    evaluateMaterialAlpha(material, intersection, textures, texturePixels, baseColorSamplePtr),
                    0.0f,
                    1.0f);
                thrust::uniform_real_distribution<float> u01(0, 1);
                if (alpha <= EPSILON || u01(rng) > alpha)
                {
                    const glm::vec3 hitPoint = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                    pathSegment.ray.origin = offsetRayOriginAlongDirection(
                        hitPoint,
                        pathSegment.ray.direction,
                        RENDER_CONFIG_THIN_SURFACE_DIRECTION_OFFSET_SCALE);
                    pathSegment.lastBounceWasDelta = 1;
                    pathSegment.lastBsdfPdf = 1.0f;
                    pathSegment.ignoreTriangleIndex = intersection.triangleIndex;
                    return;
                }
            }

            Material shadingMaterial = evaluateShadingMaterial(material, intersection, textures, texturePixels, baseColorSamplePtr);
            intersection.surfaceNormal = evaluateShadingNormal(material, intersection, textures, texturePixels);
            pathSegment.ignoreTriangleIndex = -1;

            if (renderDebugMode != RENDER_DEBUG_NONE && intersection.geomId < 0)
            {
                const glm::vec3 debugColor = evaluateMeshDebugColor(
                    renderDebugMode,
                    material,
                    shadingMaterial,
                    intersection,
                    textures,
                    texturePixels);
                image[pathSegment.pixelIndex] += pathSegment.color * debugColor;
                pathSegment.remainingBounces = 0;
                return;
            }

            const glm::vec3 emission = evaluateMaterialEmission(material, intersection, textures, texturePixels);

            // If the material indicates that the object was a light, "light" the ray
            if (glm::length2(emission) > 0.0f) {
                float misWeight = 1.0f;
                if (!pathSegment.lastBounceWasDelta && lightGeomCount > 0 && intersection.geomId >= 0) {
                    const Geom& lightGeom = geoms[intersection.geomId];
                    glm::vec3 wi = glm::normalize(pathSegment.ray.direction);
                    float cosLight = glm::max(0.0f, glm::dot(intersection.surfaceNormal, -wi));
                    if (cosLight > 0.0f) {
                        const glm::vec3 lightPoint = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                        const float lightPdf = evaluateLightPdf(
                            lightGeom,
                            material,
                            geomLightSelectionPmf[intersection.geomId],
                            pathSegment.ray.origin,
                            wi,
                            lightPoint,
                            intersection.surfaceNormal);
                        if (lightPdf > 0.0f) {
                            misWeight = powerHeuristic(pathSegment.lastBsdfPdf, lightPdf);
                        }
                    }
                }

                pathSegment.color *= (emission * misWeight);
                image[pathSegment.pixelIndex] += pathSegment.color;
                pathSegment.remainingBounces = 0;
            }
            else {
                glm::vec3 intersect = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                glm::vec3 geometricNormal = intersection.geometricNormal;
                glm::vec3 shadingNormal = intersection.surfaceNormal;
                const glm::vec3 wo = -glm::normalize(pathSegment.ray.direction);
                if (shadingMaterial.doubleSided)
                {
                    if (glm::dot(shadingNormal, wo) < 0.0f)
                    {
                        shadingNormal = -shadingNormal;
                    }
                    if (glm::dot(geometricNormal, wo) < 0.0f)
                    {
                        geometricNormal = -geometricNormal;
                    }
                }
                else if (glm::dot(shadingNormal, pathSegment.ray.direction) > 0.0f) {
                    shadingNormal = -shadingNormal;
                }

                const bool useDirectLightSampling = materialHasNonDeltaBsdf(shadingMaterial);
                if (useDirectLightSampling) {
                    if (lightGeomCount > 0) {
                        float lightSelectionPmf = 0.0f;
                        int lightGeomIdx = sampleLightIndex(
                            rng,
                            lightGeomIndices,
                            geomLightSelectionPmf,
                            lightGeomCount,
                            lightSelectionPmf);
                        if (lightGeomIdx >= 0 && lightSelectionPmf > 0.0f) {
                        const Geom& lightGeom = geoms[lightGeomIdx];
                        const Material& lightMaterial = materials[lightGeom.materialid];

                        if (lightMaterial.emittance > 0.0f) {
                            glm::vec3 lightPoint;
                            glm::vec3 lightNormal;

                            if (lightGeom.type == SPHERE) {
                                samplePointOnSphere(lightGeom, rng, lightPoint, lightNormal);
                            }
                            else {
                                samplePointOnCube(lightGeom, rng, lightPoint, lightNormal);
                            }

                            glm::vec3 toLight = lightPoint - intersect;
                            float dist2 = glm::dot(toLight, toLight);
                            if (dist2 > EPSILON) {
                                float dist = sqrtf(dist2);
                                glm::vec3 wi = toLight / dist;
                                float cosSurface = glm::max(0.0f, glm::dot(shadingNormal, wi));
                                float cosLight = glm::max(0.0f, glm::dot(lightNormal, -wi));

                                if (cosSurface > 0.0f && cosLight > 0.0f) {
                                    Ray shadowRay;
                                    shadowRay.origin = (shadingMaterial.doubleSided && material.alphaMode == 1)
                                        ? offsetRayOriginAlongDirection(
                                            intersect,
                                            wi,
                                            RENDER_CONFIG_THIN_SURFACE_DIRECTION_OFFSET_SCALE)
                                        : offsetRayOrigin(intersect, geometricNormal, wi);
                                    shadowRay.direction = wi;

                                    glm::vec3 shadowTransmittance = computeShadowTransmittance(
                                        shadowRay,
                                        dist,
                                        lightGeomIdx,
                                        intersection.triangleIndex,
                                        geoms,
                                        meshInstances,
                                        scenePrimitives,
                                        sceneBvhNodes,
                                        sceneBvhNodeCount,
                                        materials,
                                        textures,
                                        texturePixels,
                                        triangles,
                                        triangleBvhNodes,
                                        triangleBvhNodeCount);
                                    if (glm::length2(shadowTransmittance) > 0.0f) {
                                        float lightPdf = evaluateLightPdf(
                                            lightGeom,
                                            lightMaterial,
                                            lightSelectionPmf,
                                            intersect,
                                            wi,
                                            lightPoint,
                                            lightNormal);
                                        glm::vec3 f = evaluateBsdf(shadingMaterial, wo, shadingNormal, wi);
                                        if (glm::length2(f) > 0.0f)
                                        {
                                            float bsdfPdf = evaluateBsdfPdf(shadingMaterial, wo, shadingNormal, wi);
                                            float misWeight = powerHeuristic(lightPdf, bsdfPdf);
                                            glm::vec3 Le = lightMaterial.color * lightMaterial.emittance;
                                            glm::vec3 directLi = pathSegment.color * shadowTransmittance * f * Le
                                                * (cosSurface * misWeight / fmaxf(lightPdf, EPSILON));
                                            image[pathSegment.pixelIndex] += directLi;
                                        }
                                    }
                                }
                            }
                        }
                        }
                    }

                    if (hasEnvironmentImportanceSampling(
                            environment,
                            textures,
                            environmentTexelPmf,
                            environmentAliasProb,
                            environmentAliasIndex,
                            environmentWidth,
                            environmentHeight))
                    {
                        glm::vec3 envWi(0.0f);
                        glm::vec3 envRadiance(0.0f);
                        float envPdf = 0.0f;
                        if (sampleEnvironmentLight(
                                rng,
                                environment,
                                textures,
                                texturePixels,
                                environmentTexelPmf,
                                environmentAliasProb,
                                environmentAliasIndex,
                                environmentWidth,
                                environmentHeight,
                                envWi,
                                envRadiance,
                                envPdf))
                        {
                            const float cosSurface = glm::max(0.0f, glm::dot(shadingNormal, envWi));
                            if (cosSurface > 0.0f)
                            {
                                Ray shadowRay;
                                shadowRay.origin = (shadingMaterial.doubleSided && material.alphaMode == 1)
                                    ? offsetRayOriginAlongDirection(
                                        intersect,
                                        envWi,
                                        RENDER_CONFIG_THIN_SURFACE_DIRECTION_OFFSET_SCALE)
                                    : offsetRayOrigin(intersect, geometricNormal, envWi);
                                shadowRay.direction = envWi;

                                glm::vec3 shadowTransmittance = computeShadowTransmittance(
                                    shadowRay,
                                    FLT_MAX,
                                    -1,
                                    intersection.triangleIndex,
                                    geoms,
                                    meshInstances,
                                    scenePrimitives,
                                    sceneBvhNodes,
                                    sceneBvhNodeCount,
                                    materials,
                                    textures,
                                    texturePixels,
                                    triangles,
                                    triangleBvhNodes,
                                    triangleBvhNodeCount);
                                if (glm::length2(shadowTransmittance) > 0.0f)
                                {
                                    const glm::vec3 f = evaluateBsdf(shadingMaterial, wo, shadingNormal, envWi);
                                    if (glm::length2(f) > 0.0f)
                                    {
                                        const float bsdfPdf = evaluateBsdfPdf(shadingMaterial, wo, shadingNormal, envWi);
                                        const float misWeight = powerHeuristic(envPdf, bsdfPdf);
                                        const glm::vec3 directLi = pathSegment.color * shadowTransmittance * f * envRadiance
                                            * (cosSurface * misWeight / fmaxf(envPdf, EPSILON));
                                        image[pathSegment.pixelIndex] += directLi;
                                    }
                                }
                            }
                        }
                    }
                }

                BSDFSample sample;
                scatterRay(pathSegment.ray, shadingNormal, shadingMaterial, rng, sample);
                if (!sample.isDelta && sample.pdf <= 0.0f) {
                    pathSegment.color = glm::vec3(0.0f);
                    pathSegment.remainingBounces = 0;
                    return;
                }

                pathSegment.color *= sample.pathWeight;
                pathSegment.lastBounceWasDelta = sample.isDelta;
                pathSegment.lastBsdfPdf = sample.isDelta ? 0.0f : sample.pdf;


                pathSegment.ray.origin = (shadingMaterial.doubleSided && material.alphaMode == 1)
                    ? offsetRayOriginAlongDirection(
                        intersect,
                        sample.direction,
                        RENDER_CONFIG_THIN_SURFACE_DIRECTION_OFFSET_SCALE)
                    : offsetRayOrigin(intersect, geometricNormal, sample.direction);
                pathSegment.ray.direction = sample.direction;
                pathSegment.ignoreTriangleIndex = (shadingMaterial.doubleSided && material.alphaMode == 1)
                    ? intersection.triangleIndex
                    : -1;
                pathSegment.remainingBounces--;

                int completedBounces = traceDepth - pathSegment.remainingBounces;
                if (pathSegment.remainingBounces > 0
                    && completedBounces >= RENDER_CONFIG_RUSSIAN_ROULETTE_START_BOUNCE
                    && !pathSegment.lastBounceWasDelta)
                {
                    applyRussianRoulette(pathSegment, rng);
                }
            }
        }
        else {
            float misWeight = 1.0f;
            if (!pathSegment.lastBounceWasDelta)
            {
                const float environmentPdf = evaluateEnvironmentPdf(
                    environment,
                    pathSegment.ray.direction,
                    textures,
                    environmentTexelPmf,
                    environmentWidth,
                    environmentHeight);
                if (environmentPdf > 0.0f)
                {
                    misWeight = powerHeuristic(pathSegment.lastBsdfPdf, environmentPdf);
                }
            }
            pathSegment.color *= sampleEnvironment(environment, pathSegment.ray.direction, textures, texturePixels) * misWeight;
            image[pathSegment.pixelIndex] += pathSegment.color;
            pathSegment.ignoreTriangleIndex = -1;
            pathSegment.remainingBounces = 0;
        }
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    (void)frame;
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int geoms_size = static_cast<int>(hst_scene->geoms.size());
    const int sceneBvhNodeCount = static_cast<int>(hst_scene->sceneBvhNodes.size());
    const int triangleBvhNodeCount = static_cast<int>(hst_scene->triangleBvhNodes.size());

    // 2D block for generating ray from camera
    const dim3 blockSize2d(RENDER_CONFIG_CAMERA_BLOCK_SIZE_X, RENDER_CONFIG_CAMERA_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = RENDER_CONFIG_PATH_TRACE_BLOCK_SIZE_1D;
    // Initialize array of path rays (using rays that come out of the camera)
    // You can pass the Camera object to that kernel.
    // Each path ray must carry at minimum a (ray, color) pair,
    // where color starts as the multiplicative identity, white = (1, 1, 1).
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = static_cast<int>(dev_path_end - dev_paths);
    const bool useMaterialSort = (guiData != NULL) && guiData->UseMaterialSort;
    const bool enableKernelTiming = (guiData != NULL) && guiData->EnableKernelTiming;
    const int sortEveryNIterations =
        (guiData != NULL && guiData->SortEveryNIterations > 0) ? guiData->SortEveryNIterations : 1;
    const int sortMaxBounce =
        (guiData != NULL && guiData->SortMaxBounce > 0) ? guiData->SortMaxBounce : traceDepth;
    const int sortMinPathCount =
        (guiData != NULL && guiData->SortMinPathCount > 0) ? guiData->SortMinPathCount : 1;
    float totalSortTimeMs = 0.0f;
    float totalShadeTimeMs = 0.0f;
    int totalShadedPaths = 0;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> sortTimings;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> shadeTimings;
    if (enableKernelTiming)
    {
        sortTimings.reserve(traceDepth);
        shadeTimings.reserve(traceDepth);
    }

    auto createTimingRange = []() {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        checkCUDAResult(cudaEventCreate(&start), "create timing start event");
        checkCUDAResult(cudaEventCreate(&stop), "create timing stop event");
        return std::make_pair(start, stop);
    };

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // tracing
        // Compute an intersection in the scene for each path ray.
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            num_paths,
            dev_paths,
            dev_geoms,
            dev_meshInstances,
            dev_scenePrimitives,
            dev_sceneBvhNodes,
            sceneBvhNodeCount,
            dev_triangles,
            dev_triangleBvhNodes,
            triangleBvhNodeCount,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        depth++;

        totalShadedPaths += num_paths;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate the next rays.
        const bool shouldSortThisIteration = ((iter - 1) % sortEveryNIterations) == 0;
        const bool shouldSortThisBounce = depth <= sortMaxBounce;
        const bool shouldSortThisPathCount = num_paths >= sortMinPathCount;

        if (useMaterialSort
            && shouldSortThisIteration
            && shouldSortThisBounce
            && shouldSortThisPathCount
            && num_paths > 1)
        {
            std::pair<cudaEvent_t, cudaEvent_t> sortTiming{};
            if (enableKernelTiming)
            {
                sortTiming = createTimingRange();
                checkCUDAResult(cudaEventRecord(sortTiming.first), "record sort start event");
            }
            dim3 numblocksSort = (num_paths + blockSize1d - 1) / blockSize1d;
            computeMaterialSortKeys<<<numblocksSort, blockSize1d>>>(
                num_paths,
                dev_intersections,
                dev_materialSortKeys
            );
            checkCUDAError("build material sort keys");

            thrust::sort_by_key(
                thrust::device,
                dev_materialSortKeys,
                dev_materialSortKeys + num_paths,
                dev_paths
            );
            checkCUDAError("sort paths by material");
            if (enableKernelTiming)
            {
                checkCUDAResult(cudaEventRecord(sortTiming.second), "record sort stop event");
                sortTimings.push_back(sortTiming);
            }

            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths,
                dev_paths,
                dev_geoms,
                dev_meshInstances,
                dev_scenePrimitives,
                dev_sceneBvhNodes,
                sceneBvhNodeCount,
                dev_triangles,
                dev_triangleBvhNodes,
                triangleBvhNodeCount,
                dev_intersections
            );
            checkCUDAError("retrace sorted paths");
        }

        std::pair<cudaEvent_t, cudaEvent_t> shadeTiming{};
        if (enableKernelTiming)
        {
            shadeTiming = createTimingRange();
            checkCUDAResult(cudaEventRecord(shadeTiming.first), "record shade start event");
        }
        shadeMaterialPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            traceDepth,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            dev_meshInstances,
            dev_scenePrimitives,
            dev_sceneBvhNodes,
            sceneBvhNodeCount,
            geoms_size,
            dev_triangles,
            dev_triangleBvhNodes,
            triangleBvhNodeCount,
            dev_textures,
            dev_texturePixels,
            hst_scene->state.environment,
            dev_environmentTexelPmf,
            dev_environmentAliasProb,
            dev_environmentAliasIndex,
            dev_environmentWidth,
            dev_environmentHeight,
            dev_lightGeomIndices,
            dev_geomLightSelectionPmf,
            hst_lightGeomCount,
            guiData ? guiData->RenderDebugModeValue : RENDER_DEBUG_NONE,
            dev_image
        );
        checkCUDAError("compute shading");
        if (enableKernelTiming)
        {
            checkCUDAResult(cudaEventRecord(shadeTiming.second), "record shade stop event");
            shadeTimings.push_back(shadeTiming);
        }

        // Stream compact terminated paths.
        PathSegment* new_end = thrust::remove_if(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            PathTerminated());
        checkCUDAError("thrust::remove_if");
        num_paths = static_cast<int>(new_end - dev_paths);

        iterationComplete = (num_paths == 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    const float exposureValue = (guiData != NULL) ? guiData->ExposureValue : 0.0f;
    const int toneMapMode = (guiData != NULL) ? guiData->ToneMapModeValue : TONEMAP_REINHARD;
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(
        pbo,
        cam.resolution,
        iter,
        dev_image,
        exposureValue,
        toneMapMode);

    auto accumulateTimingRanges =
        [](std::vector<std::pair<cudaEvent_t, cudaEvent_t>>& ranges, float& totalMs, const char* label) {
            for (const auto& range : ranges)
            {
                float elapsedMs = 0.0f;
                checkCUDAResult(cudaEventElapsedTime(&elapsedMs, range.first, range.second), label);
                totalMs += elapsedMs;
                checkCUDAResult(cudaEventDestroy(range.first), "destroy timing start event");
                checkCUDAResult(cudaEventDestroy(range.second), "destroy timing stop event");
            }
        };

    if (guiData != NULL)
    {
        if (enableKernelTiming)
        {
            accumulateTimingRanges(sortTimings, totalSortTimeMs, "measure sort elapsed time");
            accumulateTimingRanges(shadeTimings, totalShadeTimeMs, "measure shade elapsed time");
        }
        guiData->LastSortTimeMs = totalSortTimeMs;
        guiData->LastShadeTimeMs = totalShadeTimeMs;
        guiData->LastNumShadedPaths = totalShadedPaths;
    }

    checkCUDAError("pathtrace");
}































