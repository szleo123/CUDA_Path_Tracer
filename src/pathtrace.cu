#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <climits>
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
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

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
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_materialSortKeys = NULL;
static int* dev_lightGeomIndices = NULL;
static int hst_lightGeomCount = 0;

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

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_materialSortKeys, pixelcount * sizeof(int));

    std::vector<int> lightGeomIndices;
    lightGeomIndices.reserve(scene->geoms.size());
    for (int i = 0; i < static_cast<int>(scene->geoms.size()); ++i)
    {
        const Geom& g = scene->geoms[i];
        if (scene->materials[g.materialid].emittance > 0.0f)
        {
            lightGeomIndices.push_back(i);
        }
    }

    hst_lightGeomCount = static_cast<int>(lightGeomIndices.size());
    if (hst_lightGeomCount > 0)
    {
        cudaMalloc(&dev_lightGeomIndices, hst_lightGeomCount * sizeof(int));
        cudaMemcpy(
            dev_lightGeomIndices,
            lightGeomIndices.data(),
            hst_lightGeomCount * sizeof(int),
            cudaMemcpyHostToDevice
        );
    }
    else
    {
        dev_lightGeomIndices = nullptr;
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_materialSortKeys);
    cudaFree(dev_lightGeomIndices);
    dev_image = nullptr;
    dev_paths = nullptr;
    dev_geoms = nullptr;
    dev_materials = nullptr;
    dev_intersections = nullptr;
    dev_materialSortKeys = nullptr; 
    dev_lightGeomIndices = nullptr;
    hst_lightGeomCount = 0;

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
    }
}

__global__ void computeIntersections(
    int num_paths,
    PathSegment* pathSegments,
    const Geom* geoms,
    int geoms_size,
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
            intersections[path_index].geomId = -1;
            return;
        }

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            const Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > MIN_INTERSECTION_T && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = -1;
            intersections[path_index].geomId = -1;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].geomId = hit_geom_index;
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

constexpr int kRussianRouletteStartBounce = 4;
constexpr float kMinRussianRouletteSurvival = 0.1f;
constexpr float kMaxRussianRouletteSurvival = 0.95f;

struct SurfaceHit
{
    float t;
    glm::vec3 point;
    glm::vec3 normal;
    int geomId;
    int materialId;
};

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

__host__ __device__ inline float geomSurfaceArea(const Geom& geom)
{
    if (geom.type == SPHERE)
    {
        float rx = 0.5f * fabsf(geom.scale.x);
        float ry = 0.5f * fabsf(geom.scale.y);
        float rz = 0.5f * fabsf(geom.scale.z);
        float r = (rx + ry + rz) / 3.0f;
        return 4.0f * PI * r * r;
    }

    float sx = fabsf(geom.scale.x);
    float sy = fabsf(geom.scale.y);
    float sz = fabsf(geom.scale.z);
    return 2.0f * (sx * sy + sx * sz + sy * sz);
}

__host__ __device__ inline float geomEmissiveArea(const Geom& geom, const Material& mat)
{
    (void)mat;
    return geomSurfaceArea(geom);
}

__device__ void samplePointOnSphere(
    const Geom& sphere,
    thrust::default_random_engine& rng,
    glm::vec3& worldPoint,
    glm::vec3& worldNormal)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float z = 1.0f - 2.0f * u01(rng);
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = TWO_PI * u01(rng);
    glm::vec3 localDir(r * cosf(phi), r * sinf(phi), z);
    glm::vec3 localPoint = 0.5f * localDir;

    worldPoint = multiplyMV(sphere.transform, glm::vec4(localPoint, 1.0f));
    worldNormal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(localDir, 0.0f)));
}

__device__ void samplePointOnCube(
    const Geom& cube,
    thrust::default_random_engine& rng,
    glm::vec3& worldPoint,
    glm::vec3& worldNormal)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float sx = fabsf(cube.scale.x);
    float sy = fabsf(cube.scale.y);
    float sz = fabsf(cube.scale.z);
    float areaXY = sx * sy;
    float areaXZ = sx * sz;
    float areaYZ = sy * sz;
    float totalArea = 2.0f * (areaXY + areaXZ + areaYZ);

    float pick = u01(rng) * totalArea;
    float u = u01(rng) - 0.5f;
    float v = u01(rng) - 0.5f;
    bool positiveSide = u01(rng) < 0.5f;

    glm::vec3 localPoint(0.0f);
    glm::vec3 localNormal(0.0f);

    if (pick < 2.0f * areaXY)
    {
        localPoint = glm::vec3(u, v, positiveSide ? 0.5f : -0.5f);
        localNormal = glm::vec3(0.0f, 0.0f, positiveSide ? 1.0f : -1.0f);
    }
    else if (pick < 2.0f * (areaXY + areaXZ))
    {
        localPoint = glm::vec3(u, positiveSide ? 0.5f : -0.5f, v);
        localNormal = glm::vec3(0.0f, positiveSide ? 1.0f : -1.0f, 0.0f);
    }
    else
    {
        localPoint = glm::vec3(positiveSide ? 0.5f : -0.5f, u, v);
        localNormal = glm::vec3(positiveSide ? 1.0f : -1.0f, 0.0f, 0.0f);
    }

    worldPoint = multiplyMV(cube.transform, glm::vec4(localPoint, 1.0f));
    worldNormal = glm::normalize(multiplyMV(cube.invTranspose, glm::vec4(localNormal, 0.0f)));
}

__device__ bool intersectGeom(
    const Geom& geom,
    const Ray& ray,
    float maxDistance,
    float& t,
    glm::vec3& point,
    glm::vec3& normal)
{
    bool outside = true;
    float hitT = -1.0f;
    if (geom.type == CUBE)
    {
        hitT = boxIntersectionTest(geom, ray, point, normal, outside);
    }
    else if (geom.type == SPHERE)
    {
        hitT = sphereIntersectionTest(geom, ray, point, normal, outside);
    }

    if (hitT > MIN_INTERSECTION_T && hitT < maxDistance)
    {
        t = hitT;
        return true;
    }

    return false;
}

__device__ bool findClosestHit(
    const Ray& ray,
    float maxDistance,
    int ignoreGeomId,
    const Geom* geoms,
    int geoms_size,
    SurfaceHit& hit)
{
    hit.t = maxDistance;
    hit.geomId = -1;
    hit.materialId = -1;

    for (int i = 0; i < geoms_size; ++i)
    {
        if (i == ignoreGeomId)
        {
            continue;
        }

        SurfaceHit candidate{};
        if (intersectGeom(geoms[i], ray, hit.t, candidate.t, candidate.point, candidate.normal))
        {
            candidate.geomId = i;
            candidate.materialId = geoms[i].materialid;
            hit = candidate;
        }
    }

    return hit.geomId >= 0;
}

__device__ float evaluateDiffuseBsdfPdf(
    const Material& material,
    const glm::vec3& shadingNormal,
    const glm::vec3& wi)
{
    float pDiffuse, pReflect, pRefract;
    computeLobeProbabilities(material, pDiffuse, pReflect, pRefract);
    float cosTheta = glm::max(0.0f, glm::dot(shadingNormal, wi));
    return pDiffuse * (cosTheta / PI);
}

__device__ glm::vec3 computeShadowTransmittance(
    const Ray& shadowRay,
    float maxDistance,
    int ignoreGeomId,
    const Geom* geoms,
    const Material* materials,
    int geoms_size)
{
    (void)materials;
    SurfaceHit hit{};
    return findClosestHit(shadowRay, maxDistance, ignoreGeomId, geoms, geoms_size, hit)
        ? glm::vec3(0.0f)
        : glm::vec3(1.0f);
}

__device__ bool applyRussianRoulette(
    PathSegment& pathSegment,
    thrust::default_random_engine& rng)
{
    float survivalProbability = glm::clamp(
        maxComponent(pathSegment.color),
        kMinRussianRouletteSurvival,
        kMaxRussianRouletteSurvival);
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
    int geoms_size,
    const int* lightGeomIndices,
    int lightGeomCount,
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

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                float misWeight = 1.0f;
                if (!pathSegment.lastBounceWasDelta && lightGeomCount > 0 && intersection.geomId >= 0) {
                    const Geom& lightGeom = geoms[intersection.geomId];
                    float area = geomEmissiveArea(lightGeom, material);
                    glm::vec3 wi = glm::normalize(pathSegment.ray.direction);
                    float cosLight = glm::max(0.0f, glm::dot(intersection.surfaceNormal, -wi));
                    float dist2 = intersection.t * intersection.t * glm::dot(pathSegment.ray.direction, pathSegment.ray.direction);
                    if (area > EPSILON && cosLight > 0.0f) {
                        float lightPdf = (dist2 / (cosLight * area)) / lightGeomCount;
                        misWeight = powerHeuristic(pathSegment.lastBsdfPdf, lightPdf);
                    }
                }

                pathSegment.color *= (material.color * material.emittance * misWeight);
                image[pathSegment.pixelIndex] += pathSegment.color;
                pathSegment.remainingBounces = 0;
            }
            else {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
                glm::vec3 intersect = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                glm::vec3 geometricNormal = intersection.surfaceNormal;
                glm::vec3 shadingNormal = geometricNormal;
                if (glm::dot(shadingNormal, pathSegment.ray.direction) > 0.0f) {
                    shadingNormal = -shadingNormal;
                }

                if (lightGeomCount > 0) {
                    float pDiffuse, pReflect, pRefract;
                    computeLobeProbabilities(material, pDiffuse, pReflect, pRefract);
                    if (pDiffuse > 0.0f) {
                        thrust::uniform_real_distribution<float> u01(0, 1);
                        int picked = glm::min((int)(u01(rng) * lightGeomCount), lightGeomCount - 1);
                        int lightGeomIdx = lightGeomIndices[picked];
                        const Geom& lightGeom = geoms[lightGeomIdx];
                        const Material& lightMaterial = materials[lightGeom.materialid];
                        float lightArea = geomEmissiveArea(lightGeom, lightMaterial);

                        if (lightMaterial.emittance > 0.0f && lightArea > EPSILON) {
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
                                    shadowRay.origin = offsetRayOrigin(intersect, geometricNormal, wi);
                                    shadowRay.direction = wi;

                                    glm::vec3 shadowTransmittance = computeShadowTransmittance(
                                        shadowRay,
                                        dist,
                                        lightGeomIdx,
                                        geoms,
                                        materials,
                                        geoms_size);
                                    if (glm::length2(shadowTransmittance) > 0.0f) {
                                        float lightPdf = (dist2 / (cosLight * lightArea)) / lightGeomCount;
                                        float bsdfPdf = evaluateDiffuseBsdfPdf(material, shadingNormal, wi);
                                        float misWeight = powerHeuristic(lightPdf, bsdfPdf);
                                        glm::vec3 f = material.color / PI;
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

                BSDFSample sample;
                scatterRay(pathSegment.ray, geometricNormal, material, rng, sample);
                if (!sample.isDelta && sample.pdf <= 0.0f) {
                    pathSegment.color = glm::vec3(0.0f);
                    pathSegment.remainingBounces = 0;
                    return;
                }

                pathSegment.color *= sample.pathWeight;
                pathSegment.lastBounceWasDelta = sample.isDelta;
                pathSegment.lastBsdfPdf = sample.isDelta ? 0.0f : sample.pdf;


                pathSegment.ray.origin = offsetRayOrigin(intersect, geometricNormal, sample.direction);
                pathSegment.ray.direction = sample.direction;
                pathSegment.remainingBounces--;

                int completedBounces = traceDepth - pathSegment.remainingBounces;
                if (pathSegment.remainingBounces > 0
                    && completedBounces >= kRussianRouletteStartBounce
                    && !pathSegment.lastBounceWasDelta)
                {
                    applyRussianRoulette(pathSegment, rng);
                }
            }
        }
        else {
            pathSegment.color *= BACKGROUND_COLOR;
            image[pathSegment.pixelIndex] += pathSegment.color;
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

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;
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
    sortTimings.reserve(traceDepth);
    shadeTimings.reserve(traceDepth);

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
            geoms_size,
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
            const auto sortTiming = createTimingRange();
            checkCUDAResult(cudaEventRecord(sortTiming.first), "record sort start event");
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
                thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections))
            );
            checkCUDAError("sort paths by material");
            checkCUDAResult(cudaEventRecord(sortTiming.second), "record sort stop event");
            sortTimings.push_back(sortTiming);
        }

        const auto shadeTiming = createTimingRange();
        checkCUDAResult(cudaEventRecord(shadeTiming.first), "record shade start event");
        shadeMaterialPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            traceDepth,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            geoms_size,
            dev_lightGeomIndices,
            hst_lightGeomCount,
            dev_image
        );
        checkCUDAError("compute shading");
        checkCUDAResult(cudaEventRecord(shadeTiming.second), "record shade stop event");
        shadeTimings.push_back(shadeTiming);

        // Stream compact terminated paths.
        PathSegment* new_end = thrust::remove_if(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            PathTerminated());
        checkCUDAError("thrust::remove_if");
        num_paths = static_cast<int>(new_end - dev_paths);

        iterationComplete = (depth >= traceDepth) || (num_paths == 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    checkCUDAResult(
        cudaMemcpy(
            hst_scene->state.image.data(),
            dev_image,
            pixelcount * sizeof(glm::vec3),
            cudaMemcpyDeviceToHost),
        "copy image from device");

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

    accumulateTimingRanges(sortTimings, totalSortTimeMs, "measure sort elapsed time");
    accumulateTimingRanges(shadeTimings, totalShadeTimeMs, "measure shade elapsed time");

    if (guiData != NULL)
    {
        guiData->LastSortTimeMs = totalSortTimeMs;
        guiData->LastShadeTimeMs = totalShadeTimeMs;
        guiData->LastNumShadedPaths = totalShadedPaths;
    }

    checkCUDAError("pathtrace");
}







