#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ inline float saturate(float x)
{
    return glm::clamp(x, 0.0f, 1.0f);
}

__host__ __device__ inline float schlickReflectance(float cosTheta, float etaI, float etaT)
{
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    float oneMinusCos = 1.0f - cosTheta;
    float oneMinusCos5 = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    return r0 + (1.0f - r0) * oneMinusCos5;
}

__host__ __device__ inline glm::vec3 safeDivide(const glm::vec3& value, float scalar)
{
    return value / fmaxf(scalar, EPSILON);
}

__host__ __device__ void computeLobeProbabilities(
    const Material& m,
    float& pDiffuse,
    float& pReflect,
    float& pRefract)
{
    float reflectiveWeight = saturate(m.hasReflective);
    float refractiveWeight = saturate(m.hasRefractive);
    float diffuseWeight = fmaxf(0.0f, 1.0f - reflectiveWeight - refractiveWeight);

    float sumWeights = diffuseWeight + reflectiveWeight + refractiveWeight;
    if (sumWeights < EPSILON)
    {
        diffuseWeight = 1.0f;
        sumWeights = 1.0f;
    }

    pDiffuse = diffuseWeight / sumWeights;
    pReflect = reflectiveWeight / sumWeights;
    pRefract = refractiveWeight / sumWeights;
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng));
    float over = sqrt(1 - up * up);
    float around = u01(rng) * TWO_PI;

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    const Ray& ray,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng,
    BSDFSample& sample)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 wi = glm::normalize(ray.direction);
    glm::vec3 shadingNormal = (glm::dot(wi, normal) < 0.0f) ? normal : -normal;

    float pDiffuse, pReflect, pRefract;
    computeLobeProbabilities(m, pDiffuse, pReflect, pRefract);

    float xi = u01(rng);

    if (xi < pReflect)
    {
        sample.direction = glm::normalize(glm::reflect(wi, shadingNormal));
        sample.pathWeight = safeDivide(m.color, pReflect);
        sample.pdf = 0.0f;
        sample.isDelta = 1;
        return;
    }

    if (xi < (pReflect + pRefract))
    {
        float etaI = 1.0f;
        float etaT = (m.indexOfRefraction > 1.0f) ? m.indexOfRefraction : 1.5f;
        glm::vec3 n = normal;

        bool entering = glm::dot(wi, normal) < 0.0f;
        if (!entering)
        {
            n = -normal;
            float tmp = etaI;
            etaI = etaT;
            etaT = tmp;
        }

        float eta = etaI / etaT;
        float cosTheta = fminf(glm::dot(-wi, n), 1.0f);
        float sinTheta2 = fmaxf(0.0f, 1.0f - cosTheta * cosTheta);
        bool cannotRefract = (eta * eta * sinTheta2) > 1.0f;

        float fresnel = schlickReflectance(cosTheta, etaI, etaT);
        bool chooseReflect = cannotRefract || (u01(rng) < fresnel);
        glm::vec3 outDir = chooseReflect ? glm::reflect(wi, n) : glm::refract(wi, n, eta);
        if (glm::dot(outDir, outDir) < EPSILON)
        {
            outDir = glm::reflect(wi, n);
        }

        sample.direction = glm::normalize(outDir);
        sample.pathWeight = safeDivide(glm::vec3(1.0f), pRefract);
        sample.pdf = 0.0f;
        sample.isDelta = 1;
        return;
    }

    sample.direction = calculateRandomDirectionInHemisphere(shadingNormal, rng);
    float cosTheta = fmaxf(0.0f, glm::dot(sample.direction, shadingNormal));
    sample.pathWeight = safeDivide(m.color, pDiffuse);
    sample.pdf = pDiffuse * (cosTheta / PI);
    sample.isDelta = 0;
}
