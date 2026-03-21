#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng);

struct BSDFSample
{
    glm::vec3 direction;
    // Monte Carlo path weight for the sampled event: f * cos(theta) / p(sample).
    glm::vec3 pathWeight;
    float pdf;
    int isDelta;
};

__host__ __device__ void computeLobeProbabilities(
    const Material& m,
    float& pDiffuse,
    float& pReflect,
    float& pClearcoat,
    float& pRefract);

__host__ __device__ bool materialHasNonDeltaBsdf(const Material& material);

__host__ __device__ glm::vec3 evaluateBsdf(
    const Material& material,
    const glm::vec3& wo,
    const glm::vec3& shadingNormal,
    const glm::vec3& wi);

__host__ __device__ float evaluateBsdfPdf(
    const Material& material,
    const glm::vec3& wo,
    const glm::vec3& shadingNormal,
    const glm::vec3& wi);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method only samples the next event. The caller is responsible for
 * updating the path throughput and ray origin.
 */
__host__ __device__ void scatterRay(
    const Ray& ray,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    BSDFSample& sample);
