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

__host__ __device__ inline float materialRoughness(const Material& material)
{
    return glm::clamp(material.roughness, 0.0f, 1.0f);
}

__host__ __device__ inline float roughnessToAlpha(float roughness)
{
    const float perceptual = fmaxf(roughness, 0.02f);
    return perceptual * perceptual;
}

__host__ __device__ inline float ggxDistribution(float cosThetaH, float alpha)
{
    if (cosThetaH <= 0.0f)
    {
        return 0.0f;
    }

    const float alpha2 = alpha * alpha;
    const float denom = cosThetaH * cosThetaH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / fmaxf(PI * denom * denom, EPSILON);
}

__host__ __device__ inline float smithG1(float cosTheta, float alpha)
{
    if (cosTheta <= 0.0f)
    {
        return 0.0f;
    }

    const float alpha2 = alpha * alpha;
    const float cos2 = cosTheta * cosTheta;
    return 2.0f * cosTheta / fmaxf(cosTheta + sqrtf(alpha2 + (1.0f - alpha2) * cos2), EPSILON);
}

__host__ __device__ inline float smithG2(float cosThetaO, float cosThetaI, float alpha)
{
    return smithG1(cosThetaO, alpha) * smithG1(cosThetaI, alpha);
}

__host__ __device__ inline glm::vec3 fresnelSchlick(const glm::vec3& f0, float cosTheta)
{
    const float oneMinusCos = 1.0f - glm::clamp(cosTheta, 0.0f, 1.0f);
    const float factor = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    return f0 + (glm::vec3(1.0f) - f0) * factor;
}

__host__ __device__ inline float fresnelSchlickScalar(float f0, float cosTheta)
{
    const float oneMinusCos = 1.0f - glm::clamp(cosTheta, 0.0f, 1.0f);
    const float factor = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    return f0 + (1.0f - f0) * factor;
}

__host__ __device__ inline float materialClearcoatRoughness(const Material& material)
{
    return glm::clamp(material.clearcoatRoughness, 0.0f, 1.0f);
}

__host__ __device__ inline float materialTransmissionStrength(const Material& material)
{
    const float explicitTransmission = glm::clamp(material.transmissionFactor, 0.0f, 1.0f);
    if (explicitTransmission > 0.0f)
    {
        return explicitTransmission;
    }
    return glm::clamp(material.hasRefractive, 0.0f, 1.0f);
}

__host__ __device__ inline float evaluateClearcoatTransmission(
    const Material& material,
    const glm::vec3& wo,
    const glm::vec3& shadingNormal,
    const glm::vec3& wi)
{
    if (material.clearcoatFactor <= 0.0f)
    {
        return 1.0f;
    }

    const glm::vec3 halfVector = wo + wi;
    if (glm::dot(halfVector, halfVector) <= EPSILON)
    {
        return 1.0f;
    }

    const float clearcoatF = fresnelSchlickScalar(0.04f, fmaxf(0.0f, glm::dot(glm::normalize(halfVector), wo)));
    const float coatTransmission = 1.0f - glm::clamp(material.clearcoatFactor, 0.0f, 1.0f) * clearcoatF;
    return coatTransmission * coatTransmission;
}

__host__ __device__ inline glm::vec3 evaluateClearcoatBrdf(
    const Material& material,
    const glm::vec3& wo,
    const glm::vec3& shadingNormal,
    const glm::vec3& wi)
{
    const float clearcoatFactor = glm::clamp(material.clearcoatFactor, 0.0f, 1.0f);
    if (clearcoatFactor <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    const float cosThetaI = fmaxf(0.0f, glm::dot(shadingNormal, wi));
    const float cosThetaO = fmaxf(0.0f, glm::dot(shadingNormal, wo));
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    const glm::vec3 halfVector = wo + wi;
    if (glm::dot(halfVector, halfVector) <= EPSILON)
    {
        return glm::vec3(0.0f);
    }

    const glm::vec3 normalizedHalfVector = glm::normalize(halfVector);
    const float cosThetaH = fmaxf(0.0f, glm::dot(shadingNormal, normalizedHalfVector));
    const float cosWoH = fmaxf(0.0f, glm::dot(wo, normalizedHalfVector));
    if (cosThetaH <= 0.0f || cosWoH <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    const float alpha = roughnessToAlpha(materialClearcoatRoughness(material));
    const float D = ggxDistribution(cosThetaH, alpha);
    const float G = smithG2(cosThetaO, cosThetaI, alpha);
    const float F = fresnelSchlickScalar(0.04f, cosWoH);
    const float clearcoat = clearcoatFactor * (D * G * F / fmaxf(4.0f * cosThetaO * cosThetaI, EPSILON));
    return glm::vec3(clearcoat);
}

__host__ __device__ inline float evaluateClearcoatPdf(
    const Material& material,
    const glm::vec3& wo,
    const glm::vec3& shadingNormal,
    const glm::vec3& wi)
{
    if (material.clearcoatFactor <= 0.0f || materialClearcoatRoughness(material) <= 0.001f)
    {
        return 0.0f;
    }

    const glm::vec3 halfVector = wo + wi;
    if (glm::dot(halfVector, halfVector) <= EPSILON)
    {
        return 0.0f;
    }

    const glm::vec3 normalizedHalfVector = glm::normalize(halfVector);
    const float cosThetaH = fmaxf(0.0f, glm::dot(shadingNormal, normalizedHalfVector));
    const float cosWoH = fmaxf(0.0f, glm::dot(wo, normalizedHalfVector));
    if (cosThetaH <= 0.0f || cosWoH <= 0.0f)
    {
        return 0.0f;
    }

    const float alpha = roughnessToAlpha(materialClearcoatRoughness(material));
    const float D = ggxDistribution(cosThetaH, alpha);
    return (D * cosThetaH) / fmaxf(4.0f * cosWoH, EPSILON);
}

__host__ __device__ inline glm::vec3 sampleGgxHalfVector(
    const glm::vec3& normal,
    float alpha,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const float u1 = u01(rng);
    const float u2 = u01(rng);
    const float phi = TWO_PI * u1;
    const float tanTheta2 = (alpha * alpha) * u2 / fmaxf(1.0f - u2, EPSILON);
    const float cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    glm::vec3 tangent;
    if (fabsf(normal.x) < SQRT_OF_ONE_THIRD)
    {
        tangent = glm::normalize(glm::cross(normal, glm::vec3(1, 0, 0)));
    }
    else if (fabsf(normal.y) < SQRT_OF_ONE_THIRD)
    {
        tangent = glm::normalize(glm::cross(normal, glm::vec3(0, 1, 0)));
    }
    else
    {
        tangent = glm::normalize(glm::cross(normal, glm::vec3(0, 0, 1)));
    }
    const glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));

    return glm::normalize(
        tangent * (cosf(phi) * sinTheta) +
        bitangent * (sinf(phi) * sinTheta) +
        normal * cosTheta);
}

__host__ __device__ void computeLobeProbabilities(
    const Material& m,
    float& pDiffuse,
    float& pReflect,
    float& pClearcoat,
    float& pRefract)
{
    const float reflectiveStrength = saturate(m.hasReflective);
    const float clearcoatStrength = saturate(m.clearcoatFactor);
    float refractiveWeight = materialTransmissionStrength(m);
    const float dielectricSpecular = glm::clamp(max(max(m.specularColor.r, m.specularColor.g), m.specularColor.b), 0.0f, 1.0f);
    float reflectiveWeight = reflectiveStrength * fmaxf(dielectricSpecular, 0.04f);
    float diffuseWeight = (1.0f - refractiveWeight) * (1.0f - saturate(m.metallic));
    float clearcoatWeight = clearcoatStrength * 0.25f;
    if (reflectiveStrength <= 0.0f)
    {
        reflectiveWeight = 0.0f;
    }
    if (clearcoatStrength <= 0.0f)
    {
        clearcoatWeight = 0.0f;
    }

    float sumWeights = diffuseWeight + reflectiveWeight + clearcoatWeight + refractiveWeight;
    if (sumWeights < EPSILON)
    {
        diffuseWeight = 1.0f;
        sumWeights = 1.0f;
    }

    pDiffuse = diffuseWeight / sumWeights;
    pReflect = reflectiveWeight / sumWeights;
    pClearcoat = clearcoatWeight / sumWeights;
    pRefract = refractiveWeight / sumWeights;
}

__host__ __device__ bool materialHasNonDeltaBsdf(const Material& material)
{
    float pDiffuse, pReflect, pClearcoat, pRefract;
    computeLobeProbabilities(material, pDiffuse, pReflect, pClearcoat, pRefract);
    return pDiffuse > 0.0f
        || (pReflect > 0.0f && materialRoughness(material) > 0.001f)
        || (pClearcoat > 0.0f && materialClearcoatRoughness(material) > 0.001f);
}

__host__ __device__ glm::vec3 evaluateBsdf(
    const Material& material,
    const glm::vec3& wo,
    const glm::vec3& shadingNormal,
    const glm::vec3& wi)
{
    float pDiffuse, pReflect, pClearcoat, pRefract;
    computeLobeProbabilities(material, pDiffuse, pReflect, pClearcoat, pRefract);

    glm::vec3 f(0.0f);
    const float cosThetaI = fmaxf(0.0f, glm::dot(shadingNormal, wi));
    const float cosThetaO = fmaxf(0.0f, glm::dot(shadingNormal, wo));
    const float baseLayerTransmission = evaluateClearcoatTransmission(material, wo, shadingNormal, wi);

    if (pDiffuse > 0.0f && cosThetaI > 0.0f && cosThetaO > 0.0f)
    {
        f += baseLayerTransmission * ((material.color * (1.0f - material.metallic)) / PI);
    }

    if (pReflect > 0.0f && materialRoughness(material) > 0.001f && cosThetaI > 0.0f && cosThetaO > 0.0f)
    {
        const glm::vec3 halfVector = glm::normalize(wo + wi);
        const float cosThetaH = fmaxf(0.0f, glm::dot(shadingNormal, halfVector));
        const float cosWoH = fmaxf(0.0f, glm::dot(wo, halfVector));
        if (cosThetaH > 0.0f && cosWoH > 0.0f)
        {
            const float alpha = roughnessToAlpha(materialRoughness(material));
            const float D = ggxDistribution(cosThetaH, alpha);
            const float G = smithG2(cosThetaO, cosThetaI, alpha);
            const glm::vec3 F = fresnelSchlick(material.specularColor, cosWoH);
            f += baseLayerTransmission * ((D * G) * F / fmaxf(4.0f * cosThetaO * cosThetaI, EPSILON));
        }
    }

    if (pClearcoat > 0.0f)
    {
        f += evaluateClearcoatBrdf(material, wo, shadingNormal, wi);
    }

    return f;
}

__host__ __device__ float evaluateBsdfPdf(
    const Material& material,
    const glm::vec3& wo,
    const glm::vec3& shadingNormal,
    const glm::vec3& wi)
{
    float pDiffuse, pReflect, pClearcoat, pRefract;
    computeLobeProbabilities(material, pDiffuse, pReflect, pClearcoat, pRefract);
    float pdf = 0.0f;
    const float cosTheta = fmaxf(0.0f, glm::dot(shadingNormal, wi));
    if (pDiffuse > 0.0f)
    {
        pdf += pDiffuse * (cosTheta / PI);
    }

    if (pReflect > 0.0f && materialRoughness(material) > 0.001f)
    {
        const glm::vec3 halfVector = glm::normalize(wo + wi);
        const float cosThetaH = fmaxf(0.0f, glm::dot(shadingNormal, halfVector));
        const float cosWoH = fmaxf(0.0f, glm::dot(wo, halfVector));
        if (cosThetaH > 0.0f && cosWoH > 0.0f)
        {
            const float alpha = roughnessToAlpha(materialRoughness(material));
            const float D = ggxDistribution(cosThetaH, alpha);
            const float specPdf = (D * cosThetaH) / fmaxf(4.0f * cosWoH, EPSILON);
            pdf += pReflect * specPdf;
        }
    }

    if (pClearcoat > 0.0f)
    {
        pdf += pClearcoat * evaluateClearcoatPdf(material, wo, shadingNormal, wi);
    }

    return pdf;
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

    float pDiffuse, pReflect, pClearcoat, pRefract;
    computeLobeProbabilities(m, pDiffuse, pReflect, pClearcoat, pRefract);

    float xi = u01(rng);

    if (xi < pReflect)
    {
        const float roughness = materialRoughness(m);
        if (roughness <= 0.001f)
        {
            sample.direction = glm::normalize(glm::reflect(wi, shadingNormal));
            sample.pathWeight = safeDivide(m.specularColor, pReflect);
            sample.pdf = 0.0f;
            sample.isDelta = 1;
            return;
        }

        const glm::vec3 wo = -wi;
        glm::vec3 halfVector = sampleGgxHalfVector(shadingNormal, roughnessToAlpha(roughness), rng);
        if (glm::dot(wo, halfVector) <= 0.0f)
        {
            halfVector = -halfVector;
        }

        sample.direction = glm::reflect(-wo, halfVector);
        if (glm::dot(sample.direction, shadingNormal) <= 0.0f)
        {
            sample.direction = glm::normalize(glm::reflect(wi, shadingNormal));
            sample.pathWeight = safeDivide(m.specularColor, pReflect);
            sample.pdf = 0.0f;
            sample.isDelta = 1;
            return;
        }

        sample.direction = glm::normalize(sample.direction);
        const glm::vec3 f = evaluateBsdf(m, wo, shadingNormal, sample.direction);
        const float cosTheta = fmaxf(0.0f, glm::dot(shadingNormal, sample.direction));
        sample.pdf = evaluateBsdfPdf(m, wo, shadingNormal, sample.direction);
        sample.pathWeight = f * (cosTheta / fmaxf(sample.pdf, EPSILON));
        sample.isDelta = 0;
        return;
    }

    if (xi < (pReflect + pClearcoat))
    {
        const float clearcoatRoughness = materialClearcoatRoughness(m);
        if (clearcoatRoughness <= 0.001f)
        {
            sample.direction = glm::normalize(glm::reflect(wi, shadingNormal));
            const float cosTheta = fmaxf(0.0f, glm::dot(-wi, shadingNormal));
            const glm::vec3 clearcoatReflectance = glm::vec3(
                glm::clamp(m.clearcoatFactor, 0.0f, 1.0f) * fresnelSchlickScalar(0.04f, cosTheta));
            sample.pathWeight = safeDivide(clearcoatReflectance, pClearcoat);
            sample.pdf = 0.0f;
            sample.isDelta = 1;
            return;
        }

        const glm::vec3 wo = -wi;
        glm::vec3 halfVector = sampleGgxHalfVector(shadingNormal, roughnessToAlpha(clearcoatRoughness), rng);
        if (glm::dot(wo, halfVector) <= 0.0f)
        {
            halfVector = -halfVector;
        }

        sample.direction = glm::reflect(-wo, halfVector);
        if (glm::dot(sample.direction, shadingNormal) <= 0.0f)
        {
            sample.direction = glm::normalize(glm::reflect(wi, shadingNormal));
            const float cosTheta = fmaxf(0.0f, glm::dot(-wi, shadingNormal));
            const glm::vec3 clearcoatReflectance = glm::vec3(
                glm::clamp(m.clearcoatFactor, 0.0f, 1.0f) * fresnelSchlickScalar(0.04f, cosTheta));
            sample.pathWeight = safeDivide(clearcoatReflectance, pClearcoat);
            sample.pdf = 0.0f;
            sample.isDelta = 1;
            return;
        }

        sample.direction = glm::normalize(sample.direction);
        const glm::vec3 f = evaluateBsdf(m, wo, shadingNormal, sample.direction);
        const float cosTheta = fmaxf(0.0f, glm::dot(shadingNormal, sample.direction));
        sample.pdf = evaluateBsdfPdf(m, wo, shadingNormal, sample.direction);
        sample.pathWeight = f * (cosTheta / fmaxf(sample.pdf, EPSILON));
        sample.isDelta = 0;
        return;
    }

    if (xi < (pReflect + pClearcoat + pRefract))
    {
        if (m.thinWalled && materialTransmissionStrength(m) > 0.0f)
        {
            sample.direction = wi;
            sample.pathWeight = safeDivide(glm::vec3(materialTransmissionStrength(m)), pRefract);
            sample.pdf = 0.0f;
            sample.isDelta = 1;
            return;
        }

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
        sample.pathWeight = safeDivide(glm::vec3(materialTransmissionStrength(m)), pRefract);
        sample.pdf = 0.0f;
        sample.isDelta = 1;
        return;
    }

    sample.direction = calculateRandomDirectionInHemisphere(shadingNormal, rng);
    sample.pathWeight = safeDivide(m.color * (1.0f - saturate(m.metallic)), pDiffuse);
    sample.pdf = evaluateBsdfPdf(m, -wi, shadingNormal, sample.direction);
    sample.isDelta = 0;
}
