#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        if (glm::abs(qdxyz) < EPSILON)
        {
            if (q.origin[xyz] < -0.5f || q.origin[xyz] > 0.5f)
            {
                return -1.0f;
            }
            continue;
        }

        float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
        float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
        float ta = glm::min(t1, t2);
        float tb = glm::max(t1, t2);

        glm::vec3 n(0.0f);
        n[xyz] = (t2 < t1) ? +1.0f : -1.0f;

        if (ta > MIN_INTERSECTION_T && ta > tmin)
        {
            tmin = ta;
            tmin_n = n;
        }
        if (tb < tmax)
        {
            tmax = tb;
            tmax_n = n;
        }
    }

    if (tmax >= tmin && tmax > MIN_INTERSECTION_T)
    {
        outside = true;
        if (tmin <= MIN_INTERSECTION_T)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1.0f;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    const float radius = 0.5f;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - radius * radius);
    if (radicand < 0.0f)
    {
        return -1.0f;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0.0f;
    if (t1 < MIN_INTERSECTION_T && t2 < MIN_INTERSECTION_T)
    {
        return -1.0f;
    }
    else if (t1 > MIN_INTERSECTION_T && t2 > MIN_INTERSECTION_T)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.0f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.0f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& shadingNormal,
    glm::vec3& geometricNormal,
    glm::vec2 uv[MAX_TEXTURE_UV_SETS],
    int& uvSetMask,
    glm::vec3 tangents[MAX_TEXTURE_UV_SETS],
    float tangentSigns[MAX_TEXTURE_UV_SETS],
    int& tangentSetMask)
{
    const glm::vec3 edge1 = triangle.p1 - triangle.p0;
    const glm::vec3 edge2 = triangle.p2 - triangle.p0;
    const glm::vec3 pvec = glm::cross(r.direction, edge2);
    const float det = glm::dot(edge1, pvec);

    if (fabsf(det) < RENDER_CONFIG_TRIANGLE_DET_EPSILON)
    {
        return -1.0f;
    }

    const float invDet = 1.0f / det;
    const glm::vec3 tvec = r.origin - triangle.p0;
    const float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return -1.0f;
    }

    const glm::vec3 qvec = glm::cross(tvec, edge1);
    const float v = glm::dot(r.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f)
    {
        return -1.0f;
    }

    const float t = glm::dot(edge2, qvec) * invDet;
    if (t <= RENDER_CONFIG_TRIANGLE_MIN_INTERSECTION_T)
    {
        return -1.0f;
    }

    const float w = 1.0f - u - v;
    intersectionPoint = r.origin + t * r.direction;
    geometricNormal = glm::normalize(triangle.geometricNormal);
    shadingNormal = triangle.hasVertexNormals
        ? glm::normalize(w * triangle.n0 + u * triangle.n1 + v * triangle.n2)
        : geometricNormal;
    if (glm::dot(shadingNormal, geometricNormal) < 0.0f)
    {
        shadingNormal = -shadingNormal;
    }

    uvSetMask = triangle.uvSetMask;
    for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
    {
        uv[uvSet] = (triangle.uvSetMask & (1 << uvSet))
            ? (w * triangle.uv0[uvSet] + u * triangle.uv1[uvSet] + v * triangle.uv2[uvSet])
            : glm::vec2(0.0f);
    }
    tangentSetMask = 0;
    for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
    {
        tangents[uvSet] = glm::vec3(0.0f);
        tangentSigns[uvSet] = 1.0f;
    }

    if (triangle.hasVertexTangents)
    {
        const glm::vec4 interpolatedTangent = w * triangle.t0 + u * triangle.t1 + v * triangle.t2;
        const glm::vec3 tangent = glm::vec3(interpolatedTangent);
        if (glm::dot(tangent, tangent) > EPSILON)
        {
            tangents[0] = glm::normalize(tangent);
            tangentSigns[0] = interpolatedTangent.w < 0.0f ? -1.0f : 1.0f;
            tangentSetMask |= 1;
        }
    }

    const glm::vec3 dp1 = triangle.p1 - triangle.p0;
    const glm::vec3 dp2 = triangle.p2 - triangle.p0;
    for (int uvSet = 0; uvSet < MAX_TEXTURE_UV_SETS; ++uvSet)
    {
        if ((triangle.uvSetMask & (1 << uvSet)) == 0)
        {
            continue;
        }
        if (uvSet == 0 && (tangentSetMask & 1))
        {
            continue;
        }

        const glm::vec2 duv1 = triangle.uv1[uvSet] - triangle.uv0[uvSet];
        const glm::vec2 duv2 = triangle.uv2[uvSet] - triangle.uv0[uvSet];
        const float detUv = duv1.x * duv2.y - duv1.y * duv2.x;
        if (fabsf(detUv) > RENDER_CONFIG_TRIANGLE_UV_DET_EPSILON)
        {
            const float invDetUv = 1.0f / detUv;
            const glm::vec3 derivedTangent = (dp1 * duv2.y - dp2 * duv1.y) * invDetUv;
            const glm::vec3 derivedBitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDetUv;
            if (glm::dot(derivedTangent, derivedTangent) > EPSILON)
            {
                tangents[uvSet] = glm::normalize(derivedTangent);
                tangentSigns[uvSet] = glm::dot(glm::cross(geometricNormal, tangents[uvSet]), derivedBitangent) < 0.0f ? -1.0f : 1.0f;
                tangentSetMask |= (1 << uvSet);
            }
        }
    }

    return t;
}
