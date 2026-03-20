#include "mesh.h"

#include <stb_image.h>

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

#include "glm/gtc/matrix_inverse.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr int kWrapRepeat = 10497;
constexpr int kWrapClampToEdge = 33071;

std::string trim(const std::string& value)
{
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return std::string();
    }
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

bool readPpmToken(std::istream& input, std::string& token)
{
    token.clear();
    char ch = 0;
    while (input.get(ch))
    {
        if (ch == '#')
        {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch)))
        {
            token.push_back(ch);
            break;
        }
    }

    if (token.empty())
    {
        return false;
    }

    while (input.get(ch))
    {
        if (ch == '#')
        {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            break;
        }
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            break;
        }
        token.push_back(ch);
    }

    return true;
}

float srgbToLinearChannel(float c)
{
    if (c <= 0.04045f)
    {
        return c / 12.92f;
    }
    return powf((c + 0.055f) / 1.055f, 2.4f);
}

glm::vec3 srgbToLinear(const glm::vec3& srgb)
{
    return glm::vec3(
        srgbToLinearChannel(srgb.r),
        srgbToLinearChannel(srgb.g),
        srgbToLinearChannel(srgb.b));
}

bool loadAsciiPpmImage(
    const std::filesystem::path& texturePath,
    bool decodeSrgb,
    TextureData& outTexture,
    std::vector<glm::vec4>& outPixels,
    std::string& outError)
{
    std::ifstream input(texturePath);
    if (!input)
    {
        return false;
    }

    std::string token;
    if (!readPpmToken(input, token) || token != "P3")
    {
        return false;
    }

    std::string widthToken;
    std::string heightToken;
    std::string maxValueToken;
    if (!readPpmToken(input, widthToken)
        || !readPpmToken(input, heightToken)
        || !readPpmToken(input, maxValueToken))
    {
        outError = "Failed to parse PPM header: " + texturePath.string();
        return false;
    }

    const int width = std::stoi(widthToken);
    const int height = std::stoi(heightToken);
    const int maxValue = std::stoi(maxValueToken);
    if (width <= 0 || height <= 0 || maxValue <= 0)
    {
        outError = "Invalid PPM dimensions: " + texturePath.string();
        return false;
    }

    outTexture.width = width;
    outTexture.height = height;
    outTexture.pixelOffset = static_cast<int>(outPixels.size());
    outTexture.wrapS = kWrapRepeat;
    outTexture.wrapT = kWrapRepeat;
    outPixels.reserve(outPixels.size() + width * height);

    const float invMax = 1.0f / static_cast<float>(maxValue);
    for (int i = 0; i < width * height; ++i)
    {
        std::string rToken;
        std::string gToken;
        std::string bToken;
        if (!readPpmToken(input, rToken)
            || !readPpmToken(input, gToken)
            || !readPpmToken(input, bToken))
        {
            outError = "PPM pixel data truncated: " + texturePath.string();
            return false;
        }

        glm::vec3 rgb(
            std::stoi(rToken) * invMax,
            std::stoi(gToken) * invMax,
            std::stoi(bToken) * invMax);
        outPixels.emplace_back(decodeSrgb ? srgbToLinear(rgb) : rgb, 1.0f);
    }

    return true;
}

glm::mat4 nodeTransform(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16)
    {
        glm::mat4 matrix(1.0f);
        for (int i = 0; i < 16; ++i)
        {
            // glTF stores matrices in column-major order, and GLM is indexed as mat[col][row].
            matrix[i / 4][i % 4] = static_cast<float>(node.matrix[i]);
        }
        return matrix;
    }

    glm::vec3 translation(0.0f);
    if (node.translation.size() == 3)
    {
        translation = glm::vec3(
            static_cast<float>(node.translation[0]),
            static_cast<float>(node.translation[1]),
            static_cast<float>(node.translation[2]));
    }

    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    if (node.rotation.size() == 4)
    {
        rotation = glm::quat(
            static_cast<float>(node.rotation[3]),
            static_cast<float>(node.rotation[0]),
            static_cast<float>(node.rotation[1]),
            static_cast<float>(node.rotation[2]));
    }

    glm::vec3 scale(1.0f);
    if (node.scale.size() == 3)
    {
        scale = glm::vec3(
            static_cast<float>(node.scale[0]),
            static_cast<float>(node.scale[1]),
            static_cast<float>(node.scale[2]));
    }

    return glm::translate(translation) * glm::mat4_cast(rotation) * glm::scale(scale);
}

int accessorComponentCount(int type)
{
    switch (type)
    {
    case TINYGLTF_TYPE_SCALAR: return 1;
    case TINYGLTF_TYPE_VEC2: return 2;
    case TINYGLTF_TYPE_VEC3: return 3;
    case TINYGLTF_TYPE_VEC4: return 4;
    default: return 0;
    }
}

float normalizedComponentToFloat(const unsigned char* data, int componentType)
{
    switch (componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return static_cast<float>(*reinterpret_cast<const uint8_t*>(data)) / 255.0f;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return static_cast<float>(*reinterpret_cast<const uint16_t*>(data)) / 65535.0f;
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    {
        const int8_t value = *reinterpret_cast<const int8_t*>(data);
        return glm::max(-1.0f, static_cast<float>(value) / 127.0f);
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    {
        const int16_t value = *reinterpret_cast<const int16_t*>(data);
        return glm::max(-1.0f, static_cast<float>(value) / 32767.0f);
    }
    default:
        return 0.0f;
    }
}

float componentToFloat(const unsigned char* data, int componentType, bool normalized)
{
    if (normalized)
    {
        return normalizedComponentToFloat(data, componentType);
    }

    switch (componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return *reinterpret_cast<const float*>(data);
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        return static_cast<float>(*reinterpret_cast<const double*>(data));
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return static_cast<float>(*reinterpret_cast<const uint8_t*>(data));
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return static_cast<float>(*reinterpret_cast<const uint16_t*>(data));
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        return static_cast<float>(*reinterpret_cast<const uint32_t*>(data));
    case TINYGLTF_COMPONENT_TYPE_BYTE:
        return static_cast<float>(*reinterpret_cast<const int8_t*>(data));
    case TINYGLTF_COMPONENT_TYPE_SHORT:
        return static_cast<float>(*reinterpret_cast<const int16_t*>(data));
    default:
        return 0.0f;
    }
}

bool readAccessorVec3(
    const tinygltf::Model& model,
    int accessorIndex,
    std::vector<glm::vec3>& outValues,
    std::string& outError)
{
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size()))
    {
        outError = "Invalid glTF accessor index";
        return false;
    }

    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    if (accessor.bufferView < 0 || accessor.bufferView >= static_cast<int>(model.bufferViews.size()))
    {
        outError = "Invalid glTF accessor buffer view";
        return false;
    }
    if (accessor.type != TINYGLTF_TYPE_VEC3)
    {
        outError = "glTF accessor is not VEC3";
        return false;
    }

    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    if (bufferView.buffer < 0 || bufferView.buffer >= static_cast<int>(model.buffers.size()))
    {
        outError = "Invalid glTF buffer";
        return false;
    }

    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
    const size_t componentSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);
    const size_t stride = accessor.ByteStride(bufferView);
    const size_t offset = bufferView.byteOffset + accessor.byteOffset;
    const size_t elementSize = componentSize * 3;
    if (componentSize == 0 || stride < elementSize || offset + stride * accessor.count > buffer.data.size() + (stride - elementSize))
    {
        outError = "glTF VEC3 accessor out of range";
        return false;
    }

    outValues.resize(accessor.count);
    const unsigned char* base = buffer.data.data() + offset;
    for (size_t i = 0; i < accessor.count; ++i)
    {
        const unsigned char* element = base + i * stride;
        outValues[i] = glm::vec3(
            componentToFloat(element + componentSize * 0, accessor.componentType, accessor.normalized),
            componentToFloat(element + componentSize * 1, accessor.componentType, accessor.normalized),
            componentToFloat(element + componentSize * 2, accessor.componentType, accessor.normalized));
    }

    return true;
}

bool readAccessorVec2(
    const tinygltf::Model& model,
    int accessorIndex,
    std::vector<glm::vec2>& outValues,
    std::string& outError)
{
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size()))
    {
        outError = "Invalid glTF accessor index";
        return false;
    }

    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    if (accessor.bufferView < 0 || accessor.bufferView >= static_cast<int>(model.bufferViews.size()))
    {
        outError = "Invalid glTF accessor buffer view";
        return false;
    }
    if (accessor.type != TINYGLTF_TYPE_VEC2)
    {
        outError = "glTF accessor is not VEC2";
        return false;
    }

    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    if (bufferView.buffer < 0 || bufferView.buffer >= static_cast<int>(model.buffers.size()))
    {
        outError = "Invalid glTF buffer";
        return false;
    }

    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
    const size_t componentSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);
    const size_t stride = accessor.ByteStride(bufferView);
    const size_t offset = bufferView.byteOffset + accessor.byteOffset;
    const size_t elementSize = componentSize * 2;
    if (componentSize == 0 || stride < elementSize || offset + stride * accessor.count > buffer.data.size() + (stride - elementSize))
    {
        outError = "glTF VEC2 accessor out of range";
        return false;
    }

    outValues.resize(accessor.count);
    const unsigned char* base = buffer.data.data() + offset;
    for (size_t i = 0; i < accessor.count; ++i)
    {
        const unsigned char* element = base + i * stride;
        outValues[i] = glm::vec2(
            componentToFloat(element + componentSize * 0, accessor.componentType, accessor.normalized),
            componentToFloat(element + componentSize * 1, accessor.componentType, accessor.normalized));
    }

    return true;
}

bool readAccessorVec4(
    const tinygltf::Model& model,
    int accessorIndex,
    std::vector<glm::vec4>& outValues,
    std::string& outError)
{
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size()))
    {
        outError = "Invalid glTF accessor index";
        return false;
    }

    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    if (accessor.bufferView < 0 || accessor.bufferView >= static_cast<int>(model.bufferViews.size()))
    {
        outError = "Invalid glTF accessor buffer view";
        return false;
    }
    if (accessor.type != TINYGLTF_TYPE_VEC4)
    {
        outError = "glTF accessor is not VEC4";
        return false;
    }

    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    if (bufferView.buffer < 0 || bufferView.buffer >= static_cast<int>(model.buffers.size()))
    {
        outError = "Invalid glTF buffer";
        return false;
    }

    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
    const size_t componentSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);
    const size_t stride = accessor.ByteStride(bufferView);
    const size_t offset = bufferView.byteOffset + accessor.byteOffset;
    const size_t elementSize = componentSize * 4;
    if (componentSize == 0 || stride < elementSize || offset + stride * accessor.count > buffer.data.size() + (stride - elementSize))
    {
        outError = "glTF VEC4 accessor out of range";
        return false;
    }

    outValues.resize(accessor.count);
    const unsigned char* base = buffer.data.data() + offset;
    for (size_t i = 0; i < accessor.count; ++i)
    {
        const unsigned char* element = base + i * stride;
        outValues[i] = glm::vec4(
            componentToFloat(element + componentSize * 0, accessor.componentType, accessor.normalized),
            componentToFloat(element + componentSize * 1, accessor.componentType, accessor.normalized),
            componentToFloat(element + componentSize * 2, accessor.componentType, accessor.normalized),
            componentToFloat(element + componentSize * 3, accessor.componentType, accessor.normalized));
    }

    return true;
}

bool readIndices(
    const tinygltf::Model& model,
    int accessorIndex,
    std::vector<uint32_t>& outIndices,
    std::string& outError)
{
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size()))
    {
        outError = "Invalid glTF index accessor";
        return false;
    }

    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    if (accessor.type != TINYGLTF_TYPE_SCALAR
        || accessor.bufferView < 0
        || accessor.bufferView >= static_cast<int>(model.bufferViews.size()))
    {
        outError = "glTF indices must be SCALAR";
        return false;
    }

    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    if (bufferView.buffer < 0 || bufferView.buffer >= static_cast<int>(model.buffers.size()))
    {
        outError = "Invalid glTF index buffer";
        return false;
    }

    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
    const size_t componentSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);
    const size_t stride = accessor.ByteStride(bufferView);
    const size_t offset = bufferView.byteOffset + accessor.byteOffset;
    if (componentSize == 0 || stride < componentSize || offset + stride * accessor.count > buffer.data.size() + (stride - componentSize))
    {
        outError = "glTF index accessor out of range";
        return false;
    }

    outIndices.resize(accessor.count);
    const unsigned char* base = buffer.data.data() + offset;
    for (size_t i = 0; i < accessor.count; ++i)
    {
        const unsigned char* element = base + i * stride;
        switch (accessor.componentType)
        {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            outIndices[i] = *reinterpret_cast<const uint8_t*>(element);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            outIndices[i] = *reinterpret_cast<const uint16_t*>(element);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            outIndices[i] = *reinterpret_cast<const uint32_t*>(element);
            break;
        default:
            outError = "Unsupported glTF index component type";
            return false;
        }
    }

    return true;
}

void appendTriangle(
    const glm::mat4& transform,
    const glm::mat4& invTranspose,
    const glm::vec3& p0,
    const glm::vec3& p1,
    const glm::vec3& p2,
    const glm::vec3* n0,
    const glm::vec3* n1,
    const glm::vec3* n2,
    const glm::vec2* uv0,
    const glm::vec2* uv1,
    const glm::vec2* uv2,
    const glm::vec4* t0,
    const glm::vec4* t1,
    const glm::vec4* t2,
    int materialId,
    std::vector<Triangle>& outTriangles)
{
    Triangle tri{};
    tri.materialId = materialId;
    tri.p0 = glm::vec3(transform * glm::vec4(p0, 1.0f));
    tri.p1 = glm::vec3(transform * glm::vec4(p1, 1.0f));
    tri.p2 = glm::vec3(transform * glm::vec4(p2, 1.0f));

    const glm::vec3 faceNormal = glm::normalize(glm::cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
    tri.geometricNormal = faceNormal;
    tri.n0 = n0 ? glm::normalize(glm::vec3(invTranspose * glm::vec4(*n0, 0.0f))) : faceNormal;
    tri.n1 = n1 ? glm::normalize(glm::vec3(invTranspose * glm::vec4(*n1, 0.0f))) : faceNormal;
    tri.n2 = n2 ? glm::normalize(glm::vec3(invTranspose * glm::vec4(*n2, 0.0f))) : faceNormal;
    tri.hasVertexNormals = (n0 && n1 && n2) ? 1 : 0;

    tri.uv0 = uv0 ? *uv0 : glm::vec2(0.0f);
    tri.uv1 = uv1 ? *uv1 : glm::vec2(0.0f);
    tri.uv2 = uv2 ? *uv2 : glm::vec2(0.0f);
    tri.hasUVs = (uv0 && uv1 && uv2) ? 1 : 0;
    tri.t0 = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    tri.t1 = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    tri.t2 = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    tri.hasVertexTangents = 0;
    if (t0 && t1 && t2)
    {
        const float handednessFlip = glm::determinant(glm::mat3(transform)) < 0.0f ? -1.0f : 1.0f;
        auto transformTangent = [&](const glm::vec4& tangent) -> glm::vec4
        {
            const glm::vec3 worldTangent = glm::vec3(transform * glm::vec4(glm::vec3(tangent), 0.0f));
            const float tangentLength2 = glm::dot(worldTangent, worldTangent);
            if (tangentLength2 <= 1e-12f)
            {
                return glm::vec4(0.0f, 0.0f, 0.0f, tangent.w * handednessFlip);
            }
            return glm::vec4(glm::normalize(worldTangent), tangent.w * handednessFlip);
        };

        tri.t0 = transformTangent(*t0);
        tri.t1 = transformTangent(*t1);
        tri.t2 = transformTangent(*t2);
        tri.hasVertexTangents = 1;
    }

    outTriangles.push_back(tri);
}

int ensureMaterialDefinition(
    const tinygltf::Model& model,
    int gltfMaterialIndex,
    std::vector<MeshMaterialDefinition>& outMaterials)
{
    if (gltfMaterialIndex < 0 || gltfMaterialIndex >= static_cast<int>(model.materials.size()))
    {
        if (outMaterials.empty())
        {
            outMaterials.push_back(MeshMaterialDefinition{});
            outMaterials.back().name = "default";
        }
        return 0;
    }

    while (static_cast<int>(outMaterials.size()) <= gltfMaterialIndex)
    {
        outMaterials.push_back(MeshMaterialDefinition{});
    }
    if (!outMaterials[gltfMaterialIndex].name.empty())
    {
        return gltfMaterialIndex;
    }

    const tinygltf::Material& src = model.materials[gltfMaterialIndex];
    MeshMaterialDefinition material{};
    material.name = src.name.empty() ? ("material_" + std::to_string(gltfMaterialIndex)) : src.name;
    material.alphaMode = 0;
    if (src.alphaMode == "MASK")
    {
        material.alphaMode = 1;
    }
    else if (src.alphaMode == "BLEND")
    {
        material.alphaMode = 2;
    }
    material.alphaCutoff = static_cast<float>(src.alphaCutoff);
    material.doubleSided = src.doubleSided ? 1 : 0;

    int diffuseTextureIndex = -1;
    int diffuseTexcoordSet = 0;
    int normalTextureIndex = -1;

    if (src.extensions.find("KHR_materials_pbrSpecularGlossiness") != src.extensions.end())
    {
        const tinygltf::Value& ext = src.extensions.at("KHR_materials_pbrSpecularGlossiness");
        if (ext.IsObject())
        {
            if (ext.Has("diffuseFactor"))
            {
                const tinygltf::Value& diffuseFactor = ext.Get("diffuseFactor");
                if (diffuseFactor.IsArray() && diffuseFactor.ArrayLen() >= 3)
                {
                    material.diffuseColor = glm::vec3(
                        static_cast<float>(diffuseFactor.Get(0).GetNumberAsDouble()),
                        static_cast<float>(diffuseFactor.Get(1).GetNumberAsDouble()),
                        static_cast<float>(diffuseFactor.Get(2).GetNumberAsDouble()));
                }
            }
            if (ext.Has("specularFactor"))
            {
                const tinygltf::Value& specularFactor = ext.Get("specularFactor");
                if (specularFactor.IsArray() && specularFactor.ArrayLen() >= 3)
                {
                    material.specularColor = glm::vec3(
                        static_cast<float>(specularFactor.Get(0).GetNumberAsDouble()),
                        static_cast<float>(specularFactor.Get(1).GetNumberAsDouble()),
                        static_cast<float>(specularFactor.Get(2).GetNumberAsDouble()));
                }
            }
            if (ext.Has("glossinessFactor"))
            {
                const float glossiness = static_cast<float>(ext.Get("glossinessFactor").GetNumberAsDouble());
                material.roughnessFactor = glm::clamp(1.0f - glossiness, 0.0f, 1.0f);
            }
            material.metallicFactor = 0.0f;

            if (ext.Has("diffuseTexture"))
            {
                const tinygltf::Value& diffuseTexture = ext.Get("diffuseTexture");
                if (diffuseTexture.IsObject() && diffuseTexture.Has("index"))
                {
                    diffuseTextureIndex = diffuseTexture.Get("index").GetNumberAsInt();
                    diffuseTexcoordSet = diffuseTexture.Has("texCoord")
                        ? diffuseTexture.Get("texCoord").GetNumberAsInt()
                        : 0;
                }
            }
        }
    }
    else
    {
        const auto& baseColorFactor = src.pbrMetallicRoughness.baseColorFactor;
        if (baseColorFactor.size() >= 3)
        {
            material.diffuseColor = glm::vec3(
                static_cast<float>(baseColorFactor[0]),
                static_cast<float>(baseColorFactor[1]),
                static_cast<float>(baseColorFactor[2]));
        }
        material.metallicFactor = static_cast<float>(src.pbrMetallicRoughness.metallicFactor);
        material.roughnessFactor = static_cast<float>(src.pbrMetallicRoughness.roughnessFactor);
        diffuseTextureIndex = src.pbrMetallicRoughness.baseColorTexture.index;
        diffuseTexcoordSet = src.pbrMetallicRoughness.baseColorTexture.texCoord;
    }

    material.diffuseTexcoordSet = diffuseTexcoordSet;
    material.flipV = false;
    material.wrapS = kWrapRepeat;
    material.wrapT = kWrapRepeat;

    if (diffuseTextureIndex >= 0 && diffuseTextureIndex < static_cast<int>(model.textures.size()))
    {
        const tinygltf::Texture& texture = model.textures[diffuseTextureIndex];
        if (texture.sampler >= 0 && texture.sampler < static_cast<int>(model.samplers.size()))
        {
            const tinygltf::Sampler& sampler = model.samplers[texture.sampler];
            material.wrapS = sampler.wrapS;
            material.wrapT = sampler.wrapT;
        }

        if (texture.source >= 0 && texture.source < static_cast<int>(model.images.size()))
        {
            const tinygltf::Image& image = model.images[texture.source];
            if (!image.image.empty())
            {
                material.diffuseTextureBytes = image.image;
            }
            if (!image.uri.empty())
            {
                material.diffuseTexturePath = image.uri;
            }
            material.diffuseTextureKey = "gltf_image_" + std::to_string(texture.source);
        }
    }

    const tinygltf::TextureInfo& metallicRoughnessInfo = src.pbrMetallicRoughness.metallicRoughnessTexture;
    material.metallicRoughnessTexcoordSet = metallicRoughnessInfo.texCoord;
    if (metallicRoughnessInfo.index >= 0 && metallicRoughnessInfo.index < static_cast<int>(model.textures.size()))
    {
        const tinygltf::Texture& texture = model.textures[metallicRoughnessInfo.index];
        if (texture.source >= 0 && texture.source < static_cast<int>(model.images.size()))
        {
            const tinygltf::Image& image = model.images[texture.source];
            if (!image.image.empty())
            {
                material.metallicRoughnessTextureBytes = image.image;
            }
            if (!image.uri.empty())
            {
                material.metallicRoughnessTexturePath = image.uri;
            }
            material.metallicRoughnessTextureKey = "gltf_image_" + std::to_string(texture.source);
        }
    }

    normalTextureIndex = src.normalTexture.index;
    material.normalTexcoordSet = src.normalTexture.texCoord;
    material.normalTextureScale = static_cast<float>(src.normalTexture.scale);
    if (normalTextureIndex >= 0 && normalTextureIndex < static_cast<int>(model.textures.size()))
    {
        const tinygltf::Texture& texture = model.textures[normalTextureIndex];
        if (texture.source >= 0 && texture.source < static_cast<int>(model.images.size()))
        {
            const tinygltf::Image& image = model.images[texture.source];
            if (!image.image.empty())
            {
                material.normalTextureBytes = image.image;
            }
            if (!image.uri.empty())
            {
                material.normalTexturePath = image.uri;
            }
            material.normalTextureKey = "gltf_image_" + std::to_string(texture.source);
        }
    }

    outMaterials[gltfMaterialIndex] = std::move(material);
    return gltfMaterialIndex;
}

bool appendNodeMesh(
    const tinygltf::Model& model,
    int nodeIndex,
    const glm::mat4& parentTransform,
    std::vector<MeshMaterialDefinition>& outMaterials,
    std::vector<Triangle>& outTriangles,
    std::string& outError)
{
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model.nodes.size()))
    {
        outError = "Invalid glTF node index";
        return false;
    }

    const tinygltf::Node& node = model.nodes[nodeIndex];
    const glm::mat4 localTransform = nodeTransform(node);
    const glm::mat4 worldTransform = parentTransform * localTransform;
    const glm::mat4 invTranspose = glm::inverseTranspose(worldTransform);

    if (node.mesh >= 0)
    {
        if (node.mesh >= static_cast<int>(model.meshes.size()))
        {
            outError = "Invalid glTF mesh index";
            return false;
        }

        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (const tinygltf::Primitive& primitive : mesh.primitives)
        {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
            {
                continue;
            }
            auto posIt = primitive.attributes.find("POSITION");
            if (posIt == primitive.attributes.end())
            {
                continue;
            }

            const int materialIndex = ensureMaterialDefinition(model, primitive.material, outMaterials);
            const int texcoordSet = (materialIndex >= 0 && materialIndex < static_cast<int>(outMaterials.size()))
                ? outMaterials[materialIndex].diffuseTexcoordSet
                : 0;

            std::vector<glm::vec3> positions;
            std::vector<glm::vec3> normals;
            std::vector<glm::vec2> texcoords;
            std::vector<glm::vec4> tangents;
            std::vector<uint32_t> indices;

            if (!readAccessorVec3(model, posIt->second, positions, outError))
            {
                return false;
            }

            auto normalIt = primitive.attributes.find("NORMAL");
            if (normalIt != primitive.attributes.end()
                && !readAccessorVec3(model, normalIt->second, normals, outError))
            {
                return false;
            }

            auto tangentIt = primitive.attributes.find("TANGENT");
            if (tangentIt != primitive.attributes.end()
                && !readAccessorVec4(model, tangentIt->second, tangents, outError))
            {
                return false;
            }

            const std::string texcoordSemantic = "TEXCOORD_" + std::to_string(texcoordSet);
            auto texcoordIt = primitive.attributes.find(texcoordSemantic);
            if (texcoordIt == primitive.attributes.end() && texcoordSet != 0)
            {
                texcoordIt = primitive.attributes.find("TEXCOORD_0");
            }
            if (texcoordIt != primitive.attributes.end()
                && !readAccessorVec2(model, texcoordIt->second, texcoords, outError))
            {
                return false;
            }

            if (primitive.indices >= 0)
            {
                if (!readIndices(model, primitive.indices, indices, outError))
                {
                    return false;
                }
            }
            else
            {
                indices.resize(positions.size());
                for (size_t i = 0; i < positions.size(); ++i)
                {
                    indices[i] = static_cast<uint32_t>(i);
                }
            }

            if (indices.size() % 3 != 0)
            {
                outError = "glTF primitive index count is not divisible by 3";
                return false;
            }

            for (size_t i = 0; i < indices.size(); i += 3)
            {
                const uint32_t i0 = indices[i + 0];
                const uint32_t i1 = indices[i + 1];
                const uint32_t i2 = indices[i + 2];
                if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size())
                {
                    outError = "glTF primitive references invalid vertex index";
                    return false;
                }

                const glm::vec3* n0 = (i0 < normals.size()) ? &normals[i0] : nullptr;
                const glm::vec3* n1 = (i1 < normals.size()) ? &normals[i1] : nullptr;
                const glm::vec3* n2 = (i2 < normals.size()) ? &normals[i2] : nullptr;
                const glm::vec2* uv0 = (i0 < texcoords.size()) ? &texcoords[i0] : nullptr;
                const glm::vec2* uv1 = (i1 < texcoords.size()) ? &texcoords[i1] : nullptr;
                const glm::vec2* uv2 = (i2 < texcoords.size()) ? &texcoords[i2] : nullptr;
                const glm::vec4* t0 = (i0 < tangents.size()) ? &tangents[i0] : nullptr;
                const glm::vec4* t1 = (i1 < tangents.size()) ? &tangents[i1] : nullptr;
                const glm::vec4* t2 = (i2 < tangents.size()) ? &tangents[i2] : nullptr;

                appendTriangle(
                    worldTransform,
                    invTranspose,
                    positions[i0],
                    positions[i1],
                    positions[i2],
                    n0,
                    n1,
                    n2,
                    uv0,
                    uv1,
                    uv2,
                    t0,
                    t1,
                    t2,
                    materialIndex,
                    outTriangles);
            }
        }
    }

    for (int childIndex : node.children)
    {
        if (!appendNodeMesh(model, childIndex, worldTransform, outMaterials, outTriangles, outError))
        {
            return false;
        }
    }

    return true;
}

bool loadImageBytesAsIs(
    tinygltf::Image* image,
    const int imageIndex,
    std::string* err,
    std::string* warn,
    int reqWidth,
    int reqHeight,
    const unsigned char* bytes,
    int size,
    void* userData)
{
    (void)imageIndex;
    (void)warn;
    (void)reqWidth;
    (void)reqHeight;
    (void)userData;

    if (image == nullptr || bytes == nullptr || size <= 0)
    {
        if (err)
        {
            *err = "tinygltf image callback received invalid image data";
        }
        return false;
    }

    image->width = -1;
    image->height = -1;
    image->component = -1;
    image->bits = -1;
    image->pixel_type = -1;
    image->as_is = true;
    image->image.assign(bytes, bytes + size);
    return true;
}

bool loadGltfMesh(
    const std::filesystem::path& meshPath,
    std::vector<Triangle>& outTriangles,
    std::vector<MeshMaterialDefinition>& outMaterials,
    std::string& outError)
{
    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(loadImageBytesAsIs, nullptr);

    tinygltf::Model model;
    std::string warning;
    std::string error;
    const std::string ext = meshPath.extension().string();
    bool loaded = false;
    if (ext == ".glb")
    {
        loaded = loader.LoadBinaryFromFile(&model, &error, &warning, meshPath.string());
    }
    else
    {
        loaded = loader.LoadASCIIFromFile(&model, &error, &warning, meshPath.string());
    }

    if (!warning.empty())
    {
        outError = warning;
    }
    if (!loaded)
    {
        outError = error.empty() ? ("Failed to load glTF: " + meshPath.string()) : error;
        return false;
    }

    const size_t startTriangleCount = outTriangles.size();
    if (model.defaultScene < 0 && model.scenes.empty() && model.nodes.empty())
    {
        outError = "glTF contains no scene data: " + meshPath.string();
        return false;
    }

    auto absolutizeImagePath = [&](MeshMaterialDefinition& material)
    {
        if (!material.diffuseTexturePath.empty())
        {
            material.diffuseTexturePath = meshPath.parent_path() / material.diffuseTexturePath;
        }
        if (!material.metallicRoughnessTexturePath.empty())
        {
            material.metallicRoughnessTexturePath = meshPath.parent_path() / material.metallicRoughnessTexturePath;
        }
        if (!material.normalTexturePath.empty())
        {
            material.normalTexturePath = meshPath.parent_path() / material.normalTexturePath;
        }
    };

    if (model.defaultScene >= 0 && model.defaultScene < static_cast<int>(model.scenes.size()))
    {
        for (int nodeIndex : model.scenes[model.defaultScene].nodes)
        {
            if (!appendNodeMesh(model, nodeIndex, glm::mat4(1.0f), outMaterials, outTriangles, outError))
            {
                return false;
            }
        }
    }
    else if (!model.scenes.empty())
    {
        for (int nodeIndex : model.scenes[0].nodes)
        {
            if (!appendNodeMesh(model, nodeIndex, glm::mat4(1.0f), outMaterials, outTriangles, outError))
            {
                return false;
            }
        }
    }
    else
    {
        for (size_t nodeIndex = 0; nodeIndex < model.nodes.size(); ++nodeIndex)
        {
            if (!appendNodeMesh(model, static_cast<int>(nodeIndex), glm::mat4(1.0f), outMaterials, outTriangles, outError))
            {
                return false;
            }
        }
    }

    for (MeshMaterialDefinition& material : outMaterials)
    {
        absolutizeImagePath(material);
    }

    if (outTriangles.size() == startTriangleCount)
    {
        outError = "glTF contains no supported triangle primitives: " + meshPath.string();
        return false;
    }

    return true;
}
} // namespace

bool loadTextureImage(
    const std::filesystem::path& texturePath,
    bool flipV,
    bool decodeSrgb,
    TextureData& outTexture,
    std::vector<glm::vec4>& outPixels,
    std::string& outError)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = stbi_load(texturePath.string().c_str(), &width, &height, &channels, 4);
    if (data)
    {
        outTexture.width = width;
        outTexture.height = height;
        outTexture.pixelOffset = static_cast<int>(outPixels.size());
        outTexture.flipV = flipV ? 1 : 0;
        outTexture.wrapS = kWrapRepeat;
        outTexture.wrapT = kWrapRepeat;
        outPixels.reserve(outPixels.size() + width * height);

        for (int i = 0; i < width * height; ++i)
        {
            const int base = i * 4;
            const glm::vec3 rgb(
                data[base + 0] / 255.0f,
                data[base + 1] / 255.0f,
                data[base + 2] / 255.0f);
            outPixels.emplace_back(decodeSrgb ? srgbToLinear(rgb) : rgb, data[base + 3] / 255.0f);
        }
        stbi_image_free(data);
        return true;
    }

    if (loadAsciiPpmImage(texturePath, decodeSrgb, outTexture, outPixels, outError))
    {
        outTexture.flipV = flipV ? 1 : 0;
        return true;
    }

    outError = "Failed to load texture: " + texturePath.string();
    return false;
}

bool loadTextureImageFromMemory(
    const unsigned char* textureBytes,
    size_t textureByteCount,
    bool flipV,
    bool decodeSrgb,
    TextureData& outTexture,
    std::vector<glm::vec4>& outPixels,
    std::string& outError)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = stbi_load_from_memory(textureBytes, static_cast<int>(textureByteCount), &width, &height, &channels, 4);
    if (!data)
    {
        outError = "Failed to decode embedded texture image";
        return false;
    }

    outTexture.width = width;
    outTexture.height = height;
    outTexture.pixelOffset = static_cast<int>(outPixels.size());
    outTexture.flipV = flipV ? 1 : 0;
    outTexture.wrapS = kWrapRepeat;
    outTexture.wrapT = kWrapRepeat;
    outPixels.reserve(outPixels.size() + width * height);
    for (int i = 0; i < width * height; ++i)
    {
        const int base = i * 4;
        const glm::vec3 rgb(
            data[base + 0] / 255.0f,
            data[base + 1] / 255.0f,
            data[base + 2] / 255.0f);
        outPixels.emplace_back(decodeSrgb ? srgbToLinear(rgb) : rgb, data[base + 3] / 255.0f);
    }
    stbi_image_free(data);
    return true;
}

bool loadHdrImage(
    const std::filesystem::path& texturePath,
    bool flipV,
    TextureData& outTexture,
    std::vector<glm::vec4>& outPixels,
    std::string& outError)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    float* data = stbi_loadf(texturePath.string().c_str(), &width, &height, &channels, 3);
    if (!data)
    {
        outError = "Failed to load HDR image: " + texturePath.string();
        return false;
    }

    outTexture.width = width;
    outTexture.height = height;
    outTexture.pixelOffset = static_cast<int>(outPixels.size());
    outTexture.flipV = flipV ? 1 : 0;
    outTexture.wrapS = kWrapRepeat;
    outTexture.wrapT = kWrapClampToEdge;
    outPixels.reserve(outPixels.size() + width * height);

    for (int i = 0; i < width * height; ++i)
    {
        const int base = i * 3;
        outPixels.emplace_back(
            glm::max(data[base + 0], 0.0f),
            glm::max(data[base + 1], 0.0f),
            glm::max(data[base + 2], 0.0f),
            1.0f);
    }

    stbi_image_free(data);
    return true;
}

bool loadMeshAsset(
    const std::filesystem::path& meshPath,
    int materialId,
    std::vector<Triangle>& outTriangles,
    std::vector<MeshMaterialDefinition>& outMaterials,
    std::string& outError)
{
    (void)materialId;

    const std::string ext = meshPath.extension().string();
    if (ext == ".gltf" || ext == ".glb")
    {
        return loadGltfMesh(meshPath, outTriangles, outMaterials, outError);
    }

    outError = "Only glTF/GLB mesh assets are supported now. Re-export this asset as .gltf or .glb: " + meshPath.string();
    return false;
}
