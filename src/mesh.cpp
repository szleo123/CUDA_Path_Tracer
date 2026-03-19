#include "mesh.h"

#include <stb_image.h>

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
struct ObjIndex
{
    int position = -1;
    int texcoord = -1;
    int normal = -1;
};

int resolveObjIndex(int rawIndex, int count)
{
    if (rawIndex > 0)
    {
        return rawIndex - 1;
    }
    if (rawIndex < 0)
    {
        return count + rawIndex;
    }
    return -1;
}

bool parseObjIndexToken(const std::string& token, ObjIndex& outIndex)
{
    size_t firstSlash = token.find('/');
    if (firstSlash == std::string::npos)
    {
        outIndex.position = std::stoi(token);
        return true;
    }

    std::string p = token.substr(0, firstSlash);
    size_t secondSlash = token.find('/', firstSlash + 1);
    std::string t;
    std::string n;

    if (secondSlash == std::string::npos)
    {
        t = token.substr(firstSlash + 1);
    }
    else
    {
        t = token.substr(firstSlash + 1, secondSlash - firstSlash - 1);
        n = token.substr(secondSlash + 1);
    }

    if (!p.empty())
    {
        outIndex.position = std::stoi(p);
    }
    if (!t.empty())
    {
        outIndex.texcoord = std::stoi(t);
    }
    if (!n.empty())
    {
        outIndex.normal = std::stoi(n);
    }
    return true;
}

glm::vec3 transformPoint(const glm::mat4& transform, const glm::vec3& p)
{
    return glm::vec3(transform * glm::vec4(p, 1.0f));
}

glm::vec3 transformNormal(const glm::mat4& invTranspose, const glm::vec3& n)
{
    return glm::normalize(glm::vec3(invTranspose * glm::vec4(n, 0.0f)));
}

bool readPpmToken(std::istream& input, std::string& token)
{
    token.clear();
    char ch = 0;
    while (input.get(ch))
    {
        if (ch == '#')
        {
            std::string ignored;
            std::getline(input, ignored);
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            continue;
        }
        token.push_back(ch);
        break;
    }

    if (token.empty())
    {
        return false;
    }

    while (input.get(ch))
    {
        if (ch == '#')
        {
            std::string ignored;
            std::getline(input, ignored);
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

bool loadAsciiPpmImage(
    const std::filesystem::path& texturePath,
    TextureData& outTexture,
    std::vector<glm::vec3>& outPixels,
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
        outError = "Malformed ASCII PPM texture: " + texturePath.string();
        return false;
    }

    const int width = std::stoi(widthToken);
    const int height = std::stoi(heightToken);
    const int maxValue = std::stoi(maxValueToken);
    if (width <= 0 || height <= 0 || maxValue <= 0)
    {
        outError = "Invalid ASCII PPM header: " + texturePath.string();
        return false;
    }

    outTexture.width = width;
    outTexture.height = height;
    outTexture.pixelOffset = static_cast<int>(outPixels.size());
    outPixels.reserve(outPixels.size() + width * height);

    for (int i = 0; i < width * height; ++i)
    {
        std::string rToken;
        std::string gToken;
        std::string bToken;
        if (!readPpmToken(input, rToken)
            || !readPpmToken(input, gToken)
            || !readPpmToken(input, bToken))
        {
            outError = "Unexpected end of ASCII PPM pixel data: " + texturePath.string();
            return false;
        }

        outPixels.emplace_back(
            std::stof(rToken) / maxValue,
            std::stof(gToken) / maxValue,
            std::stof(bToken) / maxValue);
    }

    return true;
}
}

bool loadTextureImage(
    const std::filesystem::path& texturePath,
    TextureData& outTexture,
    std::vector<glm::vec3>& outPixels,
    std::string& outError)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = stbi_load(texturePath.string().c_str(), &width, &height, &channels, 3);
    if (data)
    {
        outTexture.width = width;
        outTexture.height = height;
        outTexture.pixelOffset = static_cast<int>(outPixels.size());
        outPixels.reserve(outPixels.size() + width * height);

        for (int i = 0; i < width * height; ++i)
        {
            const int base = i * 3;
            outPixels.emplace_back(
                data[base + 0] / 255.0f,
                data[base + 1] / 255.0f,
                data[base + 2] / 255.0f);
        }

        stbi_image_free(data);
        return true;
    }

    if (loadAsciiPpmImage(texturePath, outTexture, outPixels, outError))
    {
        return true;
    }

    outError = "Failed to load texture: " + texturePath.string();
    return false;
}

bool loadObjMesh(
    const std::filesystem::path& meshPath,
    const glm::mat4& transform,
    const glm::mat4& invTranspose,
    int materialId,
    std::vector<Triangle>& outTriangles,
    std::string& outError)
{
    const size_t startTriangleCount = outTriangles.size();

    std::ifstream file(meshPath);
    if (!file)
    {
        outError = "Failed to open mesh file: " + meshPath.string();
        return false;
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> texcoords;
    std::vector<glm::vec3> normals;
    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream stream(line);
        std::string prefix;
        stream >> prefix;

        if (prefix == "v")
        {
            glm::vec3 p(0.0f);
            stream >> p.x >> p.y >> p.z;
            positions.push_back(p);
        }
        else if (prefix == "vt")
        {
            glm::vec2 uv(0.0f);
            stream >> uv.x >> uv.y;
            texcoords.push_back(uv);
        }
        else if (prefix == "vn")
        {
            glm::vec3 n(0.0f);
            stream >> n.x >> n.y >> n.z;
            normals.push_back(glm::normalize(n));
        }
        else if (prefix == "f")
        {
            std::vector<ObjIndex> faceIndices;
            std::string token;
            while (stream >> token)
            {
                ObjIndex index;
                if (!parseObjIndexToken(token, index))
                {
                    outError = "Failed to parse face token in mesh: " + meshPath.string();
                    return false;
                }
                index.position = resolveObjIndex(index.position, static_cast<int>(positions.size()));
                index.texcoord = resolveObjIndex(index.texcoord, static_cast<int>(texcoords.size()));
                index.normal = resolveObjIndex(index.normal, static_cast<int>(normals.size()));
                if (index.position < 0 || index.position >= static_cast<int>(positions.size()))
                {
                    outError = "Face references invalid position index in mesh: " + meshPath.string();
                    return false;
                }
                faceIndices.push_back(index);
            }

            if (faceIndices.size() < 3)
            {
                continue;
            }

            for (size_t i = 1; i + 1 < faceIndices.size(); ++i)
            {
                const ObjIndex triIndices[3] = {
                    faceIndices[0],
                    faceIndices[i],
                    faceIndices[i + 1]
                };

                Triangle tri{};
                tri.materialId = materialId;
                tri.hasVertexNormals = 1;
                tri.hasUVs = 1;

                const glm::vec3 localPositions[3] = {
                    positions[triIndices[0].position],
                    positions[triIndices[1].position],
                    positions[triIndices[2].position]
                };

                tri.p0 = transformPoint(transform, localPositions[0]);
                tri.p1 = transformPoint(transform, localPositions[1]);
                tri.p2 = transformPoint(transform, localPositions[2]);

                const glm::vec3 faceNormal = glm::normalize(glm::cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
                tri.geometricNormal = faceNormal;

                if (triIndices[0].normal >= 0 && triIndices[1].normal >= 0 && triIndices[2].normal >= 0)
                {
                    tri.n0 = transformNormal(invTranspose, normals[triIndices[0].normal]);
                    tri.n1 = transformNormal(invTranspose, normals[triIndices[1].normal]);
                    tri.n2 = transformNormal(invTranspose, normals[triIndices[2].normal]);
                }
                else
                {
                    tri.hasVertexNormals = 0;
                    tri.n0 = faceNormal;
                    tri.n1 = faceNormal;
                    tri.n2 = faceNormal;
                }

                if (triIndices[0].texcoord >= 0 && triIndices[1].texcoord >= 0 && triIndices[2].texcoord >= 0)
                {
                    tri.uv0 = texcoords[triIndices[0].texcoord];
                    tri.uv1 = texcoords[triIndices[1].texcoord];
                    tri.uv2 = texcoords[triIndices[2].texcoord];
                }
                else
                {
                    tri.hasUVs = 0;
                    tri.uv0 = glm::vec2(0.0f);
                    tri.uv1 = glm::vec2(0.0f);
                    tri.uv2 = glm::vec2(0.0f);
                }

                outTriangles.push_back(tri);
            }
        }
    }

    if (outTriangles.size() == startTriangleCount)
    {
        outError = "Mesh contains no triangles: " + meshPath.string();
        return false;
    }

    return true;
}
