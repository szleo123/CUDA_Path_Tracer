#include "scene.h"

#include <iostream>

int main()
{
    try
    {
        Scene scene("scenes/old_tree.json");
        std::cout << "materials=" << scene.materials.size() << "\n";
        for (size_t i = 0; i < scene.materials.size(); ++i)
        {
            const Material& m = scene.materials[i];
            std::cout
                << "material[" << i << "]"
                << " color=(" << m.color.r << "," << m.color.g << "," << m.color.b << ")"
                << " textureId=" << m.textureId
                << " metallicRoughnessTextureId=" << m.metallicRoughnessTextureId
                << " normalTextureId=" << m.normalTextureId
                << " alphaMode=" << m.alphaMode
                << " alphaCutoff=" << m.alphaCutoff
                << " doubleSided=" << m.doubleSided
                << "\n";
        }

        std::cout << "textures=" << scene.textures.size() << "\n";
        for (size_t i = 0; i < scene.textures.size(); ++i)
        {
            const TextureData& t = scene.textures[i];
            std::cout
                << "texture[" << i << "]"
                << " size=" << t.width << "x" << t.height
                << " pixelOffset=" << t.pixelOffset
                << " flipV=" << t.flipV
                << " wrapS=" << t.wrapS
                << " wrapT=" << t.wrapT
                << "\n";
        }

        for (size_t i = 0; i < scene.objects.size(); ++i)
        {
            const SceneObject& object = scene.objects[i];
            std::cout << "object[" << i << "] usedMaterials=";
            for (int materialId : object.usedMaterialIds)
            {
                std::cout << materialId << " ";
            }
            std::cout << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "inspect failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
