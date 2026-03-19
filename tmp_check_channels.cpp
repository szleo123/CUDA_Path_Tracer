#include <stb_image.h>
#include <iostream>
int main(){
  const char* files[] = {
    "assets/mario/textures/submesh_0_baseColor.png",
    "assets/mario/textures/submesh_1_baseColor.png",
    "assets/mario/textures/submesh_2_baseColor.png"
  };
  for (const char* f : files) {
    int w=0,h=0,c=0;
    if (stbi_info(f,&w,&h,&c)) {
      std::cout << f << " " << w << "x" << h << " channels=" << c << "\n";
    } else {
      std::cout << f << " failed\n";
    }
  }
}
