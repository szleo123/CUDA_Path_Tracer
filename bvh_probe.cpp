#include <iostream>
#include <vector>
#include <algorithm>
#include "scene.h"
int main(){
  Scene s("scenes/porsche911.json");
  std::cout << "triangleBvhNodes=" << s.triangleBvhNodes.size() << "\n";
  if(s.objects.empty()){ return 0; }
  for(const auto& obj : s.objects){
    if(obj.type != SceneObjectType::Mesh || obj.localBvhNodes.empty()) continue;
    int root = 0;
    int maxDepth = 0;
    int maxStack = 0;
    std::vector<int> stack{root};
    while(!stack.empty()){
      maxStack = std::max(maxStack, (int)stack.size());
      int idx = stack.back(); stack.pop_back();
      const auto& n = obj.localBvhNodes[idx];
      if(n.triCount==0){ stack.push_back(n.leftFirst); stack.push_back(n.rightChild); }
    }
    std::cout << "mesh nodes=" << obj.localBvhNodes.size() << " stack=" << maxStack << " tris=" << obj.localTriangles.size() << "\n";
  }
}
