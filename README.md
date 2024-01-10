# HelloVulkanTriangle
My learning project on Vulkan and C++.

C++ is compiled with CMAKE in shell into build directory:
	- '$ touch build' - create build direcory.
	- '$ cd build' - move to build directory
	- run '$ cmake..' - compile files
	- run '$ make' - link libraries

GLSL are shaders compiled with glslc into build directory:	
	- run '$ glslc -o TriangleVertex.spv ../Triangle.vert'
	- run '$ glslc -o TriangleFragment.spv ../Triangle.frag'


