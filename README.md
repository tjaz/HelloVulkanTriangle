# HelloVulkanTriangle
My learning project on Vulkan and C++.

C++ is compiled with CMAKE in the shell into the build directory:
        - '$ touch build' - create the build directory.
        - '$ cd build' - move to the build directory.
        - run '$ cmake ..' - compile files.
        - run '$ make' - link libraries.

GLSL shaders are compiled with glslc into the build directory:
        - run '$ glslc -o TriangleVertex.spv ../Triangle.vert'
        - run '$ glslc -o TriangleFragment.spv ../Triangle.frag'
