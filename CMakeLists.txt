# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(MyVulkanProject)

add_executable(MyVulkanProject main.cpp)

find_package(Vulkan REQUIRED)
target_include_directories(MyVulkanProject PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(MyVulkanProject PRIVATE ${Vulkan_LIBRARIES})

find_package(glfw3 REQUIRED)
target_link_libraries(MyVulkanProject PRIVATE glfw)

#find_package(glfw3 REQUIRED)
#target_link_libraries(MyVulkanProject PRIVATE ${GLFW3_LIBRARIES})

