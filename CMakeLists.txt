cmake_minimum_required(VERSION 3.2)

project(fluid417)

add_executable(main src/main.cpp)

target_include_directories(main PUBLIC include)

option(LIBIGL_WITH_OPENGL               "Use OpenGL"            ON)
option(LIBIGL_WITH_OPENGL_GLFW          "Use GLFW"              ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI    "Use ImGui"             ON)

find_package(libigl CONFIG REQUIRED)

find_package(TBB CONFIG REQUIRED)

target_link_libraries(main igl::core igl::common igl::opengl igl::opengl_glfw TBB::tbb)