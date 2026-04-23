cmake_minimum_required(VERSION 3.14)
project(RocketLib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
endif()

# Fetch pybind11 automatically
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.10.4
)
FetchContent_MakeAvailable(pybind11)

# Core library for C++ usage
add_library(rocket_core STATIC 
            src/tensor.cpp 
            src/activation.cpp 
            src/layer.cpp
            src/loss.cpp
            src/model.cpp
            src/optimizer.cpp)
target_include_directories(rocket_core PUBLIC include)
set_target_properties(rocket_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Build the Python extension module
pybind11_add_module(rocket src/rocket.cpp)
target_link_libraries(rocket PRIVATE rocket_core)


# Optional: set output directory to build so testing can easily find it
set_target_properties(rocket PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
