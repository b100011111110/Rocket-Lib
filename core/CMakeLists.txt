cmake_minimum_required(VERSION 3.14)
project(RocketLib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch pybind11 automatically
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.10.4
)
FetchContent_MakeAvailable(pybind11)

# Build the Python extension module
pybind11_add_module(rocket src/rocket.cpp 
                    src/tensor.cpp 
                    src/activation.cpp 
                    src/layer.cpp
                    src/loss.cpp
                    src/model.cpp
                    src/optimizer.cpp)

# Include directories for definitions
target_include_directories(rocket PRIVATE include)

# Optional: set output directory to build so testing can easily find it
set_target_properties(rocket PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
