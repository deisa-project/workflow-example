cmake_minimum_required(VERSION 3.18)
project(sim-deisa-ray LANGUAGES CXX)

# ---- Options ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# ---- Dependencies ----
find_package(MPI REQUIRED)
find_package(pybind11 REQUIRED)

# ---- Source ----
add_executable(sim-deisa-ray sim-deisa-ray.cpp)

# ---- Include and link ----
target_link_libraries(sim-deisa-ray
    PRIVATE
        MPI::MPI_CXX
        pybind11::embed   # for embedding Python in C++
)

# ---- Compiler warnings ----
target_compile_options(sim-deisa-ray PRIVATE
    -Wall -Wextra -Wpedantic
)

# ---- Optional: print summary ----
message(STATUS "MPI include dir: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "pybind11 include dir: ${pybind11_INCLUDE_DIRS}")

