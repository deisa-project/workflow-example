cmake_minimum_required(VERSION 3.18)
project(MyKokkosProject LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Kokkos installation path
set(Kokkos_DIR "/usr/local/lib/cmake/Kokkos")

# Find Kokkos
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

# Add executable
add_executable(sim-kokkos sim-kokkos.cpp)

# Link Kokkos
target_link_libraries(sim-kokkos MPI::MPI_CXX Kokkos::kokkos)

# Enable CUDA and OpenMP for your target
target_compile_definitions(sim-kokkos PRIVATE KOKKOS_ENABLE_CUDA=ON KOKKOS_ENABLE_OPENMP=ON)

# Make sure we use the same architecture as Kokkos was built for
target_compile_options(sim-kokkos PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=sm_90>)

