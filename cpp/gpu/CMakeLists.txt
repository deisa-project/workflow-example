cmake_minimum_required(VERSION 3.18)
project(sim-kokkos-doreisa LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Kokkos installation path
set(Kokkos_DIR "/usr/local/lib/cmake/Kokkos")

# Find Kokkos
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

# --- Add pybind11 (try system first) ---
find_package(pybind11 REQUIRED)

# Add executable
add_executable(sim-kokkos-doreisa sim-kokkos-doreisa.cpp)

# Include Python headers (for completeness)
target_include_directories(sim-kokkos-doreisa PRIVATE ${Python3_INCLUDE_DIRS})

# --- Link all dependencies ---
target_link_libraries(sim-kokkos-doreisa
    PRIVATE
        MPI::MPI_CXX
        Kokkos::kokkos
        pybind11::embed
        ${Python3_LIBRARIES}
)

# --- Enable Kokkos features ---
target_compile_definitions(sim-kokkos-doreisa PRIVATE
    KOKKOS_ENABLE_CUDA=ON
    KOKKOS_ENABLE_OPENMP=ON
)

# --- GPU Architecture ---
target_compile_options(sim-kokkos-doreisa PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=sm_90>
)
