cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(custom_ops LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_EXTENSIONS OFF)
add_subdirectory(${CMAKE_SOURCE_DIR}/kokkos)
message(STATUS "kokkos include ${Kokkos_INCLUDE_DIRS_RET}")
message(STATUS "kokkos include_dirs ${KOKKOS_INCLUDE_DIRS}")
include_directories(${Kokkos_INCLUDE_DIRS_RET})

find_package(Torch REQUIRED)
find_package(TorchSparse REQUIRED)

add_executable(hello-world simple_view_lambda.cpp)
target_link_libraries(hello-world "${TORCH_LIBRARIES}")
target_link_libraries(hello-world TorchSparse::TorchSparse)
target_link_libraries(hello-world kokkos)
set_property(TARGET hello-world PROPERTY CXX_STANDARD 14)
