cmake_minimum_required(VERSION 3.20)
project(genetic_algorithm)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 20)
include_directories(${OpenCV_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})

add_executable(gen_alg_seq sequential.cpp)
target_link_libraries(gen_alg_seq ${OpenCV_LIBS})
message(STATUS ${OpenCV_LIBS})

add_executable(gen_alg_omp openmp.cpp)
target_link_libraries(gen_alg_omp ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable(gen_alg_mpi mpi.cpp)
target_link_libraries(gen_alg_mpi ${OpenCV_LIBS} MPI::MPI_CXX)
