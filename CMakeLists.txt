cmake_minimum_required(VERSION 3.20)
project(genetic_algorithm)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 20)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(gen_alg main.cpp)

target_link_libraries(gen_alg ${OpenCV_LIBS})
