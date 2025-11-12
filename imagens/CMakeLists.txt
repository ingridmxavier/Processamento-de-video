/******************************************************
 * Nome do Programa: CMakeLists
 * Integrantes da Equipe RoadWatch:
 *  - Fernanda Ayumi - RA: 11202321172
 *  - Ingrid Xavier - RA: 11202130019
 *  - Gabriel Rothen - RA: 11202321586
 *
 * Data: 2025
 * Descrição: Sistema de detecção de movimento com OpenCV
 * Execução no Ubuntu (terminal):
 *      $ mkdir build
 *      $ cd build
 *      $ cmake ..
 *      $ make
 *      $ ./spv
 ******************************************************/

cmake_minimum_required(VERSION 3.10)
project(spv)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(spv spv.cpp)
target_link_libraries(spv ${OpenCV_LIBS})

