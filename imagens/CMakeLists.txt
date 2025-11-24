cmake_minimum_required(VERSION 3.10)
project(SPV_ROADWATCH)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

add_executable(spv spv.cpp)
target_link_libraries(spv ${OpenCV_LIBS})

