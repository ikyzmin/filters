cmake_minimum_required(VERSION 3.6)
project(filters)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /home/openCV/release_qt)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(filters ${SOURCE_FILES})

target_link_libraries(filters ${OpenCV_LIBS})
