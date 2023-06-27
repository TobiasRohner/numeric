cmake_minimum_required(VERSION 3.21)
project(numeric VERSION 0.1 LANGUAGES CXX)


option(NUMERIC_ENABLE_HIP "Enable GPU support using HIP" ON)



add_library(numeric)
set_target_properties(numeric PROPERTIES CXX_STANDARD 20)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(numeric PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NUMERIC_ENABLE_HIP)
  include("cmake/setup_hip.cmake")
  target_link_libraries(numeric PUBLIC hip)
endif()

set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE STRING "Location of the \"data\" directory")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/include/numeric/config.hpp @ONLY)
target_include_directories(numeric PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)


add_executable(test test.cpp)
set_target_properties(test PROPERTIES CXX_STANDARD 20)
target_link_libraries(test PUBLIC numeric)
