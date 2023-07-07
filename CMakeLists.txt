cmake_minimum_required(VERSION 3.21)
project(numeric VERSION 0.1 LANGUAGES CXX)


option(NUMERIC_ENABLE_HIP "Enable GPU support using HIP" ON)



add_library(numeric)
set_target_properties(numeric PROPERTIES CXX_STANDARD 20)
set(NUMERIC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NUMERIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${NUMERIC_SOURCE_DIR})
target_include_directories(numeric PUBLIC ${NUMERIC_INCLUDE_DIR})

if(NUMERIC_ENABLE_HIP)
  include("cmake/setup_hip.cmake")
  target_link_libraries(numeric PUBLIC hip)
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_HIP=1")
else()
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_HIP=0")
endif()

set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE STRING "Location of the \"data\" directory")
configure_file(${NUMERIC_INCLUDE_DIR}/numeric/config.hpp.in ${NUMERIC_INCLUDE_DIR}/numeric/config.hpp @ONLY)


add_executable(test test.cpp)
set_target_properties(test PROPERTIES CXX_STANDARD 20)
target_link_libraries(test PUBLIC numeric)
