cmake_minimum_required(VERSION 3.21)
project(numeric VERSION 0.1 LANGUAGES CXX)


option(NUMERIC_ENABLE_HIP "Enable GPU support using HIP" ON)
option(NUMERIC_BUILD_TESTS "Build Tests for the Library" ON)
option(NUMERIC_BUILD_BENCHMARKS "Build Benchmarks for the Library" ON)
option(NUMERIC_ENABLE_EIGEN "Enable Matrix Operations using Eigen" ON)


include(FetchContent)


add_library(numeric)
set_target_properties(numeric PROPERTIES CXX_STANDARD 20)
set(NUMERIC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NUMERIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${NUMERIC_SOURCE_DIR})
target_include_directories(numeric PUBLIC ${NUMERIC_INCLUDE_DIR})

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)
FetchContent_MakeAvailable(fmt)
target_link_libraries(numeric PUBLIC fmt::fmt)

if(NUMERIC_ENABLE_HIP)
  include("cmake/setup_hip.cmake")
  target_link_libraries(numeric PUBLIC hip)
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_HIP=1")
else()
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_HIP=0")
endif()

if(NUMERIC_ENABLE_EIGEN)
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )
  set(EIGEN_BUILD_DOC OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  FetchContent_MakeAvailable(Eigen)
  target_link_libraries(numeric PUBLIC Eigen3::Eigen)
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_EIGEN=1")
else()
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_EIGEN=0")
endif()

set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE STRING "Location of the \"data\" directory")
configure_file(${NUMERIC_INCLUDE_DIR}/numeric/config.hpp.in ${NUMERIC_INCLUDE_DIR}/numeric/config.hpp @ONLY)


if(NUMERIC_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.13.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(test)
endif()


if(NUMERIC_BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
  )
  FetchContent_MakeAvailable(googlebenchmark)
  add_subdirectory(benchmark)
endif()
