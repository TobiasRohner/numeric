cmake_minimum_required(VERSION 3.24)
project(numeric VERSION 0.1 LANGUAGES C CXX)


include(CMakeDependentOption)


option(NUMERIC_ENABLE_HIP "Enable GPU support using HIP" ON)
option(NUMERIC_BUILD_TESTS "Build Tests for the Library" ON)
option(NUMERIC_BUILD_EXAMPLES "Build Examples" ON)
option(NUMERIC_BUILD_BENCHMARKS "Build Benchmarks for the Library" ON)
option(NUMERIC_ENABLE_EIGEN "Enable Matrix Operations using Eigen" ON)
option(NUMERIC_ENABLE_HDF5 "Enable I/O using HDF5" ON)
option(NUMERIC_ENABLE_NETCDF "Enable I/O using NetCDF4" ON)
option(NUMERIC_ENABLE_MPI "Parallelize using MPI" ON)
option(NUMERIC_ENABLE_CLANG_TIDY "Run clang-tidy while compiling" OFF)
option(NUMERIC_BUILD_DOCS "Build the documentation" ON)
cmake_dependent_option(NUMERIC_DOXYGEN_QUIET "Suppress doxygen output" OFF "NUMERIC_BUILD_DOCS" OFF)


if (NUMERIC_ENABLE_CLANG_TIDY)
  find_program(NUMERIC_CLANG_TIDY_EXE NAMES "clang-tidy")
  set(NUMERIC_CLANG_TIDY_CMD "${NUMERIC_CLANG_TIDY_EXE}\;-checks=*,-llvmlibc-*,-modernize-use-trailing-return-type,-readability-identifier-length,-fuchsia-*,-readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers,-google-readability-avoid-underscore-in-googletest-name,-altera-id-dependent-backward-branch,-altera-unroll-loops,-portability-simd-intrinsics")
endif()


include(FetchContent)
set(FETCHCONTENT_QUIET OFF)


add_library(numeric)
set_target_properties(numeric PROPERTIES CXX_STANDARD 20)
set(NUMERIC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NUMERIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${NUMERIC_SOURCE_DIR})
target_include_directories(numeric PUBLIC ${NUMERIC_INCLUDE_DIR})


include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_ref_el_source.cmake)


FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)
set(BOOST_INCLUDE_LIBRARIES multiprecision math)
FetchContent_Declare(
  Boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG boost-1.85.0
  GIT_PROGRESS TRUE
)
if (NUMERIC_ENABLE_EIGEN)
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )
endif()
if (NUMERIC_ENABLE_NETCDF)
  if (NUMERIC_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C)
    include_directories(${MPI_C_INCLUDE_PATH})
  endif()
  set(HDF5_PREFER_PARALLEL ${NUMERIC_ENABLE_MPI})
  set(BUILD_UTILITIES OFF)
  set(ENABLE_EXAMPLES OFF)
  set(ENABLE_TESTS OFF)
  set(ENABLE_DAP ON)
  set(ENABLE_HDF5 ON)
  set(HDF5_PARALLEL ${NUMERIC_ENABLE_MPI})
  set(ENABLE_NETCDF4 ON)
  set(ENABLE_PNETCDF OFF)
  set(ENABLE_PARALLEL4 ${NUMERIC_ENABLE_MPI})
  set(ENABLE_PARALLEL ${NUMERIC_ENABLE_MPI})
  set(USE_PARALLEL ${NUMERIC_ENABLE_MPI})
  FetchContent_Declare(
    NetCDF
    GIT_REPOSITORY https://github.com/Unidata/netcdf-c.git
    GIT_TAG v4.9.3-rc1
    OVERRIDE_FIND_PACKAGE
  )
endif()
if (NUMERIC_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.13.0
  )
endif()
if (NUMERIC_BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
  )
endif()

FetchContent_MakeAvailable(fmt Boost)
if (NUMERIC_ENABLE_EIGEN)
  FetchContent_MakeAvailable(Eigen)
  set(EIGEN_BUILD_DOC OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
endif()
if (NUMERIC_BUILD_TESTS)
  FetchContent_MakeAvailable(googletest)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
if (NUMERIC_BUILD_BENCHMARKS)
  FetchContent_MakeAvailable(googlebenchmark)
endif()
if (NUMERIC_ENABLE_NETCDF)
  FetchContent_MakeAvailable(NetCDF)
  find_package(NetCDF REQUIRED)
  target_link_libraries(numeric PUBLIC netCDF::netcdf)
endif()


target_link_libraries(numeric PUBLIC fmt::fmt)
target_link_libraries(numeric PUBLIC Boost::multiprecision Boost::math Boost::headers)
if(NUMERIC_ENABLE_CLANG_TIDY)
  set_target_properties(numeric PROPERTIES CXX_CLANG_TIDY ${NUMERIC_CLANG_TIDY_CMD})
endif()

if(NUMERIC_ENABLE_HIP)
  include("cmake/setup_hip.cmake")
  target_link_libraries(numeric PUBLIC hip)
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_HIP=1")
else()
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_HIP=0")
endif()

if(NUMERIC_ENABLE_EIGEN)
  target_link_libraries(numeric PUBLIC Eigen3::Eigen)
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_EIGEN=1")
else()
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_EIGEN=0")
endif()

if (NUMERIC_ENABLE_NETCDF)
  target_include_directories(numeric PUBLIC ${NETCDF_INCLUD_DIRS})
  target_link_libraries(numeric PUBLIC ${NETCDF_LIBRARIES})
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_NETCDF=1")
else()
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_NETCDF=0")
endif()

set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE STRING "Location of the \"data\" directory")
configure_file(${NUMERIC_INCLUDE_DIR}/numeric/config.hpp.in ${NUMERIC_INCLUDE_DIR}/numeric/config.hpp @ONLY)


if(NUMERIC_BUILD_TESTS)
  add_subdirectory(test)
endif()


if(NUMERIC_BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()


if (NUMERIC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()


if (NUMERIC_BUILD_DOCS)
  add_subdirectory(doc)
endif()
