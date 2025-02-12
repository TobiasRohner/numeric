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
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)


if (NUMERIC_ENABLE_CLANG_TIDY)
  find_program(NUMERIC_CLANG_TIDY_EXE NAMES "clang-tidy")
  set(NUMERIC_CLANG_TIDY_CMD "${NUMERIC_CLANG_TIDY_EXE}\;-checks=*,-llvmlibc-*,-modernize-use-trailing-return-type,-readability-identifier-length,-fuchsia-*,-readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers,-google-readability-avoid-underscore-in-googletest-name,-altera-id-dependent-backward-branch,-altera-unroll-loops,-portability-simd-intrinsics")
endif()




add_library(numeric)
set_target_properties(numeric PROPERTIES CXX_STANDARD 20)
set(NUMERIC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NUMERIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${NUMERIC_SOURCE_DIR})
target_include_directories(numeric PUBLIC ${NUMERIC_INCLUDE_DIR})

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_dependencies.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_ref_el_source.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_basis_lagrange_source.cmake)


target_link_libraries(numeric PUBLIC fmt::fmt)
target_link_libraries(numeric PUBLIC Boost::headers)
if(NUMERIC_ENABLE_CLANG_TIDY)
  set_target_properties(numeric PROPERTIES CXX_CLANG_TIDY ${NUMERIC_CLANG_TIDY_CMD})
endif()

if(NUMERIC_ENABLE_HIP)
  #include("cmake/setup_hip.cmake")
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
  target_link_libraries(numeric PUBLIC netCDF::netcdf)
  find_package(HDF5 REQUIRED COMPONENTS C HL)
  target_link_libraries(numeric PUBLIC hdf5::hdf5_hl hdf5::hdf5)
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_NETCDF=1")
else()
  target_compile_definitions(numeric PUBLIC "-DNUMERIC_ENABLE_NETCDF=0")
endif()

if (NUMERIC_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS C)
  target_compile_definitions(numeric PUBLIC ${MPI_C_COMPILE_FLAGS})
  target_include_directories(numeric PUBLIC ${MPI_C_INCLUDE_PATH})
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
