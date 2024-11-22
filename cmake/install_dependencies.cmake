include(FetchContent)
set(FETCHCONTENT_QUIET OFF)


set(NUMERIC_DEPENDENCIES)
mark_as_advanced(FORCE NUMERIC_DEPENDENCIES)


include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/fmt.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/Boost.cmake)

if (NUMERIC_ENABLE_EIGEN)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/Eigen.cmake)
endif()

if (NUMERIC_ENABLE_NETCDF)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/NetCDF.cmake)
endif()

if (NUMERIC_BUILD_TESTS)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/googletest.cmake)
endif()

if (NUMERIC_BUILD_BENCHMARKS)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/googlebenchmark.cmake)
endif()


FetchContent_MakeAvailable(${NUMERIC_DEPENDENCIES})
