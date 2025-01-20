numeric_register_dependency(NetCDF)


if (NUMERIC_NetCDF_REQUIRE_DOWNLOAD)

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/HDF5.cmake)

  if (NUMERIC_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C)
    include_directories(${MPI_C_INCLUDE_PATH})
  endif()

  set(HDF5_PREFER_PARALLEL ${NUMERIC_ENABLE_MPI})
  set(BUILD_UTILITIES OFF)
  set(ENABLE_EXAMPLES OFF)
  set(ENABLE_TESTS OFF)
  set(HDF5_PARALLEL ${NUMERIC_ENABLE_MPI})
  set(ENABLE_NETCDF4 ON)
  set(ENABLE_PNETCDF OFF)
  set(ENABLE_PARALLEL ${NUMERIC_ENABLE_MPI})

  FetchContent_Declare(
    NetCDF
    GIT_REPOSITORY https://github.com/Unidata/netcdf-c.git
    GIT_TAG v4.9.2
  )

  list(APPEND NUMERIC_DEPENDENCIES NetCDF)

endif()
