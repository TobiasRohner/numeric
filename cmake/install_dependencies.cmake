include(FetchContent)
set(FETCHCONTENT_QUIET OFF)


set(NUMERIC_DEPENDENCIES)
mark_as_advanced(FORCE NUMERIC_DEPENDENCIES)


macro(numeric_register_dependency DEPENDENCY)
  set(NUMERIC_${DEPENDENCY}_REQUIRE_DOWNLOAD OFF)
  if (NOT NUMERIC_${DEPENDENCY}_REGISTERED)
    set(NUMERIC_USE_SYSTEM_${DEPENDENCY} AUTO CACHE STRING "Use the system version of ${DEPENDENCY}")
    set_property(CACHE NUMERIC_USE_SYSTEM_${DEPENDENCY} PROPERTY STRINGS AUTO ON OFF)
    if (NUMERIC_USE_SYSTEM_${DEPENDENCY} STREQUAL "AUTO")
      find_package(${DEPENDENCY} QUIET ${ARGN})
      if (NOT ${DEPENDENCY}_FOUND)
	set(NUMERIC_${DEPENDENCY}_REQUIRE_DOWNLOAD ON)
      endif()
    elseif (NUMERIC_USE_SYSTEM_${DEPENDENCY})
      find_package(${DEPENDENCY} REQUIRED ${ARGN})
    else()
      set(NUMERIC_${DEPENDENCY}_REQUIRE_DOWNLOAD ON)
    endif()
  set(NUMERIC_${DEPENDENCY}_REGISTERED ON)
  endif()
endmacro()




include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/fmt.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/Boost.cmake)

if (NUMERIC_ENABLE_HIP)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/HIP.cmake)
endif()

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
