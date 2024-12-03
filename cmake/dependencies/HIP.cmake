numeric_register_dependency(HIP)


if (NUMERIC_HIP_REQUIRE_DOWNLOAD)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/HIP/CMakeLists.txt.in
    ${FETCHCONTENT_BASE_DIR}/hip-subbuild/CMakeLists.txt
    @ONLY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} .
    WORKING_DIRECTORY ${FETCHCONTENT_BASE_DIR}/hip-subbuild
    RESULT_VARIABLE output
  )
  if (NOT output EQUAL 0)
    message(FATAL_ERROR "Failed to download HIP")
  endif ()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    WORKING_DIRECTORY ${FETCHCONTENT_BASE_DIR}/hip-subbuild
    RESULT_VARIABLE output
  )
  if (NOT output EQUAL 0)
    message(FATAL_ERROR "Failed to build HIP")
  endif ()

  list(APPEND CMAKE_MODULE_PATH ${FETCHCONTENT_BASE_DIR}/hipother-install/lib/cmake/hip)
  find_package(HIP REQUIRED)

  set(HIP_INCLUDE_DIR ${HIP_ROOT_DIR}/include)

  add_library(hip INTERFACE)
  target_compile_definitions(hip INTERFACE "-DNUMERIC_ENABLE_HIP=1")
  target_include_directories(hip INTERFACE ${HIP_INCLUDE_DIR})
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -I${HIP_INCLUDE_DIR}")
  if(HIP_PLATFORM STREQUAL "amd")
    target_compile_definitions(hip INTERFACE __HIP_PLATFORM_AMD__)
    message(WARNING "No Optimization Flags set for AMD HIP")
  elseif(HIP_PLATFORM STREQUAL "nvidia")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(hip INTERFACE CUDA::cudart)
    target_link_libraries(hip INTERFACE CUDA::cuda_driver)
    target_link_libraries(hip INTERFACE CUDA::nvrtc)
    target_compile_definitions(hip INTERFACE __HIP_PLATFORM_NVIDIA__)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -D__HIP_PLATFORM_NVIDIA__ -I ${CUDAToolkit_INCLUDE_DIRS}")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -G")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --dopt=on -DNDEBUG")
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --dopt=on -G -DNDEBUG")
    elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --dopt=on -DNDEBUG")
    endif()
  else()
    message(FATAL_ERROR "Unknown platform: ${HIP_PLATFORM}")
  endif()

endif()
