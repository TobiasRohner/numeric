numeric_register_dependency(HDF5)


if (NUMERIC_HDF5_REQUIRE_DOWNLOAD)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/HDF5/CMakeLists.txt.in
    ${FETCHCONTENT_BASE_DIR}/hdf5-subbuild/CMakeLists.txt
    @ONLY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} .
    WORKING_DIRECTORY ${FETCHCONTENT_BASE_DIR}/hdf5-subbuild
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    WORKING_DIRECTORY ${FETCHCONTENT_BASE_DIR}/hdf5-subbuild
    COMMAND_ERROR_IS_FATAL ANY
  )

  set(HDF5_ROOT ${FETCHCONTENT_BASE_DIR}/hdf5-install)

endif()
