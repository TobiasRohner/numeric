numeric_register_dependency(hipcc)


if (NUMERIC_hipcc_REQUIRE_DOWNLOAD)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/hipcc/CMakeLists.txt.in
    ${FETCHCONTENT_BASE_DIR}/hipcc-subbuild/CMakeLists.txt
    @ONLY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} .
    WORKING_DIRECTORY ${FETCHCONTENT_BASE_DIR}/hipcc-subbuild
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    WORKING_DIRECTORY ${FETCHCONTENT_BASE_DIR}/hipcc-subbuild
    COMMAND_ERROR_IS_FATAL ANY
  )

  set(HIPCC_BIN_DIR ${FETCHCONTENT_BASE_DIR}/hipcc-install/bin)

endif()
