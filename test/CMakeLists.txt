add_executable(test)

add_subdirectory(memory)
add_subdirectory(io)
add_subdirectory(utils)

set_target_properties(test PROPERTIES CXX_STANDARD 20)
if(NUMERIC_ENABLE_CLANG_TIDY)
  set_target_properties(test PROPERTIES CXX_CLANG_TIDY ${NUMERIC_CLANG_TIDY_CMD})
endif()
target_link_libraries(test PUBLIC numeric gtest gtest_main)
