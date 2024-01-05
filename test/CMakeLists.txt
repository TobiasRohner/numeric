add_executable(test)

add_subdirectory(memory)
add_subdirectory(io)
add_subdirectory(utils)

set_target_properties(test PROPERTIES CXX_STANDARD 20)
target_link_libraries(test PUBLIC numeric gtest gtest_main)
