add_executable(bm)

add_subdirectory(memory)
add_subdirectory(math)

set_target_properties(bm PROPERTIES CXX_STANDARD 20)
target_link_libraries(bm PUBLIC numeric benchmark::benchmark benchmark::benchmark_main)
