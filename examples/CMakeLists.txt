add_executable(kernel)
target_sources(kernel PUBLIC kernel.cpp)
set_target_properties(kernel PROPERTIES CXX_STANDARD 20)
target_link_libraries(kernel PUBLIC numeric)

add_executable(fd_wave)
target_sources(fd_wave PUBLIC fd_wave.cpp)
set_target_properties(fd_wave PROPERTIES CXX_STANDARD 20)
target_link_libraries(fd_wave PUBLIC numeric)
