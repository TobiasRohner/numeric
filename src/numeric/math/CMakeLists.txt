if(NUMERIC_ENABLE_HIP)
  target_sources(numeric PRIVATE reduce.cpp)
  target_sources(numeric PRIVATE constrained_linear_operator.cpp)
endif()

add_subdirectory(quad)
add_subdirectory(graph)
add_subdirectory(fes)
