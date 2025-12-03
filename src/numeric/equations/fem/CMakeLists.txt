if(NUMERIC_ENABLE_HIP)
  target_sources(numeric PRIVATE finite_element_vector.cpp)
  target_sources(numeric PRIVATE finite_element_matrix.cpp)
endif()
