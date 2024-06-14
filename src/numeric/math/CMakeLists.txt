if(NUMERIC_ENABLE_HIP)
  target_sources(numeric PRIVATE sum_device.cpp)
endif()

add_subdirectory(quad)
