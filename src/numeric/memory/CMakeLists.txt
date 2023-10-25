target_sources(numeric PRIVATE memory_type.cpp)
target_sources(numeric PRIVATE slice.cpp)

if(NUMERIC_ENABLE_HIP)
  target_sources(numeric PRIVATE copy_device_device.cpp)
endif()
