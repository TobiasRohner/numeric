if (NUMERIC_ENABLE_HIP)
  add_subdirectory(hip)
endif()

add_subdirectory(memory)
add_subdirectory(utils)
add_subdirectory(io)
add_subdirectory(math)
