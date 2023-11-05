target_sources(numeric PRIVATE variable.cpp)
target_sources(numeric PRIVATE hierarchical_file.cpp)
target_sources(numeric PRIVATE file_mode.cpp)

if (NUMERIC_ENABLE_NETCDF)
  target_sources(numeric PRIVATE netcdf_variable.cpp)
  target_sources(numeric PRIVATE netcdf_file.cpp)
endif()
