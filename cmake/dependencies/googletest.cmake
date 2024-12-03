numeric_register_dependency(googletest)


if (NUMERIC_googletest_REQUIRE_DOWNLOAD)

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.13.0
  )

  list(APPEND NUMERIC_DEPENDENCIES googletest)

endif()
