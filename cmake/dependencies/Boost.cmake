numeric_register_dependency(Boost)


if (NUMERIC_Boost_REQUIRE_DOWNLOAD)

  set(BOOST_INCLUDE_LIBRARIES multiprecision math)
  set(BOOST_ENABLE_COMPATIBILITY_TARGETS ON)

  FetchContent_Declare(
    Boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG boost-1.85.0
    GIT_PROGRESS TRUE
  )

  list(APPEND NUMERIC_DEPENDENCIES Boost)

endif()
