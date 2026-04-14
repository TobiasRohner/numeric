numeric_register_dependency(fmt)


if (NUMERIC_fmt_REQUIRE_DOWNLOAD)

  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.1.0
  )

  list(APPEND NUMERIC_DEPENDENCIES fmt)

endif()
