numeric_register_dependency(googlebenchmark)


if (NUMERIC_googlebenchmark_REQUIRE_DOWNLOAD)

  set(BENCHMARK_ENABLE_TESTING OFF)

  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
  )

  list(APPEND NUMERIC_DEPENDENCIES googlebenchmark)

endif()
