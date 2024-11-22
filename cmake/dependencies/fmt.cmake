FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)

list(APPEND NUMERIC_DEPENDENCIES fmt)
