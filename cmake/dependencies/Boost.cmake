set(BOOST_INCLUDE_LIBRARIES multiprecision math)

FetchContent_Declare(
  Boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG boost-1.85.0
  GIT_PROGRESS TRUE
)

list(APPEND NUMERIC_DEPENDENCIES Boost)
