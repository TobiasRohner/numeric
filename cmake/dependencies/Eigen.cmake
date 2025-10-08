numeric_register_dependency(Eigen)


if (NUMERIC_Eigen_REQUIRE_DOWNLOAD)

  set(EIGEN_BUILD_DOC OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)

  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )

  list(APPEND NUMERIC_DEPENDENCIES Eigen)

endif()
