#include <cstdio>
#include <gtest/gtest.h>
#include <numeric/io/netcdf_file.hpp>
#include <numeric/io/netcdf_variable.hpp>
#include <numeric/memory/array.hpp>
#include <string_view>

TEST(netcdf, create) {
  char filename[L_tmpnam];
  std::tmpnam(filename);
  const auto file = numeric::io::NetCDFFile::create(filename);
  ASSERT_NE(file, nullptr);
}
