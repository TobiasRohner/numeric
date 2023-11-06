#include <cstdio>
#include <filesystem>
#include <gtest/gtest.h>
#include <numeric/io/netcdf_file.hpp>
#include <numeric/io/netcdf_variable.hpp>
#include <numeric/memory/array.hpp>
#include <string>
#include <string_view>

static std::filesystem::path get_temp_file(const std::string &name = "") {
  const ::testing::TestInfo *info =
      ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string suite = info->test_suite_name();
  const std::string test = info->name();
  std::string filename = "numeric_test_" + suite + "_" + test;
  if (name == "") {
    filename += ".nc";
  } else {
    filename += "_" + name;
  }
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();
  const std::filesystem::path path = tmp / filename;
  return path;
}

static std::shared_ptr<numeric::io::NetCDFFile> open_test_file() {
  std::filesystem::path path = numeric::DATA_DIR;
  path /= "test/test_file.nc";
  auto file = numeric::io::NetCDFFile::open(path.c_str());
  return file;
}

TEST(netcdf, create) {
  const std::filesystem::path filename = get_temp_file();
  const auto file = numeric::io::NetCDFFile::create(filename.c_str());
  ASSERT_NE(file, nullptr);
}

TEST(netcdf, create_group) {
  const std::filesystem::path filename = get_temp_file();
  auto file = numeric::io::NetCDFFile::create(filename.c_str());
  const auto group = file->create_group("group_1");
  ASSERT_NE(group, nullptr);
}

TEST(netcdf, create_variable) {
  const std::filesystem::path filename = get_temp_file();
  auto file = numeric::io::NetCDFFile::create(filename.c_str());
  const auto var = file->create_variable<double>(
      "var", numeric::memory::Layout<4>(3, 4, 5, 6));
  ASSERT_NE(var, nullptr);
}

TEST(netcdf, write_variable) {
  numeric::memory::Layout<4> shape(3, 4, 5, 6);
  numeric::memory::Array<double, 4> arr(shape);
  for (numeric::dim_t i = 0; i < arr.size(); ++i) {
    arr.raw()[i] = i;
  }
  const std::filesystem::path filename = get_temp_file();
  auto file = numeric::io::NetCDFFile::create(filename.c_str());
  file->write("var", arr);
}

TEST(netcdf, open) {
  const auto file = open_test_file();
  ASSERT_NE(file, nullptr);
}

TEST(netcdf, open_group) {
  const auto file = open_test_file();
  const auto group = file->open_group("grp_a");
  ASSERT_NE(group, nullptr);
}

TEST(netcdf, read_variable) {
  const auto file = open_test_file();
  const auto grp_a = file->open_group("grp_a");
  const numeric::memory::Array<float, 2> arr_float =
      grp_a->read<float, 2>("var_float");
  const numeric::memory::Array<int32_t, 1> arr_int =
      grp_a->read<int32_t, 1>("var_int");
}
