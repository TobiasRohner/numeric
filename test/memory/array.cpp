#include <gtest/gtest.h>
#include <numeric/memory/array.hpp>

TEST(numeric_array, row_major) {
  numeric::memory::Layout<3> shape(2, 3, 4);
  numeric::memory::Array<int, 3> array(shape);
  for (size_t i = 0; i < array.size(); ++i) {
    array.raw()[i] = i;
  }
  for (size_t i = 0; i < array.shape(0); ++i) {
    for (size_t j = 0; j < array.shape(1); ++j) {
      for (size_t k = 0; k < array.shape(2); ++k) {
        const int expected = k + array.shape(2) * (j + array.shape(1) * i);
        const int gotten = array(i, j, k);
        ASSERT_EQ(expected, gotten);
      }
    }
  }
}

TEST(numeric_array, decay_to_view) {
  numeric::memory::Layout<3> shape(2, 3, 4);
  numeric::memory::Array<int, 3> array(shape);
  for (size_t i = 0; i < array.size(); ++i) {
    array.raw()[i] = i;
  }
  const auto check = [](numeric::memory::ArrayView<int, 3> view) {
    for (size_t i = 0; i < view.shape(0); ++i) {
      for (size_t j = 0; j < view.shape(1); ++j) {
        for (size_t k = 0; k < view.shape(2); ++k) {
          const int expected = k + view.shape(2) * (j + view.shape(1) * i);
          const int gotten = view(i, j, k);
          ASSERT_EQ(expected, gotten);
        }
      }
    }
  };
  check(array);
}

TEST(numeric_array, decay_to_const_view) {
  numeric::memory::Layout<3> shape(2, 3, 4);
  numeric::memory::Array<int, 3> array(shape);
  for (size_t i = 0; i < array.size(); ++i) {
    array.raw()[i] = i;
  }
  const auto check = [](numeric::memory::ArrayConstView<int, 3> view) {
    for (size_t i = 0; i < view.shape(0); ++i) {
      for (size_t j = 0; j < view.shape(1); ++j) {
        for (size_t k = 0; k < view.shape(2); ++k) {
          const int expected = k + view.shape(2) * (j + view.shape(1) * i);
          const int gotten = view(i, j, k);
          ASSERT_EQ(expected, gotten);
        }
      }
    }
  };
  check(array);
}

TEST(numeric_array, default_construct) {
  numeric::memory::Array<double, 3> arr;
  ASSERT_EQ(arr.memory_type(), numeric::memory::MemoryType::UNKNOWN);
  ASSERT_EQ(arr.shape(0), 0);
  ASSERT_EQ(arr.shape(1), 0);
  ASSERT_EQ(arr.shape(2), 0);
  ASSERT_EQ(arr.raw(), nullptr);
}

TEST(numeric_array, default_construct_view) {
  numeric::memory::ArrayView<double, 3> arr;
  ASSERT_EQ(arr.memory_type(), numeric::memory::MemoryType::UNKNOWN);
  ASSERT_EQ(arr.shape(0), 0);
  ASSERT_EQ(arr.shape(1), 0);
  ASSERT_EQ(arr.shape(2), 0);
  ASSERT_EQ(arr.raw(), nullptr);
}

TEST(numeric_array, default_construct_const_view) {
  numeric::memory::ArrayConstView<double, 3> arr;
  ASSERT_EQ(arr.memory_type(), numeric::memory::MemoryType::UNKNOWN);
  ASSERT_EQ(arr.shape(0), 0);
  ASSERT_EQ(arr.shape(1), 0);
  ASSERT_EQ(arr.shape(2), 0);
  ASSERT_EQ(arr.raw(), nullptr);
}

TEST(numeric_array, movable) {
  numeric::memory::Layout<3> shape(3, 4, 5);
  numeric::memory::Array<double, 3> dst;
  numeric::memory::Array<double, 3> src(shape);
  const double *mem = src.raw();
  const auto mem_t = src.memory_type();
  for (size_t i = 0; i < src.size(); ++i) {
    src.raw()[i] = i;
  }
  dst = std::move(src);
  ASSERT_EQ(dst.shape(0), shape.shape(0));
  ASSERT_EQ(dst.shape(1), shape.shape(1));
  ASSERT_EQ(dst.shape(2), shape.shape(2));
  ASSERT_EQ(dst.memory_type(), mem_t);
  ASSERT_EQ(dst.raw(), mem);
  ASSERT_EQ(src.raw(), nullptr);
}
