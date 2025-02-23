#include <gtest/gtest.h>
#include <numeric/memory/slice.hpp>

TEST(slice, size_size) {
  const numeric::memory::Slice slice0(0, -1, 3);
  ASSERT_EQ(slice0.size(30), 10);
  const numeric::memory::Slice slice1(1, -1, 3);
  ASSERT_EQ(slice1.size(30), 10);
  const numeric::memory::Slice slice2(2, -1, 3);
  ASSERT_EQ(slice2.size(30), 10);
};
