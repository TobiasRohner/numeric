#include <gtest/gtest.h>
#include <numeric/memory/linspace.hpp>

TEST(linspace, elements) {
  numeric::memory::Linspace<double> l1(0, 10, 11);
  ASSERT_EQ(l1.shape(0), 11);
  for (double i = 0; i <= 10; ++i) {
    ASSERT_EQ(l1(i), i);
  }
  numeric::memory::Linspace<double> l2(0, 10, 10, false);
  ASSERT_EQ(l2.shape(0), 10);
  for (double i = 0; i < 10; ++i) {
    ASSERT_EQ(l1(i), i);
  }
}

TEST(linspace, slice_start) {
  numeric::memory::Linspace<double> ls(0, 10, 11);
  const auto lss = ls(numeric::memory::Slice(0, 5));
  ASSERT_EQ(lss.shape(0), 5);
  for (numeric::dim_t i = 0; i < 5; ++i) {
    ASSERT_EQ(lss(i), i);
  }
}

TEST(linspace, slice_end) {
  numeric::memory::Linspace<double> ls(0, 10, 11);
  const auto lss = ls(numeric::memory::Slice(5, -1));
  ASSERT_EQ(lss.shape(0), 6);
  for (numeric::dim_t i = 0; i < 6; ++i) {
    ASSERT_EQ(lss(i), i + 5);
  }
}

TEST(linspace, slice_middle) {
  numeric::memory::Linspace<double> ls(0, 10, 11);
  const auto lss = ls(numeric::memory::Slice(3, 7));
  ASSERT_EQ(lss.shape(0), 4);
  for (numeric::dim_t i = 0; i < 4; ++i) {
    ASSERT_EQ(lss(i), i + 3);
  }
}
