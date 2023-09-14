#include <gtest/gtest.h>
#include <numeric/math/array_op.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/copy.hpp>
#include <numeric/memory/slice.hpp>

TEST(copy, array_to_array) {
  numeric::memory::Layout<3> shape(2, 3, 4);
  numeric::memory::Array<int, 3> a(shape);
  numeric::memory::Array<int, 3> b(shape);
  for (size_t i = 0; i < a.size(); ++i) {
    a.raw()[i] = i;
  }
  b = a;
  for (size_t i = 0; i < b.shape(0); ++i) {
    for (size_t j = 0; j < b.shape(1); ++j) {
      for (size_t k = 0; k < b.shape(2); ++k) {
        const int expected = a(i, j, k);
        const int gotten = b(i, j, k);
        ASSERT_EQ(expected, gotten);
      }
    }
  }
}

TEST(copy, array_to_array_broadcast) {
  numeric::memory::Layout<2> shape_a(3, 1);
  numeric::memory::Layout<3> shape_b(2, 3, 4);
  numeric::memory::Array<int, 2> a(shape_a);
  numeric::memory::Array<int, 3> b(shape_b);
  for (size_t i = 0; i < a.size(); ++i) {
    a.raw()[i] = i;
  }
  b = a;
  for (size_t i = 0; i < b.shape(0); ++i) {
    for (size_t j = 0; j < b.shape(1); ++j) {
      for (size_t k = 0; k < b.shape(2); ++k) {
        const int expected = a(j);
        const int gotten = b(i, j, k);
        ASSERT_EQ(expected, gotten);
      }
    }
  }
}

TEST(copy, array_to_slice) {
  numeric::memory::Layout<3> shape_a(2, 3, 4);
  numeric::memory::Layout<4> shape_b(5, 2, 3, 4);
  numeric::memory::Array<int, 3> a(shape_a);
  numeric::memory::Array<int, 4> b(shape_b);
  for (size_t i = 0; i < a.size(); ++i) {
    a.raw()[i] = i;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b.raw()[i] = 0;
  }
  using sl = numeric::memory::Slice;
  b(sl(0, 3), sl(), sl(), sl()) = a;
  for (size_t i = 0; i < b.shape(0); ++i) {
    for (size_t j = 0; j < b.shape(1); ++j) {
      for (size_t k = 0; k < b.shape(2); ++k) {
        for (size_t l = 0; l < b.shape(3); ++l) {
          const int expected = i < 3 ? a(j, k, l) : 0;
          const int gotten = b(i, j, k, l);
          ASSERT_EQ(expected, gotten);
        }
      }
    }
  }
}

TEST(copy, expr_to_array) {
  numeric::memory::Layout<4> shape_a(5, 2, 3, 4);
  numeric::memory::Layout<3> shape_b(2, 3, 4);
  numeric::memory::Layout<3> shape_c(2, 1, 4);
  numeric::memory::Array<int, 4> a(shape_a);
  numeric::memory::Array<int, 3> b(shape_b);
  numeric::memory::Array<int, 3> c(shape_c);
  for (size_t i = 0; i < a.size(); ++i) {
    a.raw()[i] = 0;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b.raw()[i] = i;
  }
  for (size_t i = 0; i < c.size(); ++i) {
    c.raw()[i] = i;
  }
  a = b + c;
  for (size_t i = 0; i < a.shape(0); ++i) {
    for (size_t j = 0; j < a.shape(1); ++j) {
      for (size_t k = 0; k < a.shape(2); ++k) {
        for (size_t l = 0; l < a.shape(3); ++l) {
          const int expected = b(j, k, l) + c(j, 0, l);
          const int gotten = a(i, j, k, l);
          ASSERT_EQ(expected, gotten);
        }
      }
    }
  }
}
