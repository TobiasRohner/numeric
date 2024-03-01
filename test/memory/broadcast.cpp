#include <gtest/gtest.h>
#include <numeric/memory/array.hpp>
#include <numeric/memory/broadcast.hpp>

TEST(broadcast, additional_dims) {
  numeric::memory::Shape<3> shape(1, 2, 3);
  numeric::memory::Array<double, 3> arr(shape);
  for (numeric::dim_t i = 0; i < arr.size(); ++i) {
    arr.raw()[i] = i;
  }
  numeric::memory::Shape<5> broadcasted_shape(2, 3, 1, 2, 3);
  const numeric::memory::Broadcast<numeric::memory::ArrayConstView<double, 3>,
                                   5>
      broadcasted(arr, broadcasted_shape);
  for (numeric::dim_t i = 0; i < broadcasted_shape[0]; ++i) {
    for (numeric::dim_t j = 0; j < broadcasted_shape[1]; ++j) {
      for (numeric::dim_t k = 0; k < broadcasted_shape[2]; ++k) {
        for (numeric::dim_t l = 0; l < broadcasted_shape[3]; ++l) {
          for (numeric::dim_t m = 0; m < broadcasted_shape[4]; ++m) {
            const double val_orig = arr(k, l, m);
            const double val_brd = broadcasted(i, j, k, l, m);
            ASSERT_EQ(val_orig, val_brd);
          }
        }
      }
    }
  }
}

TEST(broadcast, expand_dims) {
  numeric::memory::Shape<4> shape(4, 5, 1, 3);
  numeric::memory::Array<double, 4> arr(shape);
  for (numeric::dim_t i = 0; i < arr.size(); ++i) {
    arr.raw()[i] = i;
  }
  numeric::memory::Shape<4> broadcasted_shape(4, 5, 6, 3);
  const numeric::memory::Broadcast<numeric::memory::ArrayConstView<double, 4>,
                                   4>
      broadcasted(arr, broadcasted_shape);
  for (numeric::dim_t i = 0; i < broadcasted_shape[0]; ++i) {
    for (numeric::dim_t j = 0; j < broadcasted_shape[1]; ++j) {
      for (numeric::dim_t k = 0; k < broadcasted_shape[2]; ++k) {
        for (numeric::dim_t l = 0; l < broadcasted_shape[3]; ++l) {
          const double val_orig = arr(i, j, 0, l);
          const double val_brd = broadcasted(i, j, k, l);
          ASSERT_EQ(val_orig, val_brd);
        }
      }
    }
  }
}
