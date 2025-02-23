#include <gtest/gtest.h>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/memory/slice.hpp>

TEST(array, row_major) {
  numeric::memory::Shape<3> shape(2, 3, 4);
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

TEST(array, decay_to_view) {
  numeric::memory::Shape<3> shape(2, 3, 4);
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

TEST(array, decay_to_const_view) {
  numeric::memory::Shape<3> shape(2, 3, 4);
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

TEST(array, default_construct) {
  numeric::memory::Array<double, 3> arr;
  ASSERT_EQ(arr.memory_type(), numeric::memory::MemoryType::UNKNOWN);
  ASSERT_EQ(arr.shape(0), 0);
  ASSERT_EQ(arr.shape(1), 0);
  ASSERT_EQ(arr.shape(2), 0);
  ASSERT_EQ(arr.raw(), nullptr);
}

TEST(array, default_construct_view) {
  numeric::memory::ArrayView<double, 3> arr;
  ASSERT_EQ(arr.memory_type(), numeric::memory::MemoryType::UNKNOWN);
  ASSERT_EQ(arr.shape(0), 0);
  ASSERT_EQ(arr.shape(1), 0);
  ASSERT_EQ(arr.shape(2), 0);
  ASSERT_EQ(arr.raw(), nullptr);
}

TEST(array, default_construct_const_view) {
  numeric::memory::ArrayConstView<double, 3> arr;
  ASSERT_EQ(arr.memory_type(), numeric::memory::MemoryType::UNKNOWN);
  ASSERT_EQ(arr.shape(0), 0);
  ASSERT_EQ(arr.shape(1), 0);
  ASSERT_EQ(arr.shape(2), 0);
  ASSERT_EQ(arr.raw(), nullptr);
}

TEST(array, movable) {
  numeric::memory::Shape<3> shape(3, 4, 5);
  numeric::memory::Array<double, 3> dst;
  numeric::memory::Array<double, 3> src(shape);
  const double *mem = src.raw();
  const auto mem_t = src.memory_type();
  for (size_t i = 0; i < src.size(); ++i) {
    src.raw()[i] = i;
  }
  dst = std::move(src);
  ASSERT_EQ(dst.shape(0), shape[0]);
  ASSERT_EQ(dst.shape(1), shape[1]);
  ASSERT_EQ(dst.shape(2), shape[2]);
  ASSERT_EQ(dst.memory_type(), mem_t);
  ASSERT_EQ(dst.raw(), mem);
  ASSERT_EQ(src.raw(), nullptr);
}

TEST(array, broadcast) {
  numeric::memory::Shape<4> shape(1, 3, 1, 6);
  numeric::memory::Array<double, 4> arr(shape);
  numeric::memory::Layout<4> shape_strided = shape;
  shape_strided.stride(3) = 2;
  numeric::memory::ArrayView<double, 4> strided_view(arr.raw(), shape_strided,
                                                     arr.memory_type());
  for (numeric::dim_t i = 0; i < arr.size(); ++i) {
    arr.raw()[i] = i;
  }
  numeric::memory::Shape<6> broadcasted_shape(2, 3, 4, 3, 2, 3);
  numeric::memory::ArrayView<double, 6> broadcasted_view =
      strided_view.broadcast(broadcasted_shape);
  for (numeric::dim_t i = 0; i < broadcasted_shape[0]; ++i) {
    for (numeric::dim_t j = 0; j < broadcasted_shape[1]; ++j) {
      for (numeric::dim_t k = 0; k < broadcasted_shape[2]; ++k) {
        for (numeric::dim_t l = 0; l < broadcasted_shape[3]; ++l) {
          for (numeric::dim_t m = 0; m < broadcasted_shape[4]; ++m) {
            for (numeric::dim_t n = 0; n < broadcasted_shape[5]; ++n) {
              const double val_orig = strided_view(0, l, 0, n);
              const double val_brd = broadcasted_view(i, j, k, l, m, n);
              ASSERT_EQ(val_brd, val_orig);
            }
          }
        }
      }
    }
  }
}

TEST(array, slice_all) {
  numeric::memory::Shape<3> shape(5, 6, 20);
  numeric::memory::Array<double, 3> arr(shape);
  for (numeric::dim_t i = 0; i < arr.size(); ++i) {
    arr.raw()[i] = i;
  }
  const auto slice_0 =
      arr(1, numeric::memory::Slice(), numeric::memory::Slice());
  ASSERT_EQ(slice_0.dim, 2);
  ASSERT_EQ(slice_0.shape(0), shape[1]);
  ASSERT_EQ(slice_0.shape(1), shape[2]);
  for (numeric::dim_t i = 0; i < slice_0.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_0.shape(1); ++j) {
      const double val_orig = arr(1, i, j);
      const double val_slic = slice_0(i, j);
      ASSERT_EQ(val_slic, val_orig);
    }
  }
  const auto slice_1 =
      arr(numeric::memory::Slice(), 2, numeric::memory::Slice());
  ASSERT_EQ(slice_1.dim, 2);
  ASSERT_EQ(slice_1.shape(0), shape[0]);
  ASSERT_EQ(slice_1.shape(1), shape[2]);
  for (numeric::dim_t i = 0; i < slice_1.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_1.shape(1); ++j) {
      const double val_orig = arr(i, 2, j);
      const double val_slic = slice_1(i, j);
      ASSERT_EQ(val_slic, val_orig);
    }
  }
  const auto slice_2 =
      arr(numeric::memory::Slice(), numeric::memory::Slice(), 3);
  ASSERT_EQ(slice_2.dim, 2);
  ASSERT_EQ(slice_2.shape(0), shape[0]);
  ASSERT_EQ(slice_2.shape(1), shape[1]);
  for (numeric::dim_t i = 0; i < slice_2.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_2.shape(1); ++j) {
      const double val_orig = arr(i, j, 3);
      const double val_slic = slice_2(i, j);
      ASSERT_EQ(val_slic, val_orig);
    }
  }
}

TEST(array, slice_front) {
  numeric::memory::Shape<3> shape(5, 6, 20);
  numeric::memory::Array<double, 3> arr(shape);
  for (numeric::dim_t i = 0; i < arr.size(); ++i) {
    arr.raw()[i] = i;
  }
  const auto slice_0 = arr(numeric::memory::Slice(0, 3),
                           numeric::memory::Slice(), numeric::memory::Slice());
  ASSERT_EQ(slice_0.dim, 3);
  ASSERT_EQ(slice_0.shape(0), 3);
  ASSERT_EQ(slice_0.shape(1), shape[1]);
  ASSERT_EQ(slice_0.shape(2), shape[2]);
  for (numeric::dim_t i = 0; i < slice_0.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_0.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < slice_0.shape(2); ++k) {
        const double val_orig = arr(i, j, k);
        const double val_slic = slice_0(i, j, k);
        ASSERT_EQ(val_slic, val_orig);
      }
    }
  }
  const auto slice_1 =
      arr(numeric::memory::Slice(), numeric::memory::Slice(0, 3),
          numeric::memory::Slice());
  ASSERT_EQ(slice_1.dim, 3);
  ASSERT_EQ(slice_1.shape(0), shape[0]);
  ASSERT_EQ(slice_1.shape(1), 3);
  ASSERT_EQ(slice_1.shape(2), shape[2]);
  for (numeric::dim_t i = 0; i < slice_1.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_1.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < slice_1.shape(2); ++k) {
        const double val_orig = arr(i, j, k);
        const double val_slic = slice_1(i, j, k);
        ASSERT_EQ(val_slic, val_orig);
      }
    }
  }
  const auto slice_2 = arr(numeric::memory::Slice(), numeric::memory::Slice(),
                           numeric::memory::Slice(0, 3));
  ASSERT_EQ(slice_2.dim, 3);
  ASSERT_EQ(slice_2.shape(0), shape[0]);
  ASSERT_EQ(slice_2.shape(1), shape[1]);
  ASSERT_EQ(slice_2.shape(2), 3);
  for (numeric::dim_t i = 0; i < slice_2.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_2.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < slice_2.shape(2); ++k) {
        const double val_orig = arr(i, j, k);
        const double val_slic = slice_2(i, j, k);
        ASSERT_EQ(val_slic, val_orig);
      }
    }
  }
}

TEST(array, slice_back) {
  numeric::memory::Shape<3> shape(5, 6, 20);
  numeric::memory::Array<double, 3> arr(shape);
  for (numeric::dim_t i = 0; i < arr.size(); ++i) {
    arr.raw()[i] = i;
  }
  const auto slice_0 = arr(numeric::memory::Slice(2, -1),
                           numeric::memory::Slice(), numeric::memory::Slice());
  ASSERT_EQ(slice_0.dim, 3);
  ASSERT_EQ(slice_0.shape(0), 3);
  ASSERT_EQ(slice_0.shape(1), shape[1]);
  ASSERT_EQ(slice_0.shape(2), shape[2]);
  for (numeric::dim_t i = 0; i < slice_0.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_0.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < slice_0.shape(2); ++k) {
        const double val_orig = arr(i + 2, j, k);
        const double val_slic = slice_0(i, j, k);
        ASSERT_EQ(val_slic, val_orig);
      }
    }
  }
  const auto slice_1 =
      arr(numeric::memory::Slice(), numeric::memory::Slice(3, -1),
          numeric::memory::Slice());
  ASSERT_EQ(slice_1.dim, 3);
  ASSERT_EQ(slice_1.shape(0), shape[0]);
  ASSERT_EQ(slice_1.shape(1), 3);
  ASSERT_EQ(slice_1.shape(2), shape[2]);
  for (numeric::dim_t i = 0; i < slice_1.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_1.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < slice_1.shape(2); ++k) {
        const double val_orig = arr(i, j + 3, k);
        const double val_slic = slice_1(i, j, k);
        ASSERT_EQ(val_slic, val_orig);
      }
    }
  }
  const auto slice_2 = arr(numeric::memory::Slice(), numeric::memory::Slice(),
                           numeric::memory::Slice(17, -1));
  ASSERT_EQ(slice_2.dim, 3);
  ASSERT_EQ(slice_2.shape(0), shape[0]);
  ASSERT_EQ(slice_2.shape(1), shape[1]);
  ASSERT_EQ(slice_2.shape(2), 3);
  for (numeric::dim_t i = 0; i < slice_2.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_2.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < slice_2.shape(2); ++k) {
        const double val_orig = arr(i, j, k + 17);
        const double val_slic = slice_2(i, j, k);
        ASSERT_EQ(val_slic, val_orig);
      }
    }
  }
}

TEST(array, stride) {
  numeric::memory::Shape<3> shape(5, 6, 20);
  numeric::memory::Array<double, 3> arr(shape);
  for (numeric::dim_t i = 0; i < arr.size(); ++i) {
    arr.raw()[i] = i;
  }
  const auto slice_0 = arr(numeric::memory::Slice(0, -1, 2),
                           numeric::memory::Slice(), numeric::memory::Slice());
  ASSERT_EQ(slice_0.dim, 3);
  ASSERT_EQ(slice_0.shape(0), 3);
  ASSERT_EQ(slice_0.shape(1), shape[1]);
  ASSERT_EQ(slice_0.shape(2), shape[2]);
  for (numeric::dim_t i = 0; i < slice_0.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_0.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < slice_0.shape(2); ++k) {
        const double val_orig = arr(2 * i, j, k);
        const double val_slic = slice_0(i, j, k);
        ASSERT_EQ(val_slic, val_orig);
      }
    }
  }
  const auto slice_1 =
      arr(numeric::memory::Slice(), numeric::memory::Slice(0, -1, 2),
          numeric::memory::Slice());
  ASSERT_EQ(slice_1.dim, 3);
  ASSERT_EQ(slice_1.shape(0), shape[0]);
  ASSERT_EQ(slice_1.shape(1), 3);
  ASSERT_EQ(slice_1.shape(2), shape[2]);
  for (numeric::dim_t i = 0; i < slice_1.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_1.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < slice_1.shape(2); ++k) {
        const double val_orig = arr(i, 2 * j, k);
        const double val_slic = slice_1(i, j, k);
        ASSERT_EQ(val_slic, val_orig);
      }
    }
  }
  const auto slice_2 = arr(numeric::memory::Slice(), numeric::memory::Slice(),
                           numeric::memory::Slice(0, -1, 2));
  ASSERT_EQ(slice_2.dim, 3);
  ASSERT_EQ(slice_2.shape(0), shape[0]);
  ASSERT_EQ(slice_2.shape(1), shape[1]);
  ASSERT_EQ(slice_2.shape(2), 10);
  for (numeric::dim_t i = 0; i < slice_2.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < slice_2.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < slice_2.shape(2); ++k) {
        const double val_orig = arr(i, j, 2 * k);
        const double val_slic = slice_2(i, j, k);
        ASSERT_EQ(val_slic, val_orig);
      }
    }
  }
}
