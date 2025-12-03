#include <gtest/gtest.h>
#include <numeric/math/functions.hpp>
#include <numeric/math/reduce.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/meshgrid.hpp>
#include <random>

TEST(sum, host_1d_contiguous) {
  for (numeric::dim_t size = 1; size < 1ll << 20; size <<= 1) {
    numeric::memory::Shape<1> shape(size);
    numeric::memory::Array<double, 1> data(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(0, 1);
    double total = 0;
    for (numeric::dim_t i = 0; i < size; ++i) {
      const double val = dist(rng);
      data(i) = val;
      total += val;
    }
    const double sum = numeric::math::sum(data);
    ASSERT_NEAR(sum, total, 1e-8);
  }
}

TEST(sum, host_nd_contiguous) {
  numeric::memory::Shape<5> shape(2, 3, 4, 5, 6);
  numeric::memory::Array<double, 5> data(shape);
  std::mt19937 rng;
  std::uniform_real_distribution<double> dist(0, 1);
  double total = 0;
  for (numeric::dim_t i = 0; i < 2; ++i) {
    for (numeric::dim_t j = 0; j < 3; ++j) {
      for (numeric::dim_t k = 0; k < 4; ++k) {
        for (numeric::dim_t l = 0; l < 5; ++l) {
          for (numeric::dim_t m = 0; m < 6; ++m) {
            const double val = dist(rng);
            data(i, j, k, l, m) = val;
            total += val;
          }
        }
      }
    }
  }
  const double sum = numeric::math::sum(data);
  ASSERT_NEAR(sum, total, 1e-8);
}

TEST(sum, host_expr) {
  numeric::memory::Linspace<double> x(1, 10, 10);
  const auto [X, Y] = numeric::memory::meshgrid(x, x);
  const double val = numeric::math::sum(X + Y);
  ASSERT_NEAR(val, 1100, 1e-8);
}

TEST(min, host_1d_contiguous) {
  for (numeric::dim_t size = 1; size < 1ll << 20; size <<= 1) {
    numeric::memory::Shape<1> shape(size);
    numeric::memory::Array<double, 1> data(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(0, 1);
    double total = 1;
    for (numeric::dim_t i = 0; i < size; ++i) {
      const double val = dist(rng);
      data(i) = val;
      total = numeric::math::min(total, val);
    }
    const double min = numeric::math::min(data);
    ASSERT_NEAR(min, total, 1e-8);
  }
}

TEST(min, host_nd_contiguous) {
  numeric::memory::Shape<5> shape(2, 3, 4, 5, 6);
  numeric::memory::Array<double, 5> data(shape);
  std::mt19937 rng;
  std::uniform_real_distribution<double> dist(0, 1);
  double total = 1;
  for (numeric::dim_t i = 0; i < 2; ++i) {
    for (numeric::dim_t j = 0; j < 3; ++j) {
      for (numeric::dim_t k = 0; k < 4; ++k) {
        for (numeric::dim_t l = 0; l < 5; ++l) {
          for (numeric::dim_t m = 0; m < 6; ++m) {
            const double val = dist(rng);
            data(i, j, k, l, m) = val;
            total = numeric::math::min(total, val);
          }
        }
      }
    }
  }
  const double min = numeric::math::min(data);
  ASSERT_NEAR(min, total, 1e-8);
}

TEST(min, host_expr) {
  numeric::memory::Linspace<double> x(1, 10, 10);
  const auto [X, Y] = numeric::memory::meshgrid(x, x);
  const double val = numeric::math::min(X + Y);
  ASSERT_NEAR(val, 2, 1e-8);
}

TEST(max, host_1d_contiguous) {
  for (numeric::dim_t size = 1; size < 1ll << 20; size <<= 1) {
    numeric::memory::Shape<1> shape(size);
    numeric::memory::Array<double, 1> data(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(0, 1);
    double total = 0;
    for (numeric::dim_t i = 0; i < size; ++i) {
      const double val = dist(rng);
      data(i) = val;
      total = numeric::math::max(total, val);
    }
    const double max = numeric::math::max(data);
    ASSERT_NEAR(max, total, 1e-8);
  }
}

TEST(max, host_nd_contiguous) {
  numeric::memory::Shape<5> shape(2, 3, 4, 5, 6);
  numeric::memory::Array<double, 5> data(shape);
  std::mt19937 rng;
  std::uniform_real_distribution<double> dist(0, 1);
  double total = 0;
  for (numeric::dim_t i = 0; i < 2; ++i) {
    for (numeric::dim_t j = 0; j < 3; ++j) {
      for (numeric::dim_t k = 0; k < 4; ++k) {
        for (numeric::dim_t l = 0; l < 5; ++l) {
          for (numeric::dim_t m = 0; m < 6; ++m) {
            const double val = dist(rng);
            data(i, j, k, l, m) = val;
            total = numeric::math::max(total, val);
          }
        }
      }
    }
  }
  const double max = numeric::math::max(data);
  ASSERT_NEAR(max, total, 1e-8);
}

TEST(max, host_expr) {
  numeric::memory::Linspace<double> x(1, 10, 10);
  const auto [X, Y] = numeric::memory::meshgrid(x, x);
  const double val = numeric::math::max(X + Y);
  ASSERT_NEAR(val, 20, 1e-8);
}

#if NUMERIC_ENABLE_HIP
TEST(sum, device_1d_contiguous) {
  for (numeric::dim_t size = 1; size < 1ll << 20; size <<= 1) {
    numeric::memory::Shape<1> shape(size);
    numeric::memory::Array<double, 1> data(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(0, 1);
    double total = 0;
    for (numeric::dim_t i = 0; i < size; ++i) {
      const double val = dist(rng);
      data(i) = val;
      total += val;
    }
    data.to(numeric::memory::MemoryType::DEVICE);
    const double sum = numeric::math::sum(data);
    ASSERT_NEAR(sum, total, 1e-8);
  }
}

TEST(sum, device_nd_contiguous) {
  numeric::memory::Shape<5> shape(2, 3, 4, 5, 6);
  numeric::memory::Array<double, 5> data(shape);
  std::mt19937 rng;
  std::uniform_real_distribution<double> dist(0, 1);
  double total = 0;
  for (numeric::dim_t i = 0; i < 2; ++i) {
    for (numeric::dim_t j = 0; j < 3; ++j) {
      for (numeric::dim_t k = 0; k < 4; ++k) {
        for (numeric::dim_t l = 0; l < 5; ++l) {
          for (numeric::dim_t m = 0; m < 6; ++m) {
            const double val = dist(rng);
            data(i, j, k, l, m) = val;
            total += val;
          }
        }
      }
    }
  }
  data.to(numeric::memory::MemoryType::DEVICE);
  std::cout << data.layout() << std::endl;
  const double sum = numeric::math::sum(data);
  ASSERT_NEAR(sum, total, 1e-8);
}

TEST(sum, device_expr) {
  numeric::memory::Linspace<double> x(1, 10, 10, true,
                                      numeric::memory::MemoryType::DEVICE);
  const auto [X, Y] = numeric::memory::meshgrid(x, x);
  const double val = numeric::math::sum(X + Y);
  ASSERT_NEAR(val, 1100, 1e-8);
}

TEST(min, device_1d_contiguous) {
  for (numeric::dim_t size = 1; size < 1ll << 20; size <<= 1) {
    numeric::memory::Shape<1> shape(size);
    numeric::memory::Array<double, 1> data(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(0, 1);
    double total = 1;
    for (numeric::dim_t i = 0; i < size; ++i) {
      const double val = dist(rng);
      data(i) = val;
      total = numeric::math::min(total, val);
    }
    data.to(numeric::memory::MemoryType::DEVICE);
    const double min = numeric::math::min(data);
    ASSERT_NEAR(min, total, 1e-8);
  }
}

TEST(min, device_nd_contiguous) {
  numeric::memory::Shape<5> shape(2, 3, 4, 5, 6);
  numeric::memory::Array<double, 5> data(shape);
  std::mt19937 rng;
  std::uniform_real_distribution<double> dist(0, 1);
  double total = 1;
  for (numeric::dim_t i = 0; i < 2; ++i) {
    for (numeric::dim_t j = 0; j < 3; ++j) {
      for (numeric::dim_t k = 0; k < 4; ++k) {
        for (numeric::dim_t l = 0; l < 5; ++l) {
          for (numeric::dim_t m = 0; m < 6; ++m) {
            const double val = dist(rng);
            data(i, j, k, l, m) = val;
            total = numeric::math::min(total, val);
          }
        }
      }
    }
  }
  data.to(numeric::memory::MemoryType::DEVICE);
  const double min = numeric::math::min(data);
  ASSERT_NEAR(min, total, 1e-8);
}

TEST(min, device_expr) {
  numeric::memory::Linspace<double> x(1, 10, 10, true,
                                      numeric::memory::MemoryType::DEVICE);
  const auto [X, Y] = numeric::memory::meshgrid(x, x);
  const double val = numeric::math::min(X + Y);
  ASSERT_NEAR(val, 2, 1e-8);
}

TEST(max, device_1d_contiguous) {
  for (numeric::dim_t size = 1; size < 1ll << 20; size <<= 1) {
    numeric::memory::Shape<1> shape(size);
    numeric::memory::Array<double, 1> data(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(0, 1);
    double total = 0;
    for (numeric::dim_t i = 0; i < size; ++i) {
      const double val = dist(rng);
      data(i) = val;
      total = numeric::math::max(total, val);
    }
    data.to(numeric::memory::MemoryType::DEVICE);
    const double max = numeric::math::max(data);
    ASSERT_NEAR(max, total, 1e-8);
  }
}

TEST(max, device_nd_contiguous) {
  numeric::memory::Shape<5> shape(2, 3, 4, 5, 6);
  numeric::memory::Array<double, 5> data(shape);
  std::mt19937 rng;
  std::uniform_real_distribution<double> dist(0, 1);
  double total = 0;
  for (numeric::dim_t i = 0; i < 2; ++i) {
    for (numeric::dim_t j = 0; j < 3; ++j) {
      for (numeric::dim_t k = 0; k < 4; ++k) {
        for (numeric::dim_t l = 0; l < 5; ++l) {
          for (numeric::dim_t m = 0; m < 6; ++m) {
            const double val = dist(rng);
            data(i, j, k, l, m) = val;
            total = numeric::math::max(total, val);
          }
        }
      }
    }
  }
  data.to(numeric::memory::MemoryType::DEVICE);
  const double max = numeric::math::max(data);
  ASSERT_NEAR(max, total, 1e-8);
}

TEST(max, device_expr) {
  numeric::memory::Linspace<double> x(1, 10, 10, true,
                                      numeric::memory::MemoryType::DEVICE);
  const auto [X, Y] = numeric::memory::meshgrid(x, x);
  const double val = numeric::math::max(X + Y);
  ASSERT_NEAR(val, 20, 1e-8);
}
#endif
