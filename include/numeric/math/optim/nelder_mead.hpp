#ifndef NUMERIC_MATH_OPTIM_NELDER_MEAD_HPP_
#define NUMERIC_MATH_OPTIM_NELDER_MEAD_HPP_

#include <algorithm>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>

namespace numeric::math::optim {

template <typename T> class NelderMead {
public:
  NelderMead() : alpha_(1), gamma_(2), rho_(0.5), sigma_(0.5) {}

  T alpha() const { return alpha_; }
  NelderMead &alpha(T val) {
    alpha_ = val;
    return *this;
  }

  T gamma() const { return gamma_; }
  NelderMead &gamma(T val) {
    gamma_ = val;
    return *this;
  }

  T rho() const { return rho_; }
  NelderMead &rho(T val) {
    rho_ = val;
    return *this;
  }

  T sigma() const { return sigma_; }
  NelderMead &sigma(T val) {
    sigma_ = val;
    return *this;
  }

  memory::ArrayConstView<T, 1> x() const { return x0_; }

  template <typename F, typename TF>
  NelderMead &minimize(F &&f, const memory::ArrayConstView<T, 1> &x0,
                       TF &&terminate, T size = 1) {
    x0_ = memory::Array<T, 1>(x0.shape(), memory::MemoryType::HOST);
    x0_ = x0;
    xr_ = memory::Array<T, 1>(x0.shape(), memory::MemoryType::HOST);
    xe_ = memory::Array<T, 1>(x0.shape(), memory::MemoryType::HOST);
    xc_ = memory::Array<T, 1>(x0.shape(), memory::MemoryType::HOST);
    init_simplex(size);
    init_f_vals(f);

    while (!terminate(x0_, simplex_)) {
      step(f);
    }

    return *this;
  }

private:
  T alpha_;
  T gamma_;
  T rho_;
  T sigma_;
  memory::Array<T, 1> x0_;
  memory::Array<T, 1> xr_;
  memory::Array<T, 1> xe_;
  memory::Array<T, 1> xc_;
  memory::Array<T, 2> simplex_;
  memory::Array<T, 1> f_vals_;
  memory::Array<dim_t, 1> f_vals_order_;

  template <typename F> void step(F &&f) {
    const dim_t N = x0_.shape(0);
    // Order the function evaluationos
    std::stable_sort(f_vals_order_.raw(), f_vals_order_.raw() + N + 1,
                     [&](dim_t i, dim_t j) { return f_vals_(i) < f_vals_(j); });
    const dim_t ix1 = f_vals_order_(0);
    const dim_t ixN = f_vals_order_(N - 1);
    const dim_t ixNp1 = f_vals_order_(N);
    // Compute Centroid x0_
    x0_ = 0;
    for (dim_t i = 0; i < N; ++i) {
      x0_ += simplex_(f_vals_order_(i)) / N;
    }
    // Compute Reflection xr_
    xr_ = x0_ + alpha_ * (x0_ - simplex_(ixNp1));
    const T fr = f(xr_);

    // Reflect
    if (f_vals_(ix1) <= fr && fr < f_vals_(ixN)) {
      simplex_(ixNp1) = xr_;
      f_vals_(ixNp1) = fr;
      return;
    }

    // Expand
    if (fr < f_vals_(ix1)) {
      xe_ = x0_ + gamma_ * (xr_ - x0_);
      const T fe = f(xe_);
      if (fe < fr) {
        simplex_(ixNp1) = xe_;
        f_vals_(ixNp1) = fe;
        return;
      } else {
        simplex_(ixNp1) = xr_;
        f_vals_(ixNp1) = fr;
        return;
      }
    }

    // Contract
    if (fr < f_vals_(ixNp1)) {
      xc_ = x0_ + rho_ * (xr_ - x0_);
      const T fc = f(xc_);
      if (fc < fr) {
        simplex_(ixNp1) = xc_;
        f_vals_(ixNp1) = fc;
        return;
      }
    } else {
      xc_ = x0_ + rho_ * (simplex_(ixNp1) - x0_);
      const T fc = f(xc_);
      if (fc < fr) {
        simplex_(ixNp1) = xc_;
        f_vals_(ixNp1) = fc;
        return;
      }
    }

    // Shrink
    for (dim_t i = 0; i < N; ++i) {
      const dim_t ixi = f_vals_order_(i);
      simplex_(ixi) = simplex_(ix1) + sigma_ * (simplex_(ixi) - simplex_(ix1));
    }
  }

  void init_simplex(T size = 1) {
    const dim_t N = x0_.shape(0);
    simplex_ = memory::Array<T, 2>(memory::Shape<2>(N + 1, N),
                                   memory::MemoryType::HOST);
    simplex_(0) = x0_ - N * size / (N + 1);
    for (dim_t i = 0; i < N; ++i) {
      simplex_(i + 1) = simplex_(0);
      simplex_(i + 1, i) += size;
    }
  }

  template <typename F> void init_f_vals(F &&f) {
    const dim_t N = x0_.shape(0);
    f_vals_ =
        memory::Array<T, 1>(memory::Shape<1>(N + 1), memory::MemoryType::HOST);
    f_vals_order_ = memory::Array<dim_t, 1>(memory::Shape<1>(N + 1),
                                            memory::MemoryType::HOST);
    for (dim_t i = 0; i < N + 1; ++i) {
      f_vals_(i) = f(simplex_(i));
      f_vals_order_(i) = i;
    }
  }
};

} // namespace numeric::math::optim

#endif
