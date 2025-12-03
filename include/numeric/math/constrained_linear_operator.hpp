#ifndef NUMERIC_MATH_CONSTRAINED_LINEAR_OPERATOR_HPP_
#define NUMERIC_MATH_CONSTRAINED_LINEAR_OPERATOR_HPP_

#include <numeric/math/linear_operator.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/hip/kernel.hpp>
#include <numeric/utils/type_name.hpp>
#include <string_view>
#endif

namespace numeric::math {

namespace internal {

#if NUMERIC_ENABLE_HIP
hip::Kernel build_kernel_clear_constrained_impl(std::string_view scalar);
hip::Kernel build_kernel_copy_constrained_impl(std::string_view scalar);

template <typename Scalar> hip::Kernel build_kernel_clear_constrained() {
  return build_kernel_clear_constrained_impl(utils::type_name<Scalar>());
}

template <typename Scalar> hip::Kernel build_kernel_copy_constrained() {
  return build_kernel_copy_constrained_impl(utils::type_name<Scalar>());
}
#endif

} // namespace internal

template <typename Scalar>
class ConstrainedLinearOperator : public LinearOperator<Scalar> {
  using super = LinearOperator<Scalar>;

public:
  using scalar_t = Scalar;
  using op_t = LinearOperator<scalar_t>;

  ConstrainedLinearOperator(
      const std::shared_ptr<op_t> &A,
      const memory::ArrayConstView<dim_t, 1> &constrained_dofs)
      : A_(A), constrained_dofs_(constrained_dofs.shape(), A->memory_type()),
        work_(A->shape(0), A->memory_type()) {
    constrained_dofs_ = constrained_dofs;
  }
  ConstrainedLinearOperator(const ConstrainedLinearOperator &) = default;
  ConstrainedLinearOperator(ConstrainedLinearOperator &&) = default;
  virtual ~ConstrainedLinearOperator() override = default;
  ConstrainedLinearOperator &
  operator=(const ConstrainedLinearOperator &) = default;
  ConstrainedLinearOperator &operator=(ConstrainedLinearOperator &&) = default;

  virtual void to(memory::MemoryType memory_type) override {
    A_->to(memory_type);
    constrained_dofs_.to(memory_type);
    work_.to(memory_type);
  }

  virtual memory::MemoryType memory_type() const override {
    return A_->memory_type();
  }

  virtual memory::Shape<2> shape() const override { return A_->shape(); }

  virtual dim_t shape(dim_t i) const override { return A_->shape(i); }

  virtual void operator()(const memory::ArrayConstView<scalar_t, 1> &u,
                          memory::ArrayView<scalar_t, 1> out) const override {
    work_ = u;
    clear_constrained(work_);
    A_->operator()(work_, out);
    copy_constrained(out, u);
  }

  virtual memory::Array<scalar_t, 1>
  operator()(const memory::ArrayConstView<scalar_t, 1> &u) const override {
    memory::Array<scalar_t, 1> out(shape(0), u.memory_type());
    this->operator()(u, out);
    return out;
  }

  void clear_constrained(memory::ArrayView<scalar_t, 1> &x) const {
    if (is_host_accessible(x.memory_type())) {
      clear_constrained_host(x);
    } else if (is_device_accessible(x.memory_type())) {
      clear_constrained_device(x);
    } else {
      NUMERIC_ERROR("Unknown memory type \"{}\"", to_string(x.memory_type()));
    }
  }

  void copy_constrained(memory::ArrayView<scalar_t, 1> &x,
                        const memory::ArrayConstView<scalar_t, 1> &vals) const {
    if (is_host_accessible(x.memory_type())) {
      copy_constrained_host(x, vals);
    } else if (is_device_accessible(x.memory_type())) {
      copy_constrained_device(x, vals);
    } else {
      NUMERIC_ERROR("Unknown memory type \"{}\"", to_string(x.memory_type()));
    }
  }

private:
  std::shared_ptr<op_t> A_;
  memory::Array<dim_t, 1> constrained_dofs_;
  mutable memory::Array<scalar_t, 1> work_;

  void clear_constrained_host(memory::ArrayView<scalar_t, 1> &x) const {
    for (dim_t i = 0; i < constrained_dofs_.size(); ++i) {
      x(constrained_dofs_(i)) = 0;
    }
  }

  void
  copy_constrained_host(memory::ArrayView<scalar_t, 1> &x,
                        const memory::ArrayConstView<scalar_t, 1> &vals) const {
    for (dim_t i = 0; i < constrained_dofs_.size(); ++i) {
      x(constrained_dofs_(i)) = vals(constrained_dofs_(i));
    }
  }

  void clear_constrained_device(memory::ArrayView<scalar_t, 1> &x) const {
    static const hip::Kernel kernel =
        internal::build_kernel_clear_constrained<scalar_t>();
    hip::Device device;
    const hip::LaunchParams lp =
        device.launch_params_for_grid(constrained_dofs_.shape(0), 1, 1);
    kernel(lp, hip::Stream(device), constrained_dofs_, x);
  }

  void copy_constrained_device(
      memory::ArrayView<scalar_t, 1> &x,
      const memory::ArrayConstView<scalar_t, 1> &vals) const {
    static const hip::Kernel kernel =
        internal::build_kernel_copy_constrained<scalar_t>();
    hip::Device device;
    const hip::LaunchParams lp =
        device.launch_params_for_grid(constrained_dofs_.shape(0), 1, 1);
    kernel(lp, hip::Stream(device), constrained_dofs_, x, vals);
  }
};

} // namespace numeric::math

#endif
