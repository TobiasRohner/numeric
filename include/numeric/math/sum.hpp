#ifndef NUMERIC_MATH_SUM_HPP_
#define NUMERIC_MATH_SUM_HPP_

#include <memory>
#include <numeric/config.hpp>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::math {

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t sum_host(const Src &src) {
  typename memory::ArrayTraits<Src>::scalar_t value = 0;
  for (dim_t i = 0; i < src.shape(0); ++i) {
    if constexpr (memory::ArrayTraits<Src>::dim > 1) {
      value += sum_host(src(i));
    } else {
      value += src(i);
    }
  }
  return value;
}

template <typename Src> class SummerImpl {
public:
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;

  SummerImpl() = default;
  virtual ~SummerImpl() = default;

  virtual scalar_t operator()(const memory::ArrayBase<Src> &src) = 0;
};

template <typename Src> class SumHost : public SummerImpl<Src> {
  using super = SummerImpl<Src>;

public:
  using scalar_t = typename super::scalar_t;

  virtual scalar_t operator()(const memory::ArrayBase<Src> &src) override {
    return sum_host(src.derived());
  }
};

template <typename Src> class Summer {
  using impl_t = SummerImpl<Src>;

public:
  using scalar_t = typename impl_t::scalar_t;

  Summer(const std::shared_ptr<impl_t> &impl) : impl_(impl) {}
  Summer(const Src &src) : Summer(make_summer_impl(src)) {}
  Summer(const Summer &) = default;
  Summer(Summer &&) = default;
  ~Summer() = default;
  Summer &operator=(const Summer &) = default;
  Summer &operator=(Summer &&) = default;

  scalar_t operator()(const memory::ArrayBase<Src> &src) {
    return impl_->operator()(src);
  }

private:
  std::shared_ptr<impl_t> impl_;

  static std::shared_ptr<impl_t> make_summer_impl(const Src &src) {
    if (is_host_accessible(src.memory_type())) {
      return std::make_shared<SumHost<Src>>();
    }
#if NUMERIC_ENABLE_HIP
    else if (is_device_accessible(src.memory_type())) {
      NUMERIC_ERROR("Sum is not yet implemented on device!");
    }
#endif
    else {
      NUMERIC_ERROR("Unknown memory type: \"{}\"",
                    to_string(src.memory_type()));
    }
  }
};

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t
sum(const memory::ArrayBase<Src> &src) {
  Summer<Src> summer(src.derived());
  return summer(src.derived());
}

} // namespace numeric::math

#endif
