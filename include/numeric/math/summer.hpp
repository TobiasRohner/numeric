#ifndef NUMERIC_MATH_SUMMER_HPP_
#define NUMERIC_MATH_SUMMER_HPP_

#include <memory>
#include <numeric/math/sum_host.hpp>
#include <numeric/utils/error.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/math/sum_device.hpp>
#endif

namespace numeric::math {

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
      return std::make_shared<SumDevice<Src>>();
    }
#endif
    else {
      NUMERIC_ERROR("Unknown memory type: \"{}\"",
                    to_string(src.memory_type()));
    }
  }
};

} // namespace numeric::math

#endif
