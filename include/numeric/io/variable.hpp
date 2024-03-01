#ifndef NUMERIC_IO_VARIABLE_HPP_
#define NUMERIC_IO_VARIABLE_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/utils/datatype.hpp>
#include <numeric/utils/error.hpp>
#include <vector>

namespace numeric::io {

class Variable {
public:
  Variable() = default;
  Variable(const Variable &) = default;
  Variable(Variable &&) = default;
  Variable &operator=(const Variable &) = default;
  Variable &operator=(Variable &&) = default;
  virtual ~Variable() = default;

  utils::Datatype datatype() const;
  std::vector<dim_t> dims() const;

  template <typename T, dim_t N> void read(memory::ArrayView<T, N> arr) const {
    NUMERIC_ERROR_IF(!memory::is_host_accessible(arr.memory_type()),
                     "Array must be accessible on the host. Got {}",
                     to_string(arr.memory_type()));
    NUMERIC_ERROR_IF(utils::to_datatype_v<T> != datatype(),
                     "Requested wrong datatype. Expected {}, got {}.",
                     to_string(datatype()), to_string(utils::to_datatype_v<T>));
    const std::vector<dim_t> d = dims();
    NUMERIC_ERROR_IF(
        N != d.size(),
        "Requested wrong number of dimensions. Expected {}, got {}.", d.size(),
        N);
    for (dim_t i = 0; i < N; ++i) {
      NUMERIC_ERROR_IF(arr.shape(i) != d[i],
                       "Mismatch in dimension {}. Expected {}, got {}.", i,
                       d[i], arr.shape(i));
    }
    do_read(arr.raw(), arr.shape().raw(), arr.stride().raw(), N);
  }

  template <typename T, dim_t N>
  memory::Array<T, N>
  read(memory::MemoryType memory_type = memory::MemoryType::HOST) const {
    const std::vector<dim_t> d = dims();
    NUMERIC_ERROR_IF(
        N != d.size(),
        "Requested wrong number of dimensions. Expected {}, got {}.", d.size(),
        N);
    memory::Shape<N> shape;
    for (dim_t i = 0; i < N; ++i) {
      shape[i] = d[i];
    }
    memory::Array<T, N> arr(shape, memory_type);
    read(arr);
    return arr;
  }

  template <typename T, dim_t N>
  void write(const memory::ArrayConstView<T, N> &arr) {
    NUMERIC_ERROR_IF(!memory::is_host_accessible(arr.memory_type()),
                     "Array must be accessible on the host. Got {}",
                     to_string(arr.memory_type()));
    NUMERIC_ERROR_IF(utils::to_datatype_v<T> != datatype(),
                     "Requested wrong datatype. Expected {}, got {}.",
                     to_string(datatype()), to_string(utils::to_datatype_v<T>));
    const std::vector<dim_t> d = dims();
    NUMERIC_ERROR_IF(
        N != d.size(),
        "Requested wrong number of dimensions. Expected {}, got {}.", d.size(),
        N);
    for (dim_t i = 0; i < N; ++i) {
      NUMERIC_ERROR_IF(arr.shape(i) != d[i],
                       "Mismatch in dimension {}. Expected {}, got {}.", i,
                       d[i], arr.shape(i));
    }
    do_write(arr.raw(), arr.shape().raw(), arr.stride().raw(), N);
  }

protected:
  virtual utils::Datatype do_datatype() const = 0;
  virtual std::vector<dim_t> do_dims() const = 0;
  virtual void do_read(void *data, const dim_t *shape, const dim_t *stride,
                       dim_t N) const = 0;
  virtual void do_write(const void *data, const dim_t *shape,
                        const dim_t *stride, dim_t N) = 0;
};

} // namespace numeric::io

#endif
