#ifndef NUMERIC_EQUATIONS_FEM_LOAD_ELEMENT_VECTOR_HPP_
#define NUMERIC_EQUATIONS_FEM_LOAD_ELEMENT_VECTOR_HPP_

#include <numeric/math/fes/fe_space.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/mesh/elements.hpp>

namespace numeric::equations::fem {

template <typename Scalar, typename Basis, typename Element>
class LoadElementVector {
public:
  using scalar_t = Scalar;
  using basis_t = Basis;
  using element_t = Element;
  using ref_el_t = typename element_t::ref_el_t;
  using element_basis_t = basis_t::template element_basis_t<ref_el_t>;
  template <typename OtherElement>
  using rebind = LoadElementVector<Scalar, Basis, OtherElement>;

  static constexpr dim_t num_nodes = element_t::num_nodes;
  static constexpr dim_t num_basis_functions =
      basis_t::template num_basis_functions<ref_el_t>();

  LoadElementVector(const memory::ArrayConstView<scalar_t, 2> &qr_points,
                    const memory::ArrayConstView<scalar_t, 1> &qr_weights)
      : qr_points_(qr_points), qr_weights_(qr_weights) {}
  LoadElementVector(const LoadElementVector &) = default;
  LoadElementVector(LoadElementVector &&) = default;
  LoadElementVector &operator=(const LoadElementVector &) = default;
  LoadElementVector &operator=(LoadElementVector &&) = default;

  static constexpr dim_t apply_work_size(dim_t world_dim) {
    const dim_t global_work_size = world_dim * sizeof(scalar_t);
    const dim_t ie_work_size =
        element_t::template integration_element_work_size<scalar_t>(world_dim);
    return math::max(global_work_size, ie_work_size);
  }

  template <typename Func>
  void apply(Func &&f, const scalar_t (*nodes)[num_nodes], dim_t world_dim,
             scalar_t (&out)[num_basis_functions], void *work) const {
    scalar_t local[element_t::dim];
    scalar_t basis[num_basis_functions];
    scalar_t *global = static_cast<scalar_t *>(work);
    scalar_t *ie_work = static_cast<scalar_t *>(work);
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      out[i] = 0;
    }
    const dim_t num_points = qr_weights_.shape(0);
    for (dim_t i = 0; i < num_points; ++i) {
      for (dim_t j = 0; j < element_t::dim; ++j) {
        local[j] = qr_points_(i, j);
      }
      const scalar_t ie =
          element_t::integration_element(nodes, local, world_dim, ie_work);
      const scalar_t weight = ie * qr_weights_(i);
      element_t::local_to_global(nodes, local, global, world_dim);
      const scalar_t fval = f(global);
      element_basis_t::eval_basis(local, basis);
      for (dim_t j = 0; j < num_basis_functions; ++j) {
        out[j] += weight * fval * basis[j];
      }
    }
  }

private:
  memory::ArrayConstView<scalar_t, 2> qr_points_;
  memory::ArrayConstView<scalar_t, 1> qr_weights_;
};

template <typename Scalar, typename Basis> struct LoadElementVectorFactory {
  using scalar_t = Scalar;
  using basis_t = Basis;
  template <typename Element>
  using create = LoadElementVector<Scalar, Basis, Element>;
};

} // namespace numeric::equations::fem

#endif
