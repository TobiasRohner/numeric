#ifndef NUMERIC_MATH_MESH_FUNCTION_HPP_
#define NUMERIC_MATH_MESH_FUNCTION_HPP_

#include <memory>
#include <numeric/memory/array.hpp>

namespace numeric::math {

template <typename Scalar, typename FES> class MeshFunction {
public:
  using scalar_t = Scalar;
  using fes_t = FES;
  using basis_t = typename fes_t::basis_t;

  MeshFunction(const std::shared_ptr<fes_t> &fes)
      : MeshFunction(fes, fes->memory_type()) {}
  MeshFunction(const std::shared_ptr<fes_t> &fes,
               memory::MemoryType memory_type)
      : fes_(fes), dofs_(memory::Shape<1>(fes->num_dofs()), memory_type) {}
  MeshFunction(const MeshFunction &) = default;
  MeshFunction(MeshFunction &&) = default;
  MeshFunction &operator=(const MeshFunction &) = default;
  MeshFunction &operator=(MeshFunction &&) = default;

  void to(memory::MemoryType memory_type) {
    fes_->to(memory_type);
    dofs_.to(memory_type);
  }

  template <typename ElementType>
  void eval(dim_t element, const scalar_t (*refcoords)[ElementType::dim],
            scalar_t *out, dim_t num_points) const {
    static constexpr dim_t num_dofs =
        basis_t::template num_basis_functions<ElementType>();
    scalar_t coeffs[num_dofs];
    element_coefficients(element, coeffs);
    for (dim_t point = 0; point < num_points; ++point) {
      out[point] =
          basis_t::template eval<ElementType>(refcoords[point], coeffs);
    }
  }

  memory::ArrayView<scalar_t, 1> dofs() { return dofs_; }

  memory::ArrayConstView<scalar_t, 1> dofs() const { return dofs_; }

  template <typename ElementType>
  void element_coefficients(dim_t element, scalar_t *coeffs) const {
    static constexpr dim_t num_dofs =
        basis_t::template num_basis_functions<ElementType>();
    const auto dofmap = fes_->template get_dofs<ElementType>();
    for (dim_t i = 0; i < num_dofs; ++i) {
      coeffs[i] = dofs_(dofmap(i, element));
    }
  }

  template <typename ScalarOther, typename FESOther>
  void interpolate(const MeshFunction<ScalarOther, FESOther> &other) {
    using mesh_t = typename fes_t::mesh_t;
    static_assert(meta::is_same_v<mesh_t, typename FESOther::mesh_t>,
                  "Meshes must be the same for interpolation");
    auto mesh = fes_->mesh();
    mesh_t::for_all_element_types([&]<typename ElementType>(
                                      meta::type_tag<ElementType>) {
      using ref_el_t = typename ElementType::ref_el_t;
      using element_basis_t = typename basis_t::element_basis_t<ref_el_t>;
      auto dofmap = fes_->template get_dofs<ElementType>();
      ScalarOther interpolation_nodes[element_basis_t::num_interpolation_nodes]
                                     [ElementType::dim];
      element_basis_t::interpolation_nodes(interpolation_nodes);
      ScalarOther nodal_values[element_basis_t::num_interpolation_nodes];
      ScalarOther interpolated_coeffs[element_basis_t::num_basis_functions];
      for (dim_t element = 0;
           element < mesh->template num_elements<ElementType>(); ++element) {
        other.eval(element, interpolation_nodes, nodal_values,
                   element_basis_t::num_interpolation_nodes);
        element_basis_t::interpolate(nodal_values, interpolated_coeffs);
        for (dim_t dof = 0; dof < element_basis_t::num_basis_functions; ++dof) {
          dofs_(dofmap(dof, element)) = interpolated_coeffs[dof];
        }
      }
    });
  }

private:
  std::shared_ptr<fes_t> fes_;
  memory::Array<scalar_t, 1> dofs_;
};

} // namespace numeric::math

#endif
