#ifndef NUMERIC_MATH_FES_FE_SPACE_HPP_
#define NUMERIC_MATH_FES_FE_SPACE_HPP_

#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/subelement_relation.hpp>

namespace numeric::math::fes {

template <typename Basis, typename Mesh> class FESpace {
public:
  using mesh_t = Mesh;
  using basis_t = Basis;
};

template <typename Basis, typename Scalar, typename... ElementTypes>
class FESpace<Basis, mesh::UnstructuredMesh<Scalar, ElementTypes...>> {
public:
  using mesh_t = mesh::UnstructuredMesh<Scalar, ElementTypes...>;
  using basis_t = Basis;
  static constexpr dim_t mesh_order = (ElementTypes::order, ...);
  static_assert(((ElementTypes::order == mesh_order) && ...),
                "FESpace assumes uniform order of mesh elements");

  FESpace(const std::shared_ptr<mesh_t> &mesh)
      : mesh_(mesh), dof_maps_(memory::Array<dim_t, 2>(memory::Shape<2>(
                         basis_t::template num_basis_functions<ElementTypes>(),
                         mesh_->template num_elements<ElementTypes>()))...),
        num_dofs_(0) {
    init_dofs();
  }

  std::shared_ptr<const mesh_t> mesh() const { return mesh_; }

  dim_t num_dofs() const { return num_dofs_; }

  template <typename Element> memory::ArrayConstView<dim_t, 2> dof_map() const {
    return dof_maps_.template get<Element>();
  }

private:
  std::shared_ptr<mesh_t> mesh_;
  utils::TypeIndexedMap<memory::Array<dim_t, 2>, ElementTypes...> dof_maps_;
  dim_t num_dofs_;

  void init_dofs() {
    utils::TypeIndexedMap<dim_t, ElementTypes...> current_highest_dof_idx;
    ((current_highest_dof_idx.template get<ElementTypes>() = 0), ...);
    init_dofs_on<mesh::Point<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Segment<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Tria<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Quad<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Tetra<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Cube<mesh_order>>(current_highest_dof_idx);
  }

  template <typename Element>
  void init_dofs_on(
      utils::TypeIndexedMap<dim_t, ElementTypes...> &current_highest_dof_idx) {
    static constexpr bool has_dofs_associated_with_subelement =
        ((!meta::is_same_v<ElementTypes, Element> &&
          basis_t::template total_num_basis_functions<ElementTypes, Element>() >
              0) ||
         ...);
    if (has_dofs_associated_with_subelement) {
      const auto [elements, relations] = subelement_relation<Element>(*mesh_);
      (init_dofs_on_subelements<ElementTypes, Element>(
           relations.template get<ElementTypes>(),
           current_highest_dof_idx.template get<ElementTypes>()),
       ...);
      const dim_t num_elements = elements.shape(1);
      static constexpr dim_t num_dofs_per_element =
          basis_t::template num_interior_basis_functions<Element>();
      num_dofs_ += num_elements * num_dofs_per_element;
    }
    static constexpr bool has_interior_dofs_associated_with_element =
        ((meta::is_same_v<Element, ElementTypes> &&
          basis_t::template num_interior_basis_functions<Element>() > 0) ||
         ...);
    if constexpr (has_interior_dofs_associated_with_element) {
      init_interior_dofs_on<Element>(
          current_highest_dof_idx.template get<Element>());
    }
  }

  template <typename Element, typename Subelement>
  void init_dofs_on_subelements(const memory::ArrayView<dim_t, 2> &relations,
                                dim_t &current_highest_dof_idx) {
    const dim_t num_elements = relations.shape(1);
    static constexpr dim_t num_subelements =
        Element::template num_subelements<Subelement>();
    auto &dof_map = dof_maps_.template get<Element>();
    for (dim_t subelement = 0; subelement < num_subelements; ++subelement) {
      const dim_t num_dofs_on_subelement =
          basis_t::template num_basis_functions<Element, Subelement>(
              subelement);
      for (dim_t element = 0; element < num_elements; ++element) {
        for (dim_t dof = 0; dof < num_dofs_on_subelement; ++dof) {
          const dim_t local_dof_idx = current_highest_dof_idx + dof;
          const dim_t global_dof_idx =
              num_dofs_ +
              relations(subelement, element) * num_dofs_on_subelement + dof;
          dof_map(local_dof_idx, element) = global_dof_idx;
        }
      }
      current_highest_dof_idx += num_dofs_on_subelement;
    }
  }

  template <typename Element>
  void init_interior_dofs_on(dim_t &current_highest_dof_idx) {
    auto &dof_map = dof_maps_.template get<Element>();
    const dim_t num_elements = dof_map.shape(1);
    static constexpr dim_t num_interior_dofs =
        basis_t::template num_interior_basis_functions<Element>();
    for (dim_t element = 0; element < num_elements; ++element) {
      for (dim_t dof = 0; dof < num_interior_dofs; ++dof) {
        const dim_t local_dof_idx = current_highest_dof_idx + dof;
        const dim_t global_dof_idx =
            num_dofs_ + element * num_interior_dofs + dof;
        dof_map(local_dof_idx, element) = global_dof_idx;
      }
    }
    current_highest_dof_idx += num_interior_dofs;
    num_dofs_ += num_elements * num_interior_dofs;
  }
};

} // namespace numeric::math::fes

#endif
