#ifndef NUMERIC_MATH_FES_FE_SPACE_HPP_
#define NUMERIC_MATH_FES_FE_SPACE_HPP_

#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/subelement_relation.hpp>
#include <vector>

namespace numeric::math::fes {

namespace internal {

void compute_independent_element_groups(
    const memory::ArrayConstView<dim_t, 2> &external_dofs,
    std::vector<memory::Array<dim_t, 1>> &groups);

}

/**
 * @brief Ordering of the degrees of freedom on shared boundaries of elements
 *
 * LOCAL ordering orders the DOFs according to the orientation of the subelement
 * of each individual element. This is required for Lagrange-type basis
 * functions. GLOBAL ordering fixes an order of the DOFs on each subelement that
 * is consistent between all cells.
 */
enum struct BoundaryDOFOrdering { LOCAL, GLOBAL };

/**
 * @brief Generic Finite Element Space (FESpace) template declaration.
 *
 * @tparam Basis The basis function type.
 * @tparam Mesh The mesh type.
 */
template <typename Basis, typename Mesh> class FESpace {
public:
  /// Mesh type alias.
  using mesh_t = Mesh;
  /// Basis type alias.
  using basis_t = Basis;
};

/**
 * @brief Specialization of FESpace for unstructured meshes.
 *
 * Provides functionality for initializing and managing degrees of freedom
 * (DoFs) for elements in an unstructured finite element mesh.
 *
 * @tparam Basis The basis function type.
 * @tparam Scalar The scalar type used in the mesh.
 * @tparam ElementTypes Variadic list of element types in the mesh.
 */
template <typename Basis, typename Scalar, typename... ElementTypes>
class FESpace<Basis, mesh::UnstructuredMesh<Scalar, ElementTypes...>> {
public:
  /// Mesh type alias for the unstructured mesh.
  using mesh_t = mesh::UnstructuredMesh<Scalar, ElementTypes...>;
  /// Basis type alias.
  using basis_t = Basis;
  /// The polynomial order of the mesh elements.
  static constexpr dim_t mesh_order = (ElementTypes::order, ...);

  // Ensure all element types in the mesh have the same polynomial order
  static_assert(((ElementTypes::order == mesh_order) && ...),
                "FESpace assumes uniform order of mesh elements");

  /**
   * @brief Construct a finite element space from a mesh.
   *
   * Initializes the DoF mapping structures.
   *
   * @param mesh Shared pointer to a constant mesh object.
   * @param dof_order Ordering of the DOFs on shared subelements
   */
  FESpace(const std::shared_ptr<mesh_t> &mesh,
          BoundaryDOFOrdering dof_order = BoundaryDOFOrdering::LOCAL)
      : mesh_(mesh), dof_maps_(memory::Array<dim_t, 2>(memory::Shape<2>(
                         basis_t::template num_basis_functions<
                             typename ElementTypes::ref_el_t>(),
                         mesh_->template num_elements<ElementTypes>()))...),
        num_dofs_(0), dof_order_(dof_order) {
    init_dofs();
  }

  /**
   * @brief Move the FE space to the given memory
   */
  void to(memory::MemoryType memory_type) {
    mesh_->to(memory_type);
    (dof_maps_.template get<ElementTypes>().to(memory_type), ...);
    (
        [&](auto groups) {
          for (auto &group : groups) {
            group.to(memory_type);
          }
        }(independent_element_groups<ElementTypes>()),
        ...);
  }

  /**
   * @brief Get the associated mesh.
   *
   * @return Shared pointer to the constant mesh.
   */
  std::shared_ptr<const mesh_t> mesh() const { return mesh_; }

  /**
   * @brief Get the total number of degrees of freedom.
   *
   * @return Number of DoFs.
   */
  dim_t num_dofs() const { return num_dofs_; }

  /**
   * @brief Get the degree of freedom mapping for a specific element type.
   *
   * @tparam Element The element type.
   * @return A constant view of the DoF mapping array.
   */
  template <typename Element> memory::ArrayConstView<dim_t, 2> dof_map() const {
    return dof_maps_.template get<Element>();
  }

  memory::MemoryType memory_type() const noexcept {
    return mesh_->memory_type();
  }

  template <typename Element>
  const std::vector<memory::Array<dim_t, 1>> &
  independent_element_groups() const {
    auto &groups = independent_elements_.template get<Element>();
    if (groups.size() == 0) {
      static constexpr dim_t num_external_dofs =
          basis_t::template num_basis_functions<typename Element::ref_el_t>() -
          basis_t::template num_interior_basis_functions<
              typename Element::ref_el_t>();
      memory::ArrayConstView<dim_t, 2> dofs = dof_map<Element>();
      std::cout << "Computing sets of independent " << Element::name
                << std::endl;
      internal::compute_independent_element_groups(
          dofs(memory::Slice(0, num_external_dofs), memory::Slice()), groups);
      for (auto &group : groups) {
        group.to(memory_type());
      }
    }
    return groups;
  }

private:
  /// Shared pointer to the mesh.
  std::shared_ptr<mesh_t> mesh_;
  /// Mapping from element types to DoF arrays.
  utils::TypeIndexedMap<memory::Array<dim_t, 2>, ElementTypes...> dof_maps_;
  /// Total number of degrees of freedom.
  dim_t num_dofs_;
  /// Ordering of the DOFs
  BoundaryDOFOrdering dof_order_;
  /// Coloring of mesh elements so that no two elements with the same color
  /// share a DOF
  mutable utils::TypeIndexedMap<std::vector<memory::Array<dim_t, 1>>,
                                ElementTypes...>
      independent_elements_;

  /**
   * @brief Initialize the degrees of freedom for the entire mesh.
   */
  void init_dofs() {
    // Map to track the current offset of DoFs per element type
    utils::TypeIndexedMap<dim_t, ElementTypes...> current_highest_dof_idx;
    ((current_highest_dof_idx.template get<ElementTypes>() = 0), ...);

    // Initialize DoFs for all supported mesh element types
    init_dofs_on<mesh::Point<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Segment<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Tria<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Quad<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Tetra<mesh_order>>(current_highest_dof_idx);
    init_dofs_on<mesh::Cube<mesh_order>>(current_highest_dof_idx);
  }

  /**
   * @brief Initialize DoFs on mesh entities of a specific type.
   *
   * @tparam Element The element type.
   * @param current_highest_dof_idx Map tracking the current index for each
   * element type.
   */
  template <typename Element>
  void init_dofs_on(
      utils::TypeIndexedMap<dim_t, ElementTypes...> &current_highest_dof_idx) {
    // Check if subelements of this type are used in any of the element types
    static constexpr bool has_dofs_associated_with_subelement =
        ((!meta::is_same_v<ElementTypes, Element> &&
          basis_t::template total_num_basis_functions<
              typename ElementTypes::ref_el_t, typename Element::ref_el_t>() >
              0) ||
         ...);

    if (has_dofs_associated_with_subelement) {
      // Get subelement relationships
      const auto [elements, relations] = subelement_relation<Element>(*mesh_);

      // Initialize DoFs on subelements for all element types
      (init_dofs_on_subelements<ElementTypes, Element>(
           mesh_->template get_elements<ElementTypes>(), elements,
           relations.template get<ElementTypes>(),
           current_highest_dof_idx.template get<ElementTypes>()),
       ...);

      // Account for DoFs that belong directly to this subelement type
      const dim_t num_elements = elements.shape(1);
      static constexpr dim_t num_dofs_per_element =
          basis_t::template num_interior_basis_functions<
              typename Element::ref_el_t>();
      num_dofs_ += num_elements * num_dofs_per_element;
    }

    // Check if this element type has its own interior DoFs
    static constexpr bool has_interior_dofs_associated_with_element =
        ((meta::is_same_v<Element, ElementTypes> &&
          basis_t::template num_interior_basis_functions<
              typename Element::ref_el_t>() > 0) ||
         ...);
    if constexpr (has_interior_dofs_associated_with_element) {
      // Initialize those DoFs
      init_interior_dofs_on<Element>(
          current_highest_dof_idx.template get<Element>());
    }
  }

  /**
   * @brief Initialize DoFs associated with subelements of an element type.
   *
   * @tparam Element The element type.
   * @tparam Subelement The subelement type.
   * @param relations Array mapping subelements to parent elements.
   * @param current_highest_dof_idx Current offset for this element type.
   */
  template <typename Element, typename Subelement>
  void
  init_dofs_on_subelements(const memory::ArrayConstView<dim_t, 2> &elements,
                           const memory::ArrayConstView<dim_t, 2> &subelements,
                           const memory::ArrayView<dim_t, 2> &relations,
                           dim_t &current_highest_dof_idx) {
    const dim_t num_elements = relations.shape(1);
    static constexpr dim_t num_subelements =
        Element::template num_subelements<Subelement>();
    static constexpr dim_t num_corners = Subelement::ref_el_t::num_nodes;
    auto &dof_map = dof_maps_.template get<Element>();

    // For each subelement on the element
    for (dim_t subelement = 0; subelement < num_subelements; ++subelement) {
      const dim_t num_dofs_on_subelement =
          basis_t::template num_basis_functions<typename Element::ref_el_t,
                                                typename Subelement::ref_el_t>(
              subelement);
      // Loop over all parent elements
      for (dim_t element = 0; element < num_elements; ++element) {
        // Assign each DoF to the global index space
        for (dim_t dof = 0; dof < num_dofs_on_subelement; ++dof) {
          // Compute the permutation of the corners of the local subelement with
          // respect to the corners of the global element
          dim_t local_corner_dofs[num_corners];
          dim_t global_corner_dofs[num_corners];
          dim_t sub_node_idxs[num_corners];
          Element::ref_el_t::template subelement_node_idxs<
              typename Subelement::ref_el_t>(subelement, sub_node_idxs);
          for (dim_t corner = 0; corner < num_corners; ++corner) {
            local_corner_dofs[corner] =
                elements(sub_node_idxs[corner], element);
            global_corner_dofs[corner] =
                subelements(corner, relations(subelement, element));
          }
          dim_t perm[num_corners];
          for (dim_t corner_loc = 0; corner_loc < num_corners; ++corner_loc) {
            for (dim_t corner_glob = 0; corner_glob < num_corners;
                 ++corner_glob) {
              if (local_corner_dofs[corner_loc] ==
                  global_corner_dofs[corner_glob]) {
                perm[corner_loc] = corner_glob;
                break;
              }
            }
          }
          const dim_t local_dof_idx = current_highest_dof_idx + dof;
          const dim_t dof_on_subelement =
              basis_t::template interior_dof_idx_under_permutation<Subelement>(
                  dof, perm);
          const dim_t global_dof_idx =
              num_dofs_ +
              relations(subelement, element) * num_dofs_on_subelement +
              dof_on_subelement;
          dof_map(local_dof_idx, element) = global_dof_idx;
        }
      }

      // Advance the offset for the next subelement
      current_highest_dof_idx += num_dofs_on_subelement;
    }
  }

  /**
   * @brief Initialize DoFs associated with the interior of elements.
   *
   * @tparam Element The element type.
   * @param current_highest_dof_idx Current offset for this element type.
   */
  template <typename Element>
  void init_interior_dofs_on(dim_t &current_highest_dof_idx) {
    auto &dof_map = dof_maps_.template get<Element>();
    const dim_t num_elements = dof_map.shape(1);
    static constexpr dim_t num_interior_dofs =
        basis_t::template num_interior_basis_functions<
            typename Element::ref_el_t>();

    // For each element, assign interior DoFs sequentially
    for (dim_t element = 0; element < num_elements; ++element) {
      for (dim_t dof = 0; dof < num_interior_dofs; ++dof) {
        const dim_t local_dof_idx = current_highest_dof_idx + dof;
        const dim_t global_dof_idx =
            num_dofs_ + element * num_interior_dofs + dof;
        dof_map(local_dof_idx, element) = global_dof_idx;
      }
    }

    // Update current offset and total number of DoFs
    current_highest_dof_idx += num_interior_dofs;
    num_dofs_ += num_elements * num_interior_dofs;
  }
};

} // namespace numeric::math::fes

#endif
