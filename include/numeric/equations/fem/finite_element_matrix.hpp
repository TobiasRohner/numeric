#ifndef NUMERIC_EQUATIONS_FEM_FINITE_ELEMENT_MATRIX_HPP_
#define NUMERIC_EQUATIONS_FEM_FINITE_ELEMENT_MATRIX_HPP_

#include <numeric/math/fes/fe_space.hpp>
#include <numeric/math/quad/quad_rule.hpp>

namespace numeric::equations::fem {

template <typename FES, typename ElementMatrixFactory>
class FiniteElementMatrix {
  static_assert(
      !meta::is_same_v<FES, FES>,
      "FiniteElementMatrix is not specialized for the given FE space");
};

template <typename Basis, typename ScalarMesh, typename... ElementTypes,
          typename ElementMatrixFactory>
class FiniteElementMatrix<
    math::fes::FESpace<Basis,
                       mesh::UnstructuredMesh<ScalarMesh, ElementTypes...>>,
    ElementMatrixFactory> {
public:
  using scalar_t = typename ElementMatrixFactory::scalar_t;
  using scalar_mesh_t = ScalarMesh;
  using basis_t = Basis;
  using mesh_t = mesh::UnstructuredMesh<ScalarMesh, ElementTypes...>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  template <typename Element>
  using element_matrix_t =
      typename ElementMatrixFactory::template create<Element>;

  // TODO: Is this order correct?
  FiniteElementMatrix(const fes_t &fes)
      : FiniteElementMatrix(fes, 2 * basis_t::order) {}
  FiniteElementMatrix(const fes_t &fes, dim_t order)
      : FiniteElementMatrix(fes, build_qr(order)) {}

  void apply(const fes_t &fes, const memory::ArrayConstView<scalar_t, 1> &u,
             memory::ArrayView<scalar_t, 1> out) const {
    // Clear out vector
    out = 0;
    // Compute matrix-vector product for each element type
    ((apply_to_element<ElementTypes>(fes, u, out, work_.raw())), ...);
  }

private:
  mutable memory::Array<char, 1> work_;
  utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...> qr_points_;
  utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>
      qr_weights_;
  utils::Tuple<element_matrix_t<ElementTypes>...> elem_mats_;

  FiniteElementMatrix(
      const fes_t &fes,
      utils::Tuple<
          utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
          utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
          &&qr)
      : work_(memory::Shape<1>(apply_work_size(fes.mesh().world_dim())),
              memory::MemoryType::HOST),
        qr_points_(std::move(qr.template get<0>())),
        qr_weights_(std::move(qr.template get<1>())),
        elem_mats_(element_matrix_t<ElementTypes>(
            qr_points_.template get<ElementTypes>(),
            qr_weights_.template get<ElementTypes>())...) {}

  static utils::Tuple<
      utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
      utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
  build_qr(dim_t order) {
    utils::Tuple<
        utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
        utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
        qr;
    const auto init_qr = [&]<typename Element>() {
      auto [p, w] = math::quad::quad_rule<typename Element::ref_el_t>(
          Element::order * order);
      qr.template get<0>().template get<Element>() = std::move(p);
      qr.template get<1>().template get<Element>() = std::move(w);
    };
    ((init_qr.template operator()<ElementTypes>()), ...);
    return qr;
  }

  static constexpr dim_t apply_work_size(dim_t world_dim) {
    dim_t size = 0;
    ((size =
          math::max(size, apply_to_element_work_size<ElementTypes>(world_dim))),
     ...);
    return size;
  }

  template <typename Element>
  static constexpr dim_t apply_to_element_work_size(dim_t world_dim) {
    constexpr dim_t num_nodes = Element::num_nodes;
    const dim_t nodes_size = sizeof(scalar_t) * world_dim * num_nodes;
    const dim_t elem_mat_size =
        element_matrix_t<Element>::apply_work_size(world_dim);
    return nodes_size + elem_mat_size;
  }

  template <typename Element>
  void apply_to_element(const fes_t &fes,
                        const memory::ArrayConstView<scalar_t, 1> &u,
                        memory::ArrayView<scalar_t, 1> out, void *work) const {
    static constexpr dim_t num_nodes = Element::num_nodes;
    static constexpr dim_t num_basis_functions =
        basis_t::template num_basis_functions<Element>();
    const mesh_t &mesh = fes.mesh();
    const dim_t num_elements = mesh.template num_elements<Element>();
    const dim_t world_dim = mesh.world_dim();
    const memory::ArrayConstView<scalar_mesh_t, 2> vertices = mesh.vertices();
    const memory::ArrayConstView<dim_t, 2> elements =
        mesh.template get_elements<Element>();
    const memory::ArrayConstView<dim_t, 2> dof_map =
        fes.template dof_map<Element>();
    const element_matrix_t<Element> &elem_mat =
        elem_mats_.template get<element_matrix_t<Element>>();
    scalar_t(*nodes)[num_nodes] = static_cast<scalar_t(*)[num_nodes]>(work);
    void *elem_mat_work = static_cast<scalar_t *>(work) + world_dim * num_nodes;
    scalar_t elem_vec_in[num_basis_functions];
    scalar_t elem_vec_out[num_basis_functions];
    // Compute matrix vector product
    for (dim_t element = 0; element < num_elements; ++element) {
      // Collect node positions of element
      for (dim_t node = 0; node < num_nodes; ++node) {
        const dim_t node_idx = elements(node, element);
        for (dim_t i = 0; i < world_dim; ++i) {
          nodes[i][node] = vertices(i, node_idx);
        }
      }
      // Gather the local dofs
      for (dim_t bf = 0; bf < num_basis_functions; ++bf) {
        const dim_t dof_idx = dof_map(bf, element);
        elem_vec_in[bf] = u(dof_idx);
      }
      // Apply local element matrix
      elem_mat.apply(nodes, elem_vec_in, world_dim, elem_vec_out,
                     elem_mat_work);
      // Scatter onto the global coefficient vector
      for (dim_t bf = 0; bf < num_basis_functions; ++bf) {
        const dim_t dof_idx = dof_map(bf, element);
        out(dof_idx) += elem_vec_out[bf];
      }
    }
  }
};

} // namespace numeric::equations::fem

#endif
