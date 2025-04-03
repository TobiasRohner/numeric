#ifndef NUMERIC_EQUATIONS_FEM_FINITE_ELEMENT_VECTOR_HPP_
#define NUMERIC_EQUATIONS_FEM_FINITE_ELEMENT_VECTOR_HPP_

#include <numeric/math/fes/fe_space.hpp>
#include <numeric/math/quad/quad_rule.hpp>

namespace numeric::equations::fem {

template <typename FES, typename ElementVectorFactory>
class FiniteElementVector {
  static_assert(
      !meta::is_same_v<FES, FES>,
      "FiniteElementVector is not specialized for the given FE space");
};

template <typename Basis, typename ScalarMesh, typename... ElementTypes,
          typename ElementVectorFactory>
class FiniteElementVector<
    math::fes::FESpace<Basis,
                       mesh::UnstructuredMesh<ScalarMesh, ElementTypes...>>,
    ElementVectorFactory> {
public:
  using scalar_t = typename ElementVectorFactory::scalar_t;
  using scalar_mesh_t = ScalarMesh;
  using basis_t = Basis;
  using mesh_t = mesh::UnstructuredMesh<ScalarMesh, ElementTypes...>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  template <typename Element>
  using element_vector_t =
      typename ElementVectorFactory::template create<Element>;

  // TODO: Is this order correct?
  FiniteElementVector(const fes_t &fes)
      : FiniteElementVector(fes, 2 * basis_t::order) {}
  FiniteElementVector(const fes_t &fes, dim_t order)
      : FiniteElementVector(fes, build_qr(order)) {}

  template <typename Func>
  void assemble(const fes_t &fes, Func &&f,
                memory::ArrayView<scalar_t, 1> out) const {
    out = 0;
    const auto mesh = fes.mesh();
    const auto vertices = mesh->vertices();
    const dim_t world_dim = vertices.shape(0);
    mesh_t::for_all_element_types([&]<typename ElementType>(
                                      meta::type_tag<ElementType>) {
      const auto elements = mesh->template get_elements<ElementType>();
      const auto dofs = fes.template dof_map<ElementType>();
      static constexpr dim_t num_nodes = ElementType::num_nodes;
      static constexpr dim_t num_basis_functions =
          element_vector_t<ElementType>::num_basis_functions;
      const dim_t num_elements = elements.shape(1);
      scalar_t(*nodes)[num_nodes] =
          reinterpret_cast<scalar_t(*)[num_nodes]>(work_.raw());
      void *apply_work = work_.raw() + world_dim * num_nodes * sizeof(scalar_t);
      scalar_t local_vector[num_basis_functions];
      const element_vector_t<ElementType> &element_vector =
          elem_vecs_.template get<element_vector_t<ElementType>>();
      for (dim_t element = 0; element < num_elements; ++element) {
        for (dim_t node = 0; node < num_nodes; ++node) {
          for (dim_t dim = 0; dim < world_dim; ++dim) {
            nodes[dim][node] = vertices(dim, elements(node, element));
          }
        }
        element_vector.apply(f, nodes, world_dim, local_vector, apply_work);
        for (dim_t bf = 0; bf < num_basis_functions; ++bf) {
          out(dofs(bf, element)) += local_vector[bf];
        }
      }
    });
  }

private:
  mutable memory::Array<char, 1> work_;
  utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...> qr_points_;
  utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>
      qr_weights_;
  utils::Tuple<element_vector_t<ElementTypes>...> elem_vecs_;

  FiniteElementVector(
      const fes_t &fes,
      utils::Tuple<
          utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
          utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
          &&qr)
      : work_(memory::Shape<1>(apply_work_size(fes.mesh()->world_dim())),
              memory::MemoryType::HOST),
        qr_points_(std::move(qr.template get<0>())),
        qr_weights_(std::move(qr.template get<1>())),
        elem_vecs_(element_vector_t<ElementTypes>(
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
    const dim_t nodes_size = world_dim * Element::num_nodes * sizeof(scalar_t);
    const dim_t apply_size =
        element_vector_t<Element>::apply_work_size(world_dim);
    return nodes_size + apply_size;
  }
};

} // namespace numeric::equations::fem

#endif
