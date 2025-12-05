#ifndef NUMERIC_MESH_SUBELEMENT_RELATION_HPP_
#define NUMERIC_MESH_SUBELEMENT_RELATION_HPP_

#include <algorithm>
#include <array>
#include <map>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/unstructured_mesh_const_view.hpp>
#include <tuple>
#include <vector>

namespace numeric::mesh {

namespace detail {

/**
 * @brief A helper structure for managing subelements.
 *
 * NodeBags of two subelements sharing the same nodes but with a different
 * ordering will be considered to be equivalent.
 *
 * @tparam Element Type of the element.
 */
template <typename Element> struct NodeBag {
  static constexpr dim_t N = Element::num_nodes;
  std::array<dim_t, N> idxs;

  NodeBag(const std::array<dim_t, N> &nodes) {
    std::copy(std::begin(nodes), std::end(nodes), std::begin(idxs));
    std::sort(std::begin(idxs), std::end(idxs));
  }
  NodeBag(const NodeBag &) = default;
  NodeBag &operator=(const NodeBag &) = default;

  constexpr auto operator<=>(const NodeBag &other) const noexcept {
    return std::lexicographical_compare_three_way(
        std::begin(idxs), std::end(idxs), std::begin(other.idxs),
        std::end(other.idxs));
  }
};

template <typename Element, typename Subelement>
memory::Array<dim_t, 2> subelement_relation(
    const memory::ArrayConstView<dim_t, 2> &elements,
    std::vector<std::array<dim_t, Subelement::num_nodes>> &subelements,
    std::map<NodeBag<Subelement>, dim_t> &subelement_idxs) {
  if (!is_host_accessible(elements.memory_type())) {
    memory::Array<dim_t, 2> elements_host(elements.shape(),
                                          memory::MemoryType::HOST);
    elements_host = elements;
    return subelement_relation<Element, Subelement>(elements_host, subelements,
                                                    subelement_idxs);
  } else {
    static constexpr dim_t N_nodes = Subelement::num_nodes;
    static constexpr dim_t N_sub =
        Element::template num_subelements<Subelement>();
    memory::Array<dim_t, 2> relation(
        memory::Shape<2>(N_sub, elements.shape(1)));
    for (dim_t element = 0; element < elements.shape(1); ++element) {
      for (dim_t subelement = 0; subelement < N_sub; ++subelement) {
        std::array<dim_t, N_nodes> nodes;
        std::array<dim_t, N_nodes> node_idxs;
        Element::template subelement_node_idxs<Subelement>(subelement,
                                                           node_idxs.data());
        for (dim_t i = 0; i < N_nodes; ++i) {
          nodes[i] = elements(node_idxs[i], element);
        }
        detail::NodeBag<Subelement> bag(nodes);
        auto idx = subelement_idxs.find(bag);
        if (idx != subelement_idxs.end()) {
          relation(subelement, element) = idx->second;
        } else {
          relation(subelement, element) = subelements.size();
          subelement_idxs[bag] = subelements.size();
          subelements.emplace_back(nodes);
        }
      }
    }
    return relation;
  }
}

} // namespace detail

template <typename Subelement, typename Scalar, typename... ElementTypes>
std::tuple<memory::Array<dim_t, 2>,
           utils::TypeIndexedMap<memory::Array<dim_t, 2>, ElementTypes...>>
subelement_relation(
    const UnstructuredMeshConstView<Scalar, ElementTypes...> &mesh) {
  static constexpr dim_t N_nodes = Subelement::num_nodes;

  utils::TypeIndexedMap<memory::Array<dim_t, 2>, ElementTypes...> relations;
  std::vector<std::array<dim_t, N_nodes>> subelements;
  std::map<detail::NodeBag<Subelement>, dim_t> subelement_idxs;

  const auto compute_relations = [&]<typename Element>() {
    relations.template get<Element>() =
        std::move(detail::subelement_relation<Element, Subelement>(
            mesh.template get_elements<Element>(), subelements,
            subelement_idxs));
  };
  (compute_relations.template operator()<ElementTypes>(), ...);

  memory::Array<dim_t, 2> subs(
      memory::Shape<2>(Subelement::num_nodes, subelements.size()),
      memory::MemoryType::HOST);
  for (dim_t subelement = 0; subelement < subs.shape(1); ++subelement) {
    for (dim_t node = 0; node < Subelement::num_nodes; ++node) {
      subs(node, subelement) = subelements[subelement][node];
    }
  }
  return {subs, relations};
}

template <typename Subelement, typename Scalar, typename... ElementTypes>
std::tuple<memory::Array<dim_t, 2>,
           utils::TypeIndexedMap<memory::Array<dim_t, 2>, ElementTypes...>>
subelement_relation(const UnstructuredMeshView<Scalar, ElementTypes...> &mesh) {
  return subelement_relation<Subelement>(
      UnstructuredMeshConstView<Scalar, ElementTypes...>(mesh));
}

template <typename Subelement, typename Scalar, typename... ElementTypes>
std::tuple<memory::Array<dim_t, 2>,
           utils::TypeIndexedMap<memory::Array<dim_t, 2>, ElementTypes...>>
subelement_relation(const UnstructuredMesh<Scalar, ElementTypes...> &mesh) {
  return subelement_relation<Subelement>(
      UnstructuredMeshConstView<Scalar, ElementTypes...>(mesh));
}

} // namespace numeric::mesh

#endif
