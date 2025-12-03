#include <numeric/mesh/neighbor_relation.hpp>
#include <set>
#include <vector>

namespace numeric::mesh {

math::graph::SparseGraph
neighboring(dim_t num_subelements,
            const memory::ArrayConstView<dim_t, 2> &relations) {
  const dim_t num_elements = relations.shape(1);
  // Create a map from subelement index to a set of parent elements
  std::vector<std::set<dim_t>> superelements(num_subelements);
  for (dim_t element = 0; element < num_elements; ++element) {
    for (dim_t subelement = 0; subelement < relations.shape(0); ++subelement) {
      superelements[relations(subelement, element)].insert(element);
    }
  }
  // Each pair of parent elements of a given subelement is neighboring via that
  // subelement
  std::vector<std::set<dim_t>> connected(num_elements);
  for (dim_t subelement = 0; subelement < num_subelements; ++subelement) {
    for (dim_t from : superelements[subelement]) {
      for (dim_t to : superelements[subelement]) {
        if (from == to) {
          continue;
        }
        connected[from].insert(to);
      }
    }
  }
  // Convert to a sparse graph datastructure
  dim_t num_edges = 0;
  for (dim_t element = 0; element < num_elements; ++element) {
    num_edges += connected[element].size();
  }
  math::graph::SparseGraph graph(num_elements, num_edges);
  memory::ArrayView<dim_t, 1> index = graph.index_array();
  memory::ArrayView<dim_t, 1> edges = graph.edge_array();
  index(0) = 0;
  for (dim_t element = 0; element < num_elements; ++element) {
    dim_t i = index(element);
    const auto &e = connected[element];
    index(element + 1) = i + e.size();
    for (dim_t to : e) {
      edges(i) = to;
      ++i;
    }
  }
  return graph;
}

} // namespace numeric::mesh
