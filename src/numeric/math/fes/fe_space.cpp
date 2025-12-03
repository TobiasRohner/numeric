#include <map>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/math/graph/coloring.hpp>
#include <numeric/math/graph/sparse_graph.hpp>
#include <numeric/math/reduce.hpp>
#include <set>

namespace numeric::math::fes {

namespace internal {

void compute_independent_element_groups(
    const memory::ArrayConstView<dim_t, 2> &external_dofs,
    std::vector<memory::Array<dim_t, 1>> &groups) {
  if (!is_host_accessible(external_dofs.memory_type())) {
    memory::Array<dim_t, 2> ed(external_dofs.shape(), memory::MemoryType::HOST);
    ed = external_dofs;
    return compute_independent_element_groups(ed, groups);
  } else {
    const dim_t num_elements = external_dofs.shape(1);
    const dim_t num_dofs_per_element = external_dofs.shape(0);
    const dim_t min_dof = min(external_dofs);
    const dim_t max_dof = max(external_dofs);
    const dim_t num_dofs = max_dof - min_dof + 1;
    // Construct a map from dof to a set of elements contributing to it
    std::vector<std::set<dim_t>> contributing_elements(num_dofs);
    for (dim_t element = 0; element < num_elements; ++element) {
      for (dim_t dof = 0; dof < num_dofs_per_element; ++dof) {
        contributing_elements[external_dofs(dof, element) - min_dof].insert(
            element);
      }
    }
    // Convert this to a graph of conflicting elements
    std::vector<std::set<dim_t>> conflicts(num_elements);
    for (dim_t dof = 0; dof < num_dofs; ++dof) {
      for (dim_t from : contributing_elements[dof]) {
        for (dim_t to : contributing_elements[dof]) {
          if (from == to) {
            continue;
          }
          conflicts[from].insert(to);
        }
      }
    }
    contributing_elements.clear();
    // Convert the data to a sparse graph data structure
    dim_t num_edges = 0;
    for (const auto &e : conflicts) {
      num_edges += e.size();
    }
    graph::SparseGraph graph(num_elements, num_edges);
    memory::ArrayView<dim_t, 1> index = graph.index_array();
    memory::ArrayView<dim_t, 1> edges = graph.edge_array();
    index(0) = 0;
    for (dim_t element = 0; element < num_elements; ++element) {
      const auto e = conflicts[element];
      index(element + 1) = index(element) + e.size();
      dim_t i = index(element);
      for (dim_t edge : e) {
        edges(i) = edge;
        ++i;
      }
    }
    conflicts.clear();
    // Color the conflict graph
    std::cout << "Coloring Conflict Graph" << std::endl;
    memory::Array<dim_t, 1> colors = dsatur(graph);
    // Extract the differently colored groups
    std::cout << "Grouping colors" << std::endl;
    std::map<dim_t, dim_t> num_elements_per_color;
    for (dim_t element = 0; element < num_elements; ++element) {
      const dim_t col = colors(element);
      auto it = num_elements_per_color.find(col);
      if (it == num_elements_per_color.end()) {
        num_elements_per_color[col] = 1;
      } else {
        num_elements_per_color[col] += 1;
      }
    }
    const dim_t num_colors = num_elements_per_color.size();
    groups.clear();
    for (dim_t col = 0; col < num_colors; ++col) {
      groups.emplace_back(memory::Shape<1>(num_elements_per_color[col]),
                          memory::MemoryType::HOST);
    }
    std::vector<dim_t> added_elements(num_colors, 0);
    for (dim_t element = 0; element < num_elements; ++element) {
      const dim_t col = colors(element);
      groups[col](added_elements[col]) = element;
      ++added_elements[col];
    }
  }
}

} // namespace internal

} // namespace numeric::math::fes
