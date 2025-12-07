#include <map>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/math/graph/coloring.hpp>
#include <numeric/math/graph/sparse_graph.hpp>
#include <numeric/math/reduce.hpp>
#include <numeric/memory/linspace.hpp>
#include <set>
#if NUMERIC_ENABLE_HIP
#include <numeric/hip/device.hpp>
#include <numeric/hip/program.hpp>
#endif

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
    memory::Array<dim_t, 1> colors = dsatur(graph);
    // Extract the differently colored groups
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

void optimize_memory_layout_elements_host(
    memory::ArrayView<dim_t, 2> elements, memory::ArrayView<dim_t, 2> dofs,
    const std::vector<memory::Array<dim_t, 1>> &groups) {
  // TODO: Do something here
  std::cout
      << "WARNING: Optimizing memory layout for the host currently does nothing"
      << std::endl;
}

static const char kernel_optimize_memory_layout_elements_src[] = R"(
  #include <numeric/memory/array_const_view.hpp>
  #include <numeric/memory/array_view.hpp>

  __global__ void optimize_memory_layout_elements(
      numeric::memory::ArrayConstView<numeric::dim_t, 1> group,
      numeric::memory::ArrayConstView<numeric::dim_t, 2> elements_in,
      numeric::memory::ArrayView<numeric::dim_t, 2> elements_out,
      numeric::memory::ArrayConstView<numeric::dim_t, 2> dofs_in,
      numeric::memory::ArrayView<numeric::dim_t, 2> dofs_out) {
    const numeric::dim_t tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    
    const numeric::dim_t num_nodes = elements_out.shape(0);
    const numeric::dim_t num_elements = elements_out.shape(1);
    const numeric::dim_t num_dofs = dofs_out.shape(0);

    if (tid >= num_elements) {
      return;
    }

    for (numeric::dim_t i = 0 ; i < num_nodes ; ++i) {
      elements_out(i, tid) = elements_in(i, group(tid));
    }
    for (numeric::dim_t i = 0 ; i < num_dofs ; ++i) {
      dofs_out(i, tid) = dofs_in(i, group(tid));
    }
  }
)";

static hip::Kernel build_optimize_memory_layout_elements_kernel() {
  const char kernel_name[] = "optimize_memory_layout_elements";
  hip::Program program(kernel_optimize_memory_layout_elements_src);
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

void optimize_memory_layout_elements_device(
    memory::ArrayView<dim_t, 2> elements, memory::ArrayView<dim_t, 2> dofs,
    const std::vector<memory::Array<dim_t, 1>> &groups) {
  static const hip::Kernel kernel =
      build_optimize_memory_layout_elements_kernel();
  hip::Device device;
  memory::Array<dim_t, 2> elements_new(elements.shape(),
                                       elements.memory_type());
  memory::Array<dim_t, 2> dofs_new(dofs.shape(), dofs.memory_type());
  dim_t offset = 0;
  for (dim_t i = 0; i < groups.size(); ++i) {
    memory::ArrayView<dim_t, 1> group = groups[i];
    memory::ArrayView<dim_t, 2> elements_out = elements_new(
        memory::Slice(), memory::Slice(offset, offset + group.shape(0)));
    memory::ArrayView<dim_t, 2> dofs_out = dofs_new(
        memory::Slice(), memory::Slice(offset, offset + group.shape(0)));
    const hip::LaunchParams lp =
        device.launch_params_for_grid(group.shape(0), 1, 1);
    kernel(lp, hip::Stream(device), group, elements, elements_out, dofs,
           dofs_out);
    group = memory::linspace(offset, offset + group.shape(0), group.shape(0),
                             false, group.memory_type());
    offset += group.shape(0);
  }
  elements = elements_new;
  dofs = dofs_new;
}

void optimize_memory_layout_elements(
    memory::ArrayView<dim_t, 2> elements, memory::ArrayView<dim_t, 2> dofs,
    const std::vector<memory::Array<dim_t, 1>> &groups) {
  if (is_host_accessible(elements.memory_type())) {
    optimize_memory_layout_elements_host(elements, dofs, groups);
  }
#if NUMERIC_ENABLE_HIP
  else if (is_device_accessible(elements.memory_type())) {
    optimize_memory_layout_elements_device(elements, dofs, groups);
  }
#endif
  else {
    NUMERIC_ERROR("Unsupported memory type \"{}\"",
                  to_string(elements.memory_type()));
  }
}

} // namespace internal

} // namespace numeric::math::fes
