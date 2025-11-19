#include <numeric/io/vtkhdf_writer.hpp>
#include <numeric/math/functions.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>

using namespace numeric;

using scalar_t = double;

template <dim_t Order>
using mesh_t = mesh::UnstructuredMesh<scalar_t, mesh::Segment<Order>,
                                      mesh::Tria<Order>, mesh::Quad<Order>>;

template <dim_t Order> std::shared_ptr<mesh_t<Order>> generate_mesh() {
  using segment_t = mesh::Segment<Order>;
  using tria_t = mesh::Tria<Order>;
  using quad_t = mesh::Quad<Order>;
  static constexpr dim_t world_dim = 3;
  static constexpr dim_t num_vertices = math::pow<2>(segment_t::num_nodes) +
                                        math::pow<2>(tria_t::num_nodes) +
                                        math::pow<2>(quad_t::num_nodes);
  std::shared_ptr<mesh_t<Order>> mesh = std::make_shared<mesh_t<Order>>(
      world_dim, num_vertices, memory::MemoryType::HOST, segment_t::num_nodes,
      tria_t::num_nodes, quad_t::num_nodes);
  auto vertices = mesh->vertices();
  auto segments = mesh->template get_elements<segment_t>();
  auto trias = mesh->template get_elements<tria_t>();
  auto quads = mesh->template get_elements<quad_t>();
  scalar_t ref_nodes_segment[segment_t::num_nodes][segment_t::dim];
  scalar_t ref_nodes_tria[tria_t::num_nodes][tria_t::dim];
  scalar_t ref_nodes_quad[quad_t::num_nodes][quad_t::dim];
  segment_t::get_ref_nodes(ref_nodes_segment);
  tria_t::get_ref_nodes(ref_nodes_tria);
  quad_t::get_ref_nodes(ref_nodes_quad);
  for (dim_t segment = 0; segment < segment_t::num_nodes; ++segment) {
    for (dim_t node = 0; node < segment_t::num_nodes; ++node) {
      const dim_t idx = segment_t::num_nodes * segment + node;
      vertices(0, idx) = 2 * segment + ref_nodes_segment[node][0];
      vertices(1, idx) = 0;
      vertices(2, idx) = segment == node ? 1 : 0;
      segments(node, segment) = idx;
    }
  }
  for (dim_t tria = 0; tria < tria_t::num_nodes; ++tria) {
    for (dim_t node = 0; node < tria_t::num_nodes; ++node) {
      const dim_t idx =
          math::pow<2>(segment_t::num_nodes) + tria_t::num_nodes * tria + node;
      vertices(0, idx) = 2 * tria + ref_nodes_tria[node][0];
      vertices(1, idx) = -2 + ref_nodes_tria[node][1];
      vertices(2, idx) = tria == node ? 1 : 0;
      trias(node, tria) = idx;
    }
  }
  for (dim_t quad = 0; quad < quad_t::num_nodes; ++quad) {
    for (dim_t node = 0; node < quad_t::num_nodes; ++node) {
      const dim_t idx = math::pow<2>(segment_t::num_nodes) +
                        math::pow<2>(tria_t::num_nodes) +
                        quad_t::num_nodes * quad + node;
      vertices(0, idx) = 2 * quad + ref_nodes_quad[node][0];
      vertices(1, idx) = -4 + ref_nodes_quad[node][1];
      vertices(2, idx) = quad == node ? 1 : 0;
      quads(node, quad) = idx;
    }
  }
  return mesh;
}

template <dim_t Order> void write_mesh() {
  io::VTKHDFWriter<Order, io::VTKHDFFunctionSpaceType::CONTINUOUS,
                   mesh_t<Order>>
      writer("vtkhdf_mesh_writer_" + std::to_string(Order) + ".vtkhdf",
             generate_mesh<Order>());
}

int main(int argc, char *argv[]) {
  write_mesh<1>();
  write_mesh<2>();
  write_mesh<3>();
  write_mesh<4>();
  write_mesh<5>();

  return 0;
}
