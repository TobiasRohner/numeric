#include <numeric/io/vtkhdf_writer.hpp>
#include <numeric/math/fes/basis_h1.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/math/mesh_function.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>

using namespace numeric;

using scalar_t = double;
using el_segment_t = mesh::Segment<1>;
using el_tria_t = mesh::Tria<1>;
using el_quad_t = mesh::Quad<1>;
using el_tetra_t = mesh::Tetra<1>;
using el_cube_t = mesh::Cube<1>;
using mesh_t = mesh::UnstructuredMesh<scalar_t, el_segment_t, el_tria_t,
                                      el_quad_t, el_tetra_t, el_cube_t>;

std::shared_ptr<mesh_t> generate_mesh() {
  static constexpr dim_t world_dim = 3;
  static constexpr dim_t num_vertices =
      el_segment_t::num_nodes + el_tria_t::num_nodes + el_quad_t::num_nodes +
      el_tetra_t::num_nodes + el_cube_t::num_nodes;
  std::shared_ptr<mesh_t> mesh =
      std::make_shared<mesh_t>(world_dim, num_vertices, 1, 1, 1, 1, 1);
  dim_t idx = 0;
  scalar_t offset = 0;
  mesh_t::for_all_element_types(
      [&]<typename ElementType>(meta::type_tag<ElementType>) {
        auto vertices = mesh->vertices();
        auto elements = mesh->template get_elements<ElementType>();
        scalar_t ref_nodes[ElementType::num_nodes][ElementType::dim];
        ElementType::get_ref_nodes(ref_nodes);
        for (dim_t node = 0; node < ElementType::num_nodes; ++node) {
          vertices(0, idx) = offset + ref_nodes[node][0];
          if (ElementType::dim > 1) {
            vertices(1, idx) = ref_nodes[node][1];
          } else {
            vertices(1, idx) = 0;
          }
          if (ElementType::dim > 2) {
            vertices(2, idx) = ref_nodes[node][2];
          } else {
            vertices(2, idx) = 0;
          }
          elements(node, 0) = idx;
          ++idx;
        }
        offset += 2;
      });
  return mesh;
}

template <dim_t Order> void write_basis() {
  using basis_t = math::fes::BasisH1<Order>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  static constexpr dim_t max_num_dofs =
      basis_t::template num_basis_functions<typename el_cube_t::ref_el_t>();
  const auto mesh = generate_mesh();
  auto fes = std::make_shared<fes_t>(mesh);
  math::MeshFunction<scalar_t, fes_t> basis(fes);
  io::VTKHDFWriter<Order, io::VTKHDFFunctionSpaceType::CONTINUOUS, mesh_t>
      writer("vtkhdf_lagrange_basis_" + std::to_string(Order) + ".vtkhdf",
             mesh);
  auto mf_dofs = basis.dofs();
  for (dim_t dof = 0; dof < max_num_dofs; ++dof) {
    mf_dofs = 0;
    mesh_t::for_all_element_types(
        [&]<typename ElementType>(meta::type_tag<ElementType>) {
          using ref_el_t = typename ElementType::ref_el_t;
          static constexpr dim_t num_dofs =
              basis_t::template num_basis_functions<ref_el_t>();
          if (dof < num_dofs) {
            const auto dof_map = fes->template dof_map<ElementType>();
            mf_dofs(dof_map(dof, 0)) = 1;
          }
        });
    writer.write("basis_" + std::to_string(dof), basis);
  }
}

int main(int argc, char *argv[]) {
  write_basis<1>();
  write_basis<2>();
  write_basis<3>();
  write_basis<4>();
  write_basis<5>();

  return 0;
}
