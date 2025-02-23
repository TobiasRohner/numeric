#ifndef NUMERIC_IO_VTKHDF_WRITER_HPP_
#define NUMERIC_IO_VTKHDF_WRITER_HPP_

#include <numeric/io/hdf5_file.hpp>
#include <numeric/io/vtk_lagrange_element.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/memory/shape.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/meta/meta.hpp>
#include <string_view>

namespace numeric::io {

template <typename FES> class VTKHDFWriter {
  static_assert(!meta::is_same_v<FES, FES>,
                "Unsupported Finite Element Space for VTKHDFWriter");
};

template <typename Scalar, typename Basis, typename... ElementTypes>
class VTKHDFWriter<math::fes::FESpace<
    Basis, mesh::UnstructuredMesh<Scalar, ElementTypes...>>> {
public:
  using scalar_t = Scalar;
  using basis_t = Basis;
  using mesh_t = mesh::UnstructuredMesh<scalar_t, ElementTypes...>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;

  VTKHDFWriter(std::string_view path, const fes_t &fes) {
    auto file = io::HDF5File::create(path);
    VTKHDF_ = file->create_group("VTKHDF");
    VTKHDF_->write_attribute("Type", "UnstructuredGrid");
    VTKHDF_->write_attribute("Version", std::vector<int>{2, 2});
    write_mesh(fes);
  }

private:
  std::shared_ptr<HDF5Group> VTKHDF_;

  void write_mesh(const fes_t &fes) {
    const mesh_t &mesh = *fes.mesh();
    const dim_t num_vertices = fes.num_dofs();
    const dim_t num_elements =
        (mesh.template num_elements<ElementTypes>() + ...);
    const dim_t num_conn_ids = ((VTKLagrangeElement<ElementTypes>::num_nodes *
                                 mesh.template num_elements<ElementTypes>()) +
                                ...);
    auto NumberOfConnectivityIds = VTKHDF_->create_variable<dim_t>(
        "NumberOfConnectivityIds", memory::Shape<1>(1));
    auto NumberOfPoints =
        VTKHDF_->create_variable<dim_t>("NumberOfPoints", memory::Shape<1>(1));
    auto NumberOfCells =
        VTKHDF_->create_variable<dim_t>("NumberOfCells", memory::Shape<1>(1));
    auto Points = VTKHDF_->create_variable<scalar_t>(
        "Points", memory::Shape<2>(num_vertices, 3));
    auto Types = VTKHDF_->create_variable<unsigned char>(
        "Types", memory::Shape<1>(num_elements));
    auto Connectivity = VTKHDF_->create_variable<dim_t>(
        "Connectivity", memory::Shape<1>(num_conn_ids));
    auto Offsets = VTKHDF_->create_variable<dim_t>(
        "Offsets", memory::Shape<1>(num_elements + 1));
    memory::Array<scalar_t, 2> points(memory::Shape<2>(num_vertices, 3),
                                      memory::MemoryType::HOST);
    memory::Array<unsigned char, 1> types(memory::Shape<1>(num_elements),
                                          memory::MemoryType::HOST);
    memory::Array<dim_t, 1> connectivity(memory::Shape<1>(num_conn_ids),
                                         memory::MemoryType::HOST);
    memory::Array<dim_t, 1> offsets(memory::Shape<1>(num_elements + 1),
                                    memory::MemoryType::HOST);
    dim_t type_start = 0;
    dim_t connectivity_start = 0;
    offsets(0) = 0;
    ((
         [&]<typename ElementType>() {
           using vtk_el_t = VTKLagrangeElement<ElementType>;
           const dim_t nel = mesh.template num_elements<ElementType>();
           const auto dofmap = fes.template dof_map<ElementType>();
           const auto vertices = mesh.vertices();
           const auto elements = mesh.template get_elements<ElementType>();
           types(memory::Slice(type_start, type_start + nel)) = vtk_el_t::type;
           for (dim_t element = 0; element < nel; ++element) {
             scalar_t element_nodes[ElementType::dim][vtk_el_t::num_nodes];
             for (dim_t i = 0; i < ElementType::dim; ++i) {
               for (dim_t j = 0; j < vtk_el_t::num_nodes; ++j) {
                 element_nodes[i][j] = vertices(i, elements(j, element));
               }
             }
             for (dim_t dof = 0; dof < vtk_el_t::num_nodes; ++dof) {
               const dim_t dofidx = dofmap(dof, element);
               scalar_t node_loc[ElementType::dim];
               scalar_t node[ElementType::dim];
               VTKLagrangeElement<ElementType>::node(dof, node_loc);
               ElementType::local_to_global(element_nodes, node_loc, node,
                                            ElementType::dim);
               points(dofidx, 0) = node[0];
               if (ElementType::dim > 1) {
                 points(dofidx, 1) = node[1];
               } else {
                 points(dofidx, 1) = 0;
               }
               if (ElementType::dim > 2) {
                 points(dofidx, 2) = node[2];
               } else {
                 points(dofidx, 2) = 0;
               }
               connectivity(connectivity_start + vtk_el_t::num_nodes * element +
                            dof) = dofidx;
             }
             offsets(element + 1) = connectivity_start +
                                    vtk_el_t::num_nodes * element +
                                    vtk_el_t::num_nodes;
           }
           type_start += nel;
           connectivity_start += nel * vtk_el_t::num_nodes;
         }.template operator()<ElementTypes>(),
         false) ||
     ...);
    NumberOfConnectivityIds->write(memory::ArrayConstView<dim_t, 1>(
        &num_conn_ids, memory::Shape<1>(1), memory::MemoryType::HOST));
    NumberOfPoints->write(memory::ArrayConstView<dim_t, 1>(
        &num_vertices, memory::Shape<1>(1), memory::MemoryType::HOST));
    NumberOfCells->write(memory::ArrayConstView<dim_t, 1>(
        &num_elements, memory::Shape<1>(1), memory::MemoryType::HOST));
    Points->write(points);
    Types->write(types);
    Connectivity->write(connectivity);
    Offsets->write(offsets);
  }
};

} // namespace numeric::io

#endif
