#include <numeric/io/gmsh_reader.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/subelement_relation.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/utils/tuple.hpp>

using namespace numeric;

template <typename Element>
void print_elements(const memory::ArrayConstView<dim_t, 2> &elements) {
  std::cout << Element::name << '\n';
  for (dim_t node = 0; node < Element::num_nodes; ++node) {
    for (dim_t element = 0; element < elements.shape(1); ++element) {
      std::cout << '\t' << elements(node, element);
    }
    std::cout << '\n';
  }
}

template <typename Scalar, typename... ElementTypes>
void print_mesh(const mesh::UnstructuredMesh<Scalar, ElementTypes...> &mesh) {
  std::cout << mesh.world_dim() << "D mesh with " << mesh.num_vertices()
            << " vertices\n";
  std::cout << "Vertices:\n";
  for (dim_t dim = 0; dim < mesh.world_dim(); ++dim) {
    for (dim_t vertex = 0; vertex < mesh.num_vertices(); ++vertex) {
      std::cout << '\t' << mesh.vertices()(dim, vertex);
    }
    std::cout << '\n';
  }
  (print_elements<ElementTypes>(mesh.template get_elements<ElementTypes>()),
   ...);
}

template <typename Element, typename Subelement>
void print_subelements(const memory::ArrayConstView<dim_t, 2> &subelements) {
  std::cout << Element::name << " -> " << Subelement::name << '\n';
  for (dim_t node = 0; node < subelements.shape(0); ++node) {
    for (dim_t subelement = 0; subelement < subelements.shape(1);
         ++subelement) {
      std::cout << '\t' << subelements(node, subelement);
    }
    std::cout << '\n';
  }
}

int main(int argc, char *argv[]) {
  const dim_t world_dim = 2;
  const auto mesh = io::GmshReader<double, mesh::Tria<1>, mesh::Quad<1>>::load(
      argv[1], world_dim);

  const auto [points, relations_points] =
      mesh::subelement_relation<mesh::Point<1>>(mesh);
  const auto [segments, relations_segments] =
      mesh::subelement_relation<mesh::Segment<1>>(mesh);

  print_mesh(mesh);
  std::cout << "Found the following " << points.shape(1) << " points\n";
  print_elements<mesh::Point<1>>(points);
  print_subelements<mesh::Tria<1>, mesh::Point<1>>(
      relations_points.template get<mesh::Tria<1>>());
  print_subelements<mesh::Quad<1>, mesh::Point<1>>(
      relations_points.template get<mesh::Quad<1>>());
  std::cout << "Found the following " << segments.shape(1) << " segments\n";
  print_elements<mesh::Segment<1>>(segments);
  print_subelements<mesh::Tria<1>, mesh::Segment<1>>(
      relations_segments.template get<mesh::Tria<1>>());
  print_subelements<mesh::Quad<1>, mesh::Segment<1>>(
      relations_segments.template get<mesh::Quad<1>>());

  return 0;
}
