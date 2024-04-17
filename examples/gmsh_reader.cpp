#include <numeric/io/gmsh_reader.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/utils/tuple.hpp>

using namespace numeric;

int main(int argc, char *argv[]) {
  const dim_t world_dim = 2;
  const auto mesh = io::GmshReader<double, mesh::Tria<1>, mesh::Quad<1>>::load(
      argv[1], world_dim);

  const auto nodes = mesh.vertices();
  for (dim_t node = 0; node < nodes.shape(1); ++node) {
    for (dim_t i = 0; i < world_dim; ++i) {
      std::cout << nodes(i, node) << ' ';
    }
    std::cout << std::endl;
  }

  auto print_elements = [&]<typename Element>() {
    std::cout << to_string(io::to_gmsh_element_type_v<Element>) << std::endl;
    const auto elements = mesh.template get_elements<Element>().elements();
    for (dim_t element = 0; element < elements.shape(1); ++element) {
      std::cout << "  ";
      for (dim_t i = 0; i < elements.shape(0); ++i) {
        std::cout << elements(i, element) << ' ';
      }
      std::cout << std::endl;
    }
  };
  print_elements.template operator()<mesh::Tria<1>>();
  print_elements.template operator()<mesh::Quad<1>>();

  return 0;
}
