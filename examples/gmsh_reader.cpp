/**
 * @page gmsh_reader_tutorial Reading and Printing Mesh Data from Gmsh Files
 *
 * @section intro_sec Introduction
 *
 * This tutorial demonstrates how to use the **Numeric** library to read mesh
 * data from a Gmsh file and extract information about the mesh's vertices
 * and elements. In this example, we:
 *
 * - Load an unstructured mesh from a Gmsh file.
 * - Retrieve the vertex coordinates.
 * - Print the vertex data.
 * - Retrieve and print element connectivities for different element types,
 *   such as triangles and quadrilaterals.
 *
 * The complete source code is provided below:
 *
 * @snippet gmsh_reader.cpp full_example
 *
 * @section breakdown_sec Code Breakdown
 *
 * The code is organized into the following sections:
 *
 * @subsection load_mesh_sec 1. Loading the Mesh
 *
 * We begin by loading the mesh from a Gmsh file whose name is passed on the
 * command line. The mesh is read as an unstructured mesh with a given world
 * dimension (here set to 2).
 *
 * @snippet gmsh_reader.cpp load_mesh
 *
 * @subsection print_nodes_sec 2. Printing Node Coordinates
 *
 * After loading the mesh, the node (vertex) coordinates are retrieved.
 * The code then loops over each node and prints its coordinates.
 *
 * @snippet gmsh_reader.cpp print_nodes
 *
 * @subsection print_elements_sec 3. Printing Element Connectivity
 *
 * A templated lambda function is defined to print the connectivity for
 * different element types (e.g., triangles and quadrilaterals). For each
 * element type, the function prints the Gmsh element type (using a conversion
 * function) and then loops through all elements of that type, printing their
 * vertex indices.
 *
 * @snippet gmsh_reader.cpp print_elements
 *
 * @section further_sec Further Exploration
 *
 * - Experiment with other element types supported by the Numeric mesh module.
 * - Use the extracted mesh data for further processing or visualization.
 */

//! [full_example]
#include <iostream>
#include <numeric/io/gmsh_reader.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/utils/tuple.hpp>

using namespace numeric;

int main(int argc, char *argv[]) {

  //! [load_mesh]
  // Check if a filename was provided on the command line.
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mesh_file.msh>" << std::endl;
    return 1;
  }
  // Define the world dimension (2D mesh)
  const dim_t world_dim = 2;
  // Load the mesh from the provided Gmsh file. Here, the mesh is read with
  // element types:
  // - mesh::Tria<1> for triangular elements, and
  // - mesh::Quad<1> for quadrilateral elements.
  const auto mesh = io::GmshReader<double, mesh::Tria<1>, mesh::Quad<1>>::load(
      argv[1], world_dim);
  //! [load_mesh]

  //! [print_nodes]
  // Retrieve the nodes (vertices) from the mesh.
  const auto nodes = mesh->vertices();
  // Loop over each node and print its coordinates.
  for (dim_t node = 0; node < nodes.shape(1); ++node) {
    for (dim_t i = 0; i < world_dim; ++i) {
      std::cout << nodes(i, node) << ' ';
    }
    std::cout << std::endl;
  }
  //! [print_nodes]

  //! [print_elements]
  // Define a templated lambda function to print elements of a given type.
  auto print_elements = [&]<typename Element>() {
    // Print the element type (converted to its Gmsh representation)
    std::cout << to_string(io::to_gmsh_element_type_v<Element>) << std::endl;
    // Retrieve elements of the specified type from the mesh.
    const auto elements = mesh->template get_elements<Element>();
    // Loop over each element and print its connectivity (vertex indices).
    for (dim_t element = 0; element < elements.shape(1); ++element) {
      std::cout << "  ";
      for (dim_t i = 0; i < elements.shape(0); ++i) {
        std::cout << elements(i, element) << ' ';
      }
      std::cout << std::endl;
    }
  };
  // Print elements for triangles.
  print_elements.template operator()<mesh::Tria<1>>();
  // Print elements for quadrilaterals.
  print_elements.template operator()<mesh::Quad<1>>();
  //! [print_elements]

  return 0;
}
//! [full_example]
