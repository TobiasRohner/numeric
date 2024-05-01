#ifndef NUMERIC_IO_GMSH_READER_HPP_
#define NUMERIC_IO_GMSH_READER_HPP_

#include <bit>
#include <fstream>
#include <istream>
#include <map>
#include <numeric/io/gmsh_element_type.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/utils/error.hpp>
#include <optional>
#include <string>
#include <vector>

namespace numeric::io {

class GmshReaderBase {
public:
  enum FileType { ASCII, BINARY };

protected:
  double version_;
  FileType file_type_;
  std::endian endianness_;
  int data_size_;
  bool nodes_are_contiguous_;
  size_t min_node_tag_;
  std::map<size_t, size_t> node_tag_to_idx_map_;

  FileType file_type() const noexcept;
  template <typename T> T read_value_ascii(std::istream &is) const;
  template <typename T> T read_value_binary(std::istream &is) const;
  template <typename T> T read_value(std::istream &is) const;
  template <typename T>
  std::vector<T> read_values_ascii(std::istream &is, size_t N) const;
  template <typename T>
  std::vector<T> read_values_binary(std::istream &is, size_t N) const;
  template <typename T>
  std::vector<T> read_values(std::istream &is, size_t N) const;
  static void consume_whitespace(std::istream &is);
  bool parse_section(std::istream &is);
  bool parse_mesh_format(std::istream &is);
  virtual bool parse_nodes(std::istream &is) = 0;
  virtual bool parse_elements(std::istream &is) = 0;
  bool parse_unknown(std::istream &is, const std::string &name) const;

  size_t node_tag_to_idx(size_t tag);
};

template <typename Scalar, typename... ElementTypes>
class GmshReader : public GmshReaderBase {
  using super = GmshReaderBase;

public:
  using scalar_t = Scalar;
  using mesh_t = mesh::UnstructuredMesh<Scalar, ElementTypes...>;

  static mesh_t load(const std::string &path, dim_t world_dim) {
    return GmshReader(path, world_dim).get_mesh();
  }

protected:
  std::string path_;
  dim_t world_dim_;
  mesh_t mesh_;

  virtual bool parse_nodes(std::istream &is) override {
    const size_t num_entity_blocks = read_value<size_t>(is);
    const size_t num_nodes = read_value<size_t>(is);
    min_node_tag_ = read_value<size_t>(is);
    const size_t max_node_tag = read_value<size_t>(is);
    mesh_.reset_vertices(world_dim_, num_nodes);
    nodes_are_contiguous_ = (max_node_tag - min_node_tag_ + 1 == num_nodes);
    for (size_t block = 0; block < num_entity_blocks; ++block) {
      const int entity_dim = read_value<int>(is);
      const int entity_tag = read_value<int>(is);
      const bool parametric = read_value<int>(is);
      NUMERIC_ERROR_IF(
          parametric,
          "GmshReader does not support parametric nodes at the moment");
      const size_t num_nodes_in_block = read_value<size_t>(is);
      const std::vector<size_t> node_tags =
          read_values<size_t>(is, num_nodes_in_block);
      const std::vector<double> node_pos = read_values<double>(
          is, num_nodes_in_block * (3 + parametric * entity_dim));
      for (size_t node = 0; node < num_nodes; ++node) {
        const size_t tag = node_tags[node];
        const size_t idx = node_tag_to_idx(tag);
        for (dim_t i = 0; i < world_dim_; ++i) {
          mesh_.vertices()(i, idx) =
              node_pos[node * (3 + parametric * entity_dim) + i];
        }
      }
    }
    consume_whitespace(is);
    std::string end_tag;
    is >> end_tag;
    return end_tag == "$EndNodes";
  }

  virtual bool parse_elements(std::istream &is) override {
    const size_t num_entity_blocks = read_value<size_t>(is);
    const size_t num_elements = read_value<size_t>(is);
    const size_t min_element_tag = read_value<size_t>(is);
    const size_t max_element_tag = read_value<size_t>(is);
    for (size_t block = 0; block < num_entity_blocks; ++block) {
      const int entity_dim = read_value<int>(is);
      const int entity_tag = read_value<int>(is);
      const GmshElementType element_type =
          static_cast<GmshElementType>(read_value<int>(is));
      const size_t num_elements_in_block = read_value<size_t>(is);
      const dim_t num_nodes_per_element = num_nodes(element_type);
      const std::vector<size_t> elements = read_values<size_t>(
          is, num_elements_in_block * (1 + num_nodes_per_element));
      auto add_elements = [&]<typename Element>() {
        mesh_.template reset_elements<Element>(num_elements_in_block);
        for (size_t element = 0; element < num_elements_in_block; ++element) {
          const size_t tag =
              elements[element * (1 + num_nodes_per_element) + 0];
          const size_t idx = element;
          for (size_t i = 0; i < num_nodes_per_element; ++i) {
            mesh_.template get_elements<Element>()(i, idx) = node_tag_to_idx(
                elements[element * (1 + num_nodes_per_element) + i + 1]);
          }
        }
        return true;
      };
      ((to_gmsh_element_type_v<ElementTypes> == element_type &&
        add_elements.template operator()<ElementTypes>()),
       ...);
    }
    consume_whitespace(is);
    std::string end_tag;
    is >> end_tag;
    return end_tag == "$EndElements";
  }

private:
  GmshReader(const std::string &path, dim_t world_dim)
      : path_(path), world_dim_(world_dim) {}

  mesh_t get_mesh() && {
    std::ifstream is(path_.c_str(), std::ios::binary);
    NUMERIC_ERROR_IF(!is.is_open(), "Failed to open file \"{}\"", path_);
    while (is.good()) {
      const bool res = parse_section(is);
      NUMERIC_ERROR_IF(!res, "Failed to parse gmsh file \"{}\"", path_);
    }
    return std::move(mesh_);
  }
};

} // namespace numeric::io

#endif
