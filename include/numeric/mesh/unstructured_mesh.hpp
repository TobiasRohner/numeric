#ifndef NUMERIC_MESH_UNSTRUCTURED_MESH_HPP_
#define NUMERIC_MESH_UNSTRUCTURED_MESH_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/unstructured_mesh_base.hpp>
#include <numeric/meta/type_tag.hpp>
#include <numeric/utils/tuple.hpp>
#include <numeric/utils/type_indexed_map.hpp>

namespace numeric::mesh {

template <typename Scalar, typename... ElementTypes>
class UnstructuredMesh
    : public UnstructuredMeshBase<UnstructuredMesh<Scalar, ElementTypes...>> {
  using super = UnstructuredMeshBase<UnstructuredMesh<Scalar, ElementTypes...>>;

public:
  UnstructuredMesh() = default;

  template <typename... NumElementTs>
  UnstructuredMesh(dim_t world_dim, dim_t num_vertices,
                   memory::MemoryType memory_type, NumElementTs... num_elements)
      : vertices_(memory::Shape<2>(world_dim, num_vertices), memory_type),
        elements_{memory::Array<dim_t, 2>(
            memory::Shape<2>(ElementTypes::num_nodes, num_elements),
            memory_type)...} {
    static_assert(
        sizeof...(ElementTypes) == sizeof...(NumElementTs),
        "Length of num_elements does not match number of element types");
  }
  UnstructuredMesh(const UnstructuredMesh &) = delete;
  UnstructuredMesh(UnstructuredMesh &&) = default;
  UnstructuredMesh &operator=(const UnstructuredMesh &) = delete;
  UnstructuredMesh &operator=(UnstructuredMesh &&) = default;

  void to(memory::MemoryType memory_type) {
    vertices_.to(memory_type);
    (elements_.template get<ElementTypes>().to(memory_type), ...);
  }

  template <typename Func> static void for_all_element_types(Func &&f) {
    ((f(meta::type_tag<ElementTypes>{}), false) || ...);
  }

  using super::memory_type;
  using super::num_elements;
  using super::num_vertices;
  using super::world_dim;

  memory::ArrayConstView<Scalar, 2> vertices() const noexcept {
    return vertices_;
  }
  memory::ArrayView<Scalar, 2> vertices() noexcept { return vertices_; }

  template <typename ElementType>
  memory::ArrayConstView<dim_t, 2> get_elements() const noexcept {
    return elements_.template get<ElementType>();
  }
  template <typename ElementType>
  memory::ArrayView<dim_t, 2> get_elements() noexcept {
    return elements_.template get<ElementType>();
  }

  void
  reset_vertices(dim_t world_dim, dim_t num_verts,
                 memory::MemoryType memory_type = memory::MemoryType::HOST) {
    vertices_ = std::move(memory::Array<Scalar, 2>(
        memory::Shape<2>(world_dim, num_verts), memory_type));
  }

  template <typename ElementType>
  void
  reset_elements(dim_t num_elements,
                 memory::MemoryType memory_type = memory::MemoryType::HOST) {
    elements_.template get<ElementType>() = memory::Array<dim_t, 2>(
        memory::Shape<2>(ElementType::num_nodes, num_elements), memory_type);
  }

private:
  memory::Array<Scalar, 2> vertices_;
  utils::TypeIndexedMap<memory::Array<dim_t, 2>, ElementTypes...> elements_;
};

} // namespace numeric::mesh

#endif
