#ifndef NUMERIC_MESH_UNSTRUCTURED_MESH_HPP_
#define NUMERIC_MESH_UNSTRUCTURED_MESH_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/element_list.hpp>
#include <numeric/mesh/unstructured_mesh_base.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::mesh {

template <typename Scalar, typename... ElementTypes>
class UnstructuredMesh
    : public UnstructuredMeshBase<UnstructuredMesh<Scalar, ElementTypes...>> {
  using super = UnstructuredMeshBase<UnstructuredMesh<Scalar, ElementTypes...>>;

public:
  UnstructuredMesh() = default;

  template <typename... NumElementTs>
  UnstructuredMesh(dim_t world_dim, dim_t num_vertices,
                   NumElementTs... num_elements)
      : vertices_(memory::Shape<2>(world_dim, num_vertices)),
        elements_{ElementList<ElementTypes>(num_elements)...} {
    static_assert(
        sizeof...(ElementTypes) == sizeof...(NumElementTs),
        "Length of num_elements does not match number of element types");
  }
  UnstructuredMesh(const UnstructuredMesh &) = delete;
  UnstructuredMesh(UnstructuredMesh &&) = default;
  UnstructuredMesh &operator=(const UnstructuredMesh &) = delete;
  UnstructuredMesh &operator=(UnstructuredMesh &&) = default;

  memory::ArrayConstView<Scalar, 2> vertices() const noexcept {
    return vertices_;
  }
  memory::ArrayView<Scalar, 2> vertices() noexcept { return vertices_; }

  template <typename ElementType>
  const ElementList<ElementType> &get_elements() const noexcept {
    return elements_.template get<ElementList<ElementType>>();
  }
  template <typename ElementType>
  ElementList<ElementType> &get_elements() noexcept {
    return elements_.template get<ElementList<ElementType>>();
  }

  void set_vertices(memory::Array<Scalar, 2> &&vertices) {
    vertices_ = std::move(vertices);
  }

  template <typename ElementType>
  void set_elements(ElementList<ElementType> &&elements) {
    elements_.template get<ElementList<ElementType>>() = std::move(elements);
  }

private:
  memory::Array<Scalar, 2> vertices_;
  utils::Tuple<ElementList<ElementTypes>...> elements_;
};

} // namespace numeric::mesh

#endif
