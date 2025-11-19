#ifndef NUMERIC_MESH_UNSTRUCTURED_MESH_VIEW_HPP_
#define NUMERIC_MESH_UNSTRUCTURED_MESH_VIEW_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/mesh/unstructured_mesh_base.hpp>
#include <numeric/meta/type_tag.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::mesh {

template <typename Scalar, typename... ElementTypes>
class UnstructuredMeshView
    : public UnstructuredMeshBase<
          UnstructuredMeshView<Scalar, ElementTypes...>> {
  using super =
      UnstructuredMeshBase<UnstructuredMeshView<Scalar, ElementTypes...>>;

public:
  UnstructuredMeshView() = default;

  template <typename... NumElementTs>
  UnstructuredMeshView(UnstructuredMesh<Scalar, ElementTypes...> &mesh)
      : vertices_(mesh.vertices()),
        elements_(mesh.template get_elements<ElementTypes>()...) {}
  UnstructuredMeshView(const UnstructuredMeshView &) = default;
  UnstructuredMeshView(UnstructuredMeshView &&) = default;
  UnstructuredMeshView &operator=(const UnstructuredMeshView &) = default;
  UnstructuredMeshView &operator=(UnstructuredMeshView &&) = default;

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

private:
  memory::ArrayView<Scalar, 2> vertices_;
  utils::TypeIndexedMap<memory::ArrayView<dim_t, 2>, ElementTypes...> elements_;
};

} // namespace numeric::mesh

#endif
