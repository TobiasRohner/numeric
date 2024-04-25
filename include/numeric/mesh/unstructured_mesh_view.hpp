#ifndef NUMERIC_MESH_UNSTRUCTURED_MESH_VIEW_HPP_
#define NUMERIC_MESH_UNSTRUCTURED_MESH_VIEW_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/element_list_view.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/mesh/unstructured_mesh_base.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::mesh {

template <typename Scalar, typename... ElementTypes>
class UnstructuredMeshView
    : public UnstructuredMeshViewBase<
          UnstructuredMeshView<Scalar, ElementTypes...>> {
  using super =
      UnstructuredMeshViewBase<UnstructuredMeshView<Scalar, ElementTypes...>>;

public:
  UnstructuredMeshView() = default;

  template <typename... NumElementTs>
  UnstructuredMeshView(UnstructuredMesh<Scalar, ElementTypes> &mesh)
      : vertices_(mesh.vertices()),
        elements_(mesh.template get_elements<ElementTypes>()...) {}
  UnstructuredMeshView(const UnstructuredMeshView &) = default;
  UnstructuredMeshView(UnstructuredMeshView &&) = default;
  UnstructuredMeshView &operator=(const UnstructuredMeshView &) = default;
  UnstructuredMeshView &operator=(UnstructuredMeshView &&) = default;

  memory::ArrayConstView<Scalar, 2> vertices() const noexcept {
    return vertices_;
  }
  memory::ArrayView<Scalar, 2> vertices() noexcept { return vertices_; }

  template <typename ElementType>
  const ElementListView<ElementType> &get_elements() const noexcept {
    return elements_.template get<ElementListView<ElementType>>();
  }
  template <typename ElementType>
  ElementListView<ElementType> &get_elements() noexcept {
    return elements_.template get<ElementListView<ElementType>>();
  }

  void set_vertices(memory::Array<Scalar, 2> &vertices) {
    vertices_.set(vertices);
  }

  template <typename ElementType>
  void set_elements(ElementListView<ElementType> &elements) {
    elements_.template get<ElementListView<ElementType>>() =
        std::move(elements);
  }

private:
  memory::ArrayView<Scalar, 2> vertices_;
  utils::Tuple<ElementListView<ElementTypes>...> elements_;
};

} // namespace numeric::mesh

#endif
