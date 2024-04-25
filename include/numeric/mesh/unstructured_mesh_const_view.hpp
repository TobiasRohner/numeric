#ifndef NUMERIC_MESH_UNSTRUCTURED_MESH_CONST_VIEW_HPP_
#define NUMERIC_MESH_UNSTRUCTURED_MESH_CONST_VIEW_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/element_list_view.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/mesh/unstructured_mesh_base.hpp>
#include <numeric/mesh/unstructured_mesh_view.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::mesh {

template <typename Scalar, typename... ElementTypes>
class UnstructuredMeshConstView
    : public UnstructuredMeshConstViewBase<
          UnstructuredMeshConstView<Scalar, ElementTypes...>> {
  using super = UnstructuredMeshConstViewBase<
      UnstructuredMeshConstView<Scalar, ElementTypes...>>;

public:
  UnstructuredMeshConstView() = default;

  template <typename... NumElementTs>
  UnstructuredMeshConstView(const UnstructuredMesh<Scalar, ElementTypes> &mesh)
      : vertices_(mesh.vertices()),
        elements_(mesh.template get_elements<ElementTypes>()...) {}
  UnstructuredMeshConstView(
      const UnstructuredMeshView<Scalar, ElementTypes> &mesh)
      : vertices_(mesh.vertices()),
        elements_(mesh.template get_elements<ElementTypes>()...) {}
  UnstructuredMeshConstView(const UnstructuredMeshConstView &) = default;
  UnstructuredMeshConstView(UnstructuredMeshConstView &&) = default;
  UnstructuredMeshConstView &
  operator=(const UnstructuredMeshConstView &) = default;
  UnstructuredMeshConstView &operator=(UnstructuredMeshConstView &&) = default;

  memory::ArrayConstView<Scalar, 2> vertices() const noexcept {
    return vertices_;
  }

  template <typename ElementType>
  const ElementListConstView<ElementType> &get_elements() const noexcept {
    return elements_.template get<ElementListConstView<ElementType>>();
  }

private:
  memory::ArrayConstView<Scalar, 2> vertices_;
  utils::Tuple<ElementListConstView<ElementTypes>...> elements_;
};

} // namespace numeric::mesh

#endif
