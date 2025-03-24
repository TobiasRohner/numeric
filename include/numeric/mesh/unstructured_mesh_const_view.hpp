#ifndef NUMERIC_MESH_UNSTRUCTURED_MESH_CONST_VIEW_HPP_
#define NUMERIC_MESH_UNSTRUCTURED_MESH_CONST_VIEW_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/mesh/unstructured_mesh_base.hpp>
#include <numeric/mesh/unstructured_mesh_view.hpp>
#include <numeric/meta/type_tag.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::mesh {

template <typename Scalar, typename... ElementTypes>
class UnstructuredMeshConstView
    : public UnstructuredMeshBase<
          UnstructuredMeshConstView<Scalar, ElementTypes...>> {
  using super =
      UnstructuredMeshBase<UnstructuredMeshConstView<Scalar, ElementTypes...>>;

public:
  UnstructuredMeshConstView() = default;

  template <typename... NumElementTs>
  UnstructuredMeshConstView(
      const UnstructuredMesh<Scalar, ElementTypes...> &mesh)
      : vertices_(mesh.vertices()),
        elements_(mesh.template get_elements<ElementTypes>()...) {}
  UnstructuredMeshConstView(
      const UnstructuredMeshView<Scalar, ElementTypes...> &mesh)
      : vertices_(mesh.vertices()),
        elements_(mesh.template get_elements<ElementTypes>()...) {}
  UnstructuredMeshConstView(const UnstructuredMeshConstView &) = default;
  UnstructuredMeshConstView(UnstructuredMeshConstView &&) = default;
  UnstructuredMeshConstView &
  operator=(const UnstructuredMeshConstView &) = default;
  UnstructuredMeshConstView &operator=(UnstructuredMeshConstView &&) = default;

  template <typename Func> static void for_all_element_types(Func &&f) {
    ((f(meta::type_tag<ElementTypes>{}), false) || ...);
  }

  memory::ArrayConstView<Scalar, 2> vertices() const noexcept {
    return vertices_;
  }

  template <typename ElementType>
  memory::ArrayConstView<dim_t, 2> get_elements() const noexcept {
    return elements_.template get<ElementType>();
  }

private:
  memory::ArrayConstView<Scalar, 2> vertices_;
  utils::TypeIndexedMap<memory::ArrayConstView<dim_t, 2>, ElementTypes...>
      elements_;
};

} // namespace numeric::mesh

#endif
