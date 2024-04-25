#ifndef NUMERIC_MESH_ELEMENT_LIST_CONST_VIEW_HPP_
#define NUMERIC_MESH_ELEMENT_LIST_CONST_VIEW_HPP_

#include <numeric/mesh/element_list.hpp>
#include <numeric/mesh/element_list_base.hpp>
#include <numeric/mesh/element_list_traits.hpp>
#include <numeric/mesh/element_list_view.hpp>

namespace numeric::mesh {

template <typename ElementType>
class ElementListConstView
    : public ElementListBase<ElementListConstView<ElementType>> {
  using super = ElementListBase<ElementListConstView<ElementType>>;

public:
  using element_t = ElementType;

  ElementListConstView() = default;
  ElementListConstView(const ElementList<ElementType> &list)
      : elements_(list.indices()) {}
  ElementListConstView(const ElementListView<ElementType> &list)
      : elements_(list.indices()) {}
  ElementListConstView(const ElementListConstView &) = default;
  ElementListConstView(ElementListConstView &&) = default;
  ElementListConstView &operator=(const ElementListConstView &) = default;
  ElementListConstView &operator=(ElementListConstView &&) = default;

  memory::ArrayConstView<dim_t, 2> indices() const noexcept {
    return elements_;
  }

  using super::num_elements;
  using super::num_nodes_per_element;

private:
  memory::ArrayConstView<dim_t, 2> elements_;
};

template <typename ElementType>
struct ElementListTraits<ElementListConstView<ElementType>> {
  using element_t = ElementType;
};

} // namespace numeric::mesh

#endif
