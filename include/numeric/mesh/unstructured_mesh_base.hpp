#ifndef NUMERIC_MESH_UNSTRUCTURED_MESH_BASE_HPP_
#define NUMERIC_MESH_UNSTRUCTURED_MESH_BASE_HPP_

namespace numeric::mesh {

template <typename Derived> class UnstructuredMeshBase {
public:
  dim_t world_dim() const { return derived().vertices().shape(0); }

  dim_t num_vertices() const { return derived().vertices().shape(1); }

  template <typename Element> dim_t num_elements() const {
    return derived().template get_elements<Element>().shape(1);
  }

private:
  const Derived &derived() const noexcept {
    return static_cast<const Derived &>(*this);
  }
  Derived &derived() noexcept { return static_cast<Derived &>(*this); }
};

} // namespace numeric::mesh

#endif
