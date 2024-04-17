#ifndef NUMERIC_MESH_UNSTRUCTURED_MESH_BASE_HPP_
#define NUMERIC_MESH_UNSTRUCTURED_MESH_BASE_HPP_

namespace numeric::mesh {

template <typename Derived> class UnstructuredMeshBase {
private:
  const Derived &derived() const noexcept {
    return static_cast<const Derived &>(*this);
  }
  Derived &derived() noexcept { return static_cast<Derived &>(*this); }
};

} // namespace numeric::mesh

#endif
