#ifndef NUMERIC_MESH_ELEMENT_BASE_HPP_
#define NUMERIC_MESH_ELEMENT_BASE_HPP_

#include <numeric/config.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::mesh {

template <typename Derived> struct ElementBase {
  using traits_t = ElementTraits<Derived>;
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;

  template <typename Base, typename Element>
  using has_subelement_t =
      decltype(Base::num_subelements(meta::type_tag<Element>()));

  template <typename Base, typename Element>
  using has_subelement_node_idxs_t =
      decltype(Base::subelement_node_idxs(meta::declval<dim_t>(),
                                          meta::declval<dim_t *>(),
                                          meta::type_tag<Element>()));

  template <typename Element> static constexpr dim_t num_subelements() {
    if constexpr (meta::is_detected_v<has_subelement_t, Derived, Element>) {
      return Derived::num_subelements(meta::type_tag<Element>());
    } else {
      return 0;
    }
  }

  template <typename Element>
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {
    if constexpr (meta::is_detected_v<has_subelement_node_idxs_t, Derived,
                                      Element>) {
      Derived::subelement_node_idxs(subelement, idxs,
                                    meta::type_tag<Element>());
    }
  }

  /**
   * @brief Computes the integration element (determinant of the Jacobian
   * matrix) for a given set of nodes and coordinates.
   *
   * This function calculates the integration element, which is essential in
   * numerical integration over elements in finite element methods.
   *
   * @tparam Scalar The type of the scalar values (e.g., float, double).
   * @param[in] nodes An array of pointers to the node coordinates. Each node is
   * an array of coordinates with a size equal to `world_dim`.
   * @param[in] x A pointer to the coordinates where the integration element is
   * evaluated.
   * @param[in] world_dim The dimension of the physical space.
   * @param[out] work A pointer to a workspace array used for intermediate
   * calculations. This array should be of size at least `dim*dim +
   * world_dim*dim`.
   *
   * @return The integration element as a scalar value. If `dim` is zero, the
   * function returns 0.
   *
   * @note This function supports up to three dimensions. If `dim` is greater
   * than three, a static assertion fails.
   */
  template <typename Scalar>
  static Scalar integration_element(const Scalar *nodes[Derived::num_nodes()],
                                    const Scalar *x, dim_t world_dim,
                                    Scalar *work) {
    if constexpr (dim == 0) {
      return 0;
    } else {
      Scalar(&JTJres)[dim][dim] = static_cast<Scalar(&)[dim][dim]>(work);
      Scalar *JTJwork[dim] = static_cast<Scalar *[dim]>(work + dim * dim);
      JTJ(nodes, x, JTJres, world_dim, JTJwork);
      Scalar det;
      if constexpr (dim == 1) {
        det = JTJres[0][0];
      } else if constexpr (dim == 2) {
        det = JTJres[0][0] * JTJres[1][1] - JTJres[0][1] * JTJres[1][0];
      } else if constexpr (dim == 3) {
        const Scalar a = JTJres[0][0];
        const Scalar b = JTJres[0][1];
        const Scalar c = JTJres[0][2];
        const Scalar d = JTJres[1][0];
        const Scalar e = JTJres[1][1];
        const Scalar f = JTJres[1][2];
        const Scalar g = JTJres[2][0];
        const Scalar h = JTJres[2][1];
        const Scalar i = JTJres[2][2];
        det = a * e * i + b * f * g + c * d * h - c * e * g - b * d * i -
              a * f * h;
      } else {
        static_assert(
            !meta::is_same_v<Scalar, Scalar>,
            "integration element is only supported up to three dimensions");
      }
      return sqrt(abs(det));
    }
  }

private:
  /**
   * @brief Computes the product of the Jacobian matrix and its transpose.
   *
   * This function calculates the product of the Jacobian matrix and its
   * transpose, which is used as an intermediate step in computing the
   * integration element.
   *
   * @tparam Scalar The type of the scalar values (e.g., float, double).
   * @param[in] nodes An array of pointers to the node coordinates. Each node is
   * an array of coordinates with a size equal to `world_dim`.
   * @param[in] x A pointer to the coordinates where the Jacobian matrix is
   * evaluated.
   * @param[out] out The resulting matrix, which is the product of the Jacobian
   * matrix and its transpose.
   * @param[in] world_dim The dimension of the physical space.
   * @param[out] work A pointer to a workspace array used for intermediate
   * calculations. This array should be of size at least `world_dim*dim`.
   */
  template <typename Scalar>
  static void JTJ(const Scalar *nodes[Derived::num_nodes()], const Scalar *x,
                  Scalar (&out)[dim][dim], dim_t world_dim, Scalar *work[dim]) {
    Derived::jacobian(nodes, x, work, world_dim);
    for (dim_t i = 0; i < dim; ++i) {
      for (dim_t j = 0; j < dim; ++j) {
        out[i][j] = 0;
        for (dim_t k = 0; k < world_dim; ++k) {
          for (dim_t l = 0; l < world_dim; ++l) {
            out[i][j] += work[k][i] * work[l][j];
          }
        }
      }
    }
  }
};

} // namespace numeric::mesh

#endif
