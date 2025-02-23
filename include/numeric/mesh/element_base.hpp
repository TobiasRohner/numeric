#ifndef NUMERIC_MESH_ELEMENT_BASE_HPP_
#define NUMERIC_MESH_ELEMENT_BASE_HPP_

#include <iostream>
#include <numeric/config.hpp>
#include <numeric/math/basis_lagrange.hpp>
#include <numeric/math/functions.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::mesh {

template <typename Derived> struct ElementBase {
  using traits_t = ElementTraits<Derived>;
  using ref_el_t = typename traits_t::ref_el_t;
  using basis_t = typename traits_t::basis_t;
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;
  static constexpr dim_t num_nodes = traits_t::num_nodes;

  template <typename Base, typename Element>
  using has_subelement_t =
      decltype(Base::num_subelements(meta::type_tag<Element>()));

  template <typename Base, typename Element>
  using has_subelement_node_idxs_t =
      decltype(Base::subelement_node_idxs(meta::declval<dim_t>(),
                                          meta::declval<dim_t *>(),
                                          meta::type_tag<Element>()));

  template <typename Element> static constexpr dim_t num_subelements() {
    /*
    if constexpr (meta::is_detected_v<has_subelement_t, Derived, Element>) {
      return Derived::num_subelements(meta::type_tag<Element>());
    } else {
      return 0;
    }
    */
    return ref_el_t::template num_subelements<typename Element::ref_el_t>();
  }

  template <typename Element>
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {
    /*
    if constexpr (meta::is_detected_v<has_subelement_node_idxs_t, Derived,
                                      Element>) {
      Derived::subelement_node_idxs(subelement, idxs,
                                    meta::type_tag<Element>());
    }
    */
    basis_t::template subelement_node_idxs<typename Element::ref_el_t>(
        subelement, idxs);
  }

  template <typename Scalar>
  static constexpr void local_to_global(const Scalar (*nodes)[num_nodes],
                                        const Scalar *x, Scalar *out,
                                        dim_t world_dim) {
    for (dim_t i = 0; i < world_dim; ++i) {
      out[i] = basis_t::eval(x, nodes[i]);
    }
  }

  template <typename Scalar>
  static constexpr void jacobian(const Scalar (*nodes)[num_nodes],
                                 const Scalar *x, Scalar (*out)[dim],
                                 dim_t world_dim) {
    for (dim_t i = 0; i < world_dim; ++i) {
      basis_t::grad(x, nodes[i], out[i]);
    }
  }

  template <typename Scalar>
  static dim_t integration_element_work_size(dim_t world_dim) {
    static constexpr dim_t dim = ElementTraits<Derived>::dim;
    return sizeof(Scalar) * (dim * dim + world_dim * dim);
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
   * function returns 1.
   */
  template <typename Scalar>
  static Scalar
  integration_element(const Scalar (*nodes)[ElementTraits<Derived>::num_nodes],
                      const Scalar *x, dim_t world_dim, void *work) {
    if constexpr (dim == 0) {
      return 1;
    } else {
      Scalar(&JTJres)[dim][dim] = *reinterpret_cast<Scalar(*)[dim][dim]>(work);
      Scalar(*JTJwork)[dim] = reinterpret_cast<Scalar(*)[dim]>(
          static_cast<Scalar *>(work) + dim * dim);
      JTJ<Scalar>(nodes, x, JTJres, world_dim, JTJwork);
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

  template <typename Scalar>
  static dim_t jacobian_inverse_gramian_work_size(dim_t world_dim) {
    static constexpr dim_t dim = ElementTraits<Derived>::dim;
    if (world_dim == dim) {
      return sizeof(Scalar) * dim * dim;
    } else {
      return sizeof(Scalar) * (world_dim * dim + dim * dim);
    }
  }

  template <typename Scalar>
  static void jacobian_inverse_gramian(
      const Scalar (*nodes)[ElementTraits<Derived>::num_nodes], const Scalar *x,
      Scalar (*out)[dim], dim_t world_dim, void *work) {
    Scalar(*J)[dim] = reinterpret_cast<Scalar(*)[dim]>(work);
    Derived::jacobian(nodes, x, J, world_dim);
    if (world_dim == dim) {
      Scalar(&out_dim)[dim][dim] = *reinterpret_cast<Scalar(*)[dim][dim]>(out);
      if constexpr (dim == 0) {
        // Nothing to do here
      } else if constexpr (dim == 1) {
        inverse(out_dim, J[0][0]);
      } else if constexpr (dim == 2) {
        inverse(out_dim, J[0][0], J[1][0], J[0][1], J[1][1]);
      } else if constexpr (dim == 3) {
        inverse(out_dim, J[0][0], J[1][0], J[2][0], J[0][1], J[1][1], J[2][1],
                J[0][2], J[1][2], J[2][2]);
      } else {
        static_assert(!meta::is_same_v<Scalar, Scalar>,
                      "jacobian inverse gramian, is only supported up to three "
                      "dimensions");
      }
    } else {
      Scalar(&JTJ)[dim][dim] = *reinterpret_cast<Scalar(*)[dim][dim]>(
          static_cast<Scalar *>(work) + world_dim * dim);
      for (dim_t i = 0; i < dim; ++i) {
        for (dim_t j = 0; j < dim; ++j) {
          JTJ[i][j] = 0;
          for (dim_t k = 0; k < world_dim; ++k) {
            JTJ[i][j] += J[k][i] * J[k][j];
          }
        }
      }
      if constexpr (dim == 0) {
        // Nothing to do here
      } else if constexpr (dim == 1) {
        inverse(JTJ, JTJ[0][0]);
      } else if constexpr (dim == 2) {
        inverse(JTJ, JTJ[0][0], JTJ[0][1], JTJ[1][0], JTJ[1][1]);
      } else if constexpr (dim == 3) {
        inverse(JTJ, JTJ[0][0], JTJ[0][1], JTJ[0][2], JTJ[1][0], JTJ[1][1],
                JTJ[1][2], JTJ[2][0], JTJ[2][1], JTJ[2][2]);
      } else {
        static_assert(!meta::is_same_v<Scalar, Scalar>,
                      "jacobian inverse gramian, is only supported up to three "
                      "dimensions");
      }
      for (dim_t i = 0; i < world_dim; ++i) {
        for (dim_t j = 0; j < dim; ++j) {
          out[i][j] = 0;
          for (dim_t k = 0; k < dim; ++k) {
            out[i][j] += J[i][k] * JTJ[k][j];
          }
        }
      }
    }
  }

private:
  template <typename Scalar>
  static void JTJ(const Scalar (*nodes)[ElementTraits<Derived>::num_nodes],
                  const Scalar *x, Scalar (&out)[dim][dim], dim_t world_dim,
                  Scalar (*work)[dim]) {
    Derived::template jacobian<Scalar>(nodes, x, work, world_dim);
    for (dim_t i = 0; i < dim; ++i) {
      for (dim_t j = 0; j < dim; ++j) {
        out[i][j] = 0;
        for (dim_t k = 0; k < world_dim; ++k) {
          out[i][j] += work[k][i] * work[k][j];
        }
      }
    }
  }

  template <typename Scalar>
  static void inverse(Scalar (&out)[1][1], Scalar a) {
    out[0][0] = 1 / a;
  }

  template <typename Scalar>
  static void inverse(Scalar (&out)[2][2], Scalar a, Scalar b, Scalar c,
                      Scalar d) {
    const Scalar invdet = 1 / (a * d - b * c);
    out[0][0] = invdet * d;
    out[0][1] = -invdet * b;
    out[1][0] = -invdet * c;
    out[1][1] = invdet * a;
  }

  template <typename Scalar>
  static void inverse(Scalar (&out)[3][3], Scalar a, Scalar b, Scalar c,
                      Scalar d, Scalar e, Scalar f, Scalar g, Scalar h,
                      Scalar i) {
    const Scalar invdet = 1 / (a * e * i + b * f * g + c * d * h - c * e * g -
                               b * d * i - a * f * h);
    out[0][0] = invdet * (e * i - f * h);
    out[0][1] = invdet * -(b * i - c * h);
    out[0][2] = invdet * (b * f - c * e);
    out[1][0] = invdet * -(d * i - f * g);
    out[1][1] = invdet * (a * i - c * g);
    out[1][2] = invdet * -(a * f - c * d);
    out[2][0] = invdet * (d * h - e * g);
    out[2][1] = invdet * -(a * h - b * g);
    out[2][2] = invdet * (a * e - b * d);
  }
};

} // namespace numeric::mesh

#endif
