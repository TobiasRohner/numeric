#ifndef NUMERIC_MATH_POLYNOMIAL_HPP_
#define NUMERIC_MATH_POLYNOMIAL_HPP_

#include <numeric/config.hpp>
#include <numeric/math/constants.hpp>
#include <numeric/math/functions.hpp>

namespace numeric::math {

/**
 * @brief Node generator for equispaced interpolation points.
 *
 * @details
 * Generates interpolation nodes \f$x_i = \frac{i}{k}\f$ for \f$i = 0, \dots,
 * k\f$, where \f$k\f$ is the interpolation order.
 *
 * @tparam Order The order of the interpolating polynomial.
 */
template <dim_t Order> struct NodesEquispaced {
  static constexpr dim_t order = Order;

  /**
   * @brief Returns the i-th equispaced node in the interval [0, 1].
   * @param i Index of the node.
   * @return Value of the i-th node.
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar node(dim_t i) {
    return static_cast<Scalar>(i) / order;
  }

  /**
   * @brief Returns the weight at the i-th node.
   * @param i Index of the node.
   * @return Value of the weight at the i-th node.
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar> static NUMERIC_HOST_DEVICE Scalar weight(dim_t i) {
    const Scalar result = exp(order * log(static_cast<Scalar>(order)) -
                              lgamma(static_cast<Scalar>(i + 1)) -
                              lgamma(static_cast<Scalar>(order - i + 1)));
    if ((order - i) % 2) {
      return -result;
    } else {
      return result;
    }
  }
};

/**
 * @brief Node generator for Chebyshev interpolation points.
 *
 * @details
 * Generates Chebyshev nodes \f$x_i = \cos\left(\frac{i \pi}{k}\right)\f$
 * for \f$i = 0, \dots, k\f$, where \f$k\f$ is the interpolation order.
 *
 * @tparam Order The order of the interpolating polynomial.
 */
template <dim_t Order> struct NodesChebyshev {
  static constexpr dim_t order = Order;

  /**
   * @brief Returns the i-th Chebyshev node in the interval [-1, 1].
   * @param i Index of the node.
   * @return Value of the i-th node.
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar node(dim_t i) {
    return cos(i * pi<Scalar> / order);
  }

  /**
   * @brief Returns the weight at the i-th node.
   * @param i Index of the node.
   * @return Value of the weight at the i-th node.
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar> static NUMERIC_HOST_DEVICE Scalar weight(dim_t i) {
    Scalar value = 1;
    if (i == 0 || i == order) {
      value = 0.5;
    }
    return i % 2 ? -value : value;
  }
};

/**
 * @brief A utility structure for evaluating Lagrange polynomials using the
 * barycentric interpolation method.
 *
 * @details
 * Lagrange interpolation constructs a polynomial \f$ P(x) \f$ of degree \f$ k
 * \f$ that passes through a given set of \f$ k+1 \f$ data points \f$(x_i,
 * y_i)\f$. The interpolation polynomial is given by:
 *
 * \f[
 *   P(x) = \sum_{i=0}^{k} y_i \ell_i(x)
 * \f]
 * where the Lagrange basis functions \f$\ell_i(x)\f$ are defined as:
 * \f[
 *   \ell_i(x) = \prod_{\substack{j=0 \\ j \ne i}}^k \frac{x - x_j}{x_i - x_j}
 * \f]
 *
 * The barycentric form rewrites this for numerical stability and efficiency
 * using weights \f$ w_i \f$: \f[ \ell_i(x) = \frac{w_i}{x - x_i} \Bigg/
 * \sum_{j=0}^{k} \frac{w_j}{x - x_j} \f]
 */
template <typename InterpolationNodes> struct Lagrange {
  using interpolation_nodes_t = InterpolationNodes;
  static constexpr dim_t order = interpolation_nodes_t::order;

  /**
   * @brief Returns the i-th interpolation node.
   * @param i Index of the node.
   * @return \f$ x_i \f$
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar node(dim_t i) {
    return interpolation_nodes_t::template node<Scalar>(i);
  }

  /**
   * @brief Returns the weight at the i-th interpolation node.
   * @param i Index of the node.
   * @return \f$ w_i \f$
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar> static NUMERIC_HOST_DEVICE Scalar weight(dim_t i) {
    return interpolation_nodes_t::template weight<Scalar>(i);
  }

  /**
   * @brief Evaluates the Lagrange interpolating polynomial at a given point.
   *
   * @details Uses the barycentric formula:
   * \f[
   *   P(x) = \frac{\sum_{i=0}^{k} \frac{w_i y_i}{x - x_i}}{\sum_{i=0}^{k}
   * \frac{w_i}{x - x_i}} \f] If \f$ x = x_i \f$ for some \f$ i \f$, returns the
   * corresponding \f$ y_i \f$ directly.
   *
   * @param y Array of function values \f$ y_i \f$.
   * @param x Point \f$ x \f$ where the polynomial is evaluated.
   * @return Interpolated value \f$ P(x) \f$.
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar>
  static NUMERIC_HOST_DEVICE Scalar eval(const Scalar *y, Scalar x) {
    Scalar nom = 0;
    Scalar denom = 0;
    for (dim_t i = 0; i <= order; ++i) {
      if (abs(x - node<Scalar>(i)) == 0) {
        return y[i];
      }
      const Scalar fac = weight<Scalar>(i) / (x - node<Scalar>(i));
      nom += fac * y[i];
      denom += fac;
    }
    return nom / denom;
  }

  /**
   * @brief Evaluates the derivative \f$ P'(x) \f$ of the interpolating
   * polynomial.
   *
   * @details Uses the formula:
   * \f[
   *   P'(x) = \sum_{i=0}^{k} y_i \ell_i'(x)
   * \f]
   *
   * @param y Array of function values \f$ y_i \f$.
   * @param x Point of evaluation.
   * @return Derivative \f$ P'(x) \f$.
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar>
  static NUMERIC_HOST_DEVICE Scalar diff(const Scalar *y, Scalar x) {
    // TODO; Is there a better way?
    Scalar result = 0;
    for (dim_t i = 0; i <= order; ++i) {
      result += y[i] * basis_diff<Scalar>(i, x);
    }
    return result;
  }

  /**
   * @brief Evaluates the i-th Lagrange basis function \f$ \ell_i(x) \f$.
   *
   * @details
   * \f[
   *   \ell_i(x) = \frac{\frac{w_i}{x - x_i}}{\sum_{j=0}^{k} \frac{w_j}{x -
   * x_j}} \f] Special cases:
   * - If \f$ x = x_i \f$, returns 1.
   * - If \f$ x = x_j, j \ne i \f$, returns 0.
   *
   * @param i Index of the basis function.
   * @param x Point of evaluation.
   * @return Value \f$ \ell_i(x) \f$.
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar>
  static NUMERIC_HOST_DEVICE Scalar basis(dim_t i, Scalar x) {
    if (abs(x - node<Scalar>(i)) == 0) {
      return 1;
    }
    const Scalar nom = weight<Scalar>(i) / (x - node<Scalar>(i));
    Scalar denom = 0;
    for (dim_t j = 0; j <= order; ++j) {
      if (x == node<Scalar>(j)) {
        return 0;
      }
      denom += weight<Scalar>(j) / (x - node<Scalar>(j));
    }
    return nom / denom;
  }

  /**
   * @brief Evaluates the derivative \f$ \ell_i'(x) \f$ of the i-th Lagrange
   * basis function.
   *
   * @details
   * For \f$ x \ne x_j \f$:
   * \f[
   *   \ell_i'(x) = \ell_i(x) \sum_{l = 0}^k \frac{1}{x - x_l}.
   * \f]
   *
   * For \f$ x = x_j \f$, uses the limiting form:
   * - If \f$ i \ne j \f$: \f$ \ell_i'(x_j) = \frac{w_i}{w_j (x_j - x_i)} \f$
   * - If \f$ i = j \f$: \f$ \ell_i'(x_i) = -\sum_{\substack{l = 0 \\ l \ne
   * i}}^k \frac{w_l}{w_i (x_i - x_l)} \f$
   *
   * @param i Index of the basis function.
   * @param x Point of evaluation.
   * @return Derivative \f$ \ell_i'(x) \f$.
   * @tparam Scalar Floating-point type.
   */
  template <typename Scalar>
  static NUMERIC_HOST_DEVICE Scalar basis_diff(dim_t i, Scalar x) {
    if (abs(x - node<Scalar>(i)) == 0) {
      Scalar sum = 0;
      for (dim_t j = 0; j <= order; ++j) {
        if (i != j) {
          sum += -weight<Scalar>(j) / weight<Scalar>(i) / (x - node<Scalar>(j));
        }
      }
      return sum;
    } else {
      const Scalar nom = weight<Scalar>(i) / (x - node<Scalar>(i));
      Scalar denom = 0;
      Scalar sum = 0;
      for (dim_t j = 0; j <= order; ++j) {
        if (abs(x - node<Scalar>(j)) == 0) {
          return weight<Scalar>(i) / weight<Scalar>(j) / (x - node<Scalar>(i));
        }
        denom += weight<Scalar>(j) / (x - node<Scalar>(j));
        if (i != j) {
          sum += 1. / (x - node<Scalar>(j));
        }
      }
      return nom / denom * sum;
    }
  }
};

} // namespace numeric::math

#endif
