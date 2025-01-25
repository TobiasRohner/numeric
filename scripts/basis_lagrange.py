#!/usr/bin/env python3

import sympy
import numpy as np
import matplotlib.pyplot as plt

from elements import RefElPoint, RefElSegment, RefElTria, RefElQuad, RefElTetra, RefElCube


ALL_REF_EL = [RefElPoint, RefElSegment, RefElTria, RefElQuad, RefElTetra, RefElCube]


class Element:

    def __init__(self, ref_el, order):
        self.ref_el = ref_el
        self.order = order
        self.coords = sympy.symbols(' '.join([f'x[{i}]' for i in range(max(1, self.dim))]), real=True, seq=True)
        if isinstance(self.ref_el, RefElPoint):
            self.min_order_for_int_idxs = 0
        elif isinstance(self.ref_el, RefElSegment):
            self.min_order_for_int_idxs = 2
        elif isinstance(self.ref_el, RefElTria):
            self.min_order_for_int_idxs = 3
        elif isinstance(self.ref_el, RefElQuad):
            self.min_order_for_int_idxs = 2
        elif isinstance(self.ref_el, RefElTetra):
            self.min_order_for_int_idxs = 4
        elif isinstance(self.ref_el, RefElCube):
            self.min_order_for_int_idxs = 2

    @property
    def dim(self):
        return self.ref_el.world_dim

    @property
    def name(self):
        return self.ref_el.name

    def num_subelements(self, element_type):
        return self.ref_el.num_subelements(element_type)

    def subelement(self, element_type, idx):
        return Element(self.ref_el.subelement(element_type, idx), self.order)

    def sub_to_parent_coord(self, element_type, idx, point):
        if element_type is RefElPoint and self.num_subelements(RefElPoint) > 0:
            subel = self.ref_el.subelement(RefElPoint, idx)
            verts = subel.verts
            return self.order * verts[0]
        elif element_type is RefElSegment and self.num_subelements(RefElSegment) > 0:
            subel = self.ref_el.subelement(RefElSegment, idx)
            verts = subel.verts
            origin = self.order * verts[0]
            dir_x = verts[1] - verts[0]
            return origin + point[0]*dir_x
        elif element_type is RefElTria and self.num_subelements(RefElTria) > 0:
            subel = self.ref_el.subelement(RefElTria, idx)
            verts = subel.verts
            origin = self.order * verts[0]
            dir_x = verts[1] - verts[0]
            dir_y = verts[2] - verts[0]
            return origin + point[0]*dir_x + point[1]*dir_y
        elif element_type is RefElQuad and self.num_subelements(RefElQuad) > 0:
            subel = self.ref_el.subelement(RefElQuad, idx)
            verts = subel.verts
            origin = self.order * verts[0]
            dir_x = verts[1] - verts[0]
            dir_y = verts[3] - verts[0]
            return origin + point[0]*dir_x + point[1]*dir_y
        raise ValueError('Invalid subelement type')

    def has_interior_idxs(self):
        return self.order >= self.min_order_for_int_idxs

    def interior_idxs(self):
        if isinstance(self.ref_el, RefElPoint):
            return np.zeros((1, self.dim), dtype=np.int32)
        if not self.has_interior_idxs():
            return np.zeros((0, self.dim), dtype=np.int32)
        low_el = Element(self.ref_el, self.order - self.min_order_for_int_idxs)
        return 1 + low_el.idxs()

    def idxs(self):
        if self.order == 0:
            return self.ref_el.verts[0:1,:]
        if self.order == 1:
            return self.ref_el.verts
        else:
            idx_list = []
            for subelement_type in ALL_REF_EL:
                num_subel = self.num_subelements(subelement_type)
                ref_sub = Element(subelement_type(), self.order)
                ref_intidxs = ref_sub.interior_idxs()
                for i in range(num_subel):
                    intidxs = []
                    for j in range(ref_intidxs.shape[0]):
                        point = ref_intidxs[j]
                        glob = self.sub_to_parent_coord(subelement_type, i, point)
                        intidxs.append(glob.reshape((1, glob.size)))
                    idx_list += intidxs
            idx_list.append(self.interior_idxs())
            return np.concatenate(idx_list, axis=0)
                    

    def idxs_code(self):
        code = 'switch (i) {\n'
        for idx,pt in enumerate(self.idxs()):
            code += f'  case {idx}:\n'
            for d in range(self.dim):
                code += f'    out[{d}] = {pt[d]};\n'
            code += f'    break;\n'
        code += '}'
        return code

    def lagrange(self, point):
        raise NotImplementedError

    def lagrange_code(self):
        points = self.idxs()
        code = 'switch (i) {\n'
        for i,pt in enumerate(points):
            expr = sympy.simplify(self.lagrange(pt))
            repl, red = sympy.cse(expr, optimizations='basic')
            code += f'  case {i}:\n'
            code += f'    {{\n'
            for r in repl:
                code += f'      const Scalar {r[0]} = {sympy.ccode(r[1])};\n'
            code += f'      return {sympy.ccode(red[0])};\n'
            code += f'    }}\n'
        code += '  default:\n'
        code += '    return 0;\n'
        code += '}'
        return code

    def eval(self):
        points = self.idxs()
        coeffs = [sympy.Symbol(f'coeffs[{i}]', real=True) for i in range(len(points))]
        value = 0
        for pt,c in zip(points, coeffs):
            value += c * self.lagrange(pt)
        return value

    def eval_code(self):
        expr = self.eval()
        repl, red = sympy.cse(expr, optimizations='basic')
        code = ''
        for r in repl:
            code += f'const Scalar {r[0]} = {sympy.ccode(r[1])};\n'
        code += f'return {sympy.ccode(red[0])};'
        return code

    def grad_lagrange(self, point):
        b = self.lagrange(point)
        return [sympy.diff(b, coord) for coord in self.coords]

    def grad_lagrange_code(self):
        points = self.idxs()
        code = 'switch (i) {\n'
        for i,pt in enumerate(points):
            expr = self.grad_lagrange(pt)
            repl, red = sympy.cse(expr, optimizations='basic')
            code += f'  case {i}:\n'
            code += f'    {{\n'
            for r in repl:
                code += f'      const Scalar {r[0]} = {sympy.ccode(r[1])};\n'
            for d in range(self.dim):
                code += f'      out[{d}] = {sympy.ccode(red[d])};\n'
            code += f'      break;\n'
            code += f'    }}\n'
        code += '  default:\n'
        code += '    break;\n'
        code += '}'
        return code

    def grad(self):
        points = self.idxs()
        coeffs = [sympy.Symbol(f'coeffs[{i}]', real=True) for i in range(len(points))]
        grd = [0 for _ in range(self.dim)]
        for pt,c in zip(points, coeffs):
            grdb = self.grad_lagrange(pt)
            for d in range(self.dim):
                grd[d] += c * grdb[d]
        return grd

    def grad_code(self):
        expr = self.grad()
        repl, red = sympy.cse(expr, optimizations='basic')
        code = ''
        for r in repl:
            code += f'const Scalar {r[0]} = {sympy.ccode(r[1])};\n'
        for d in range(self.dim):
            code += f'out[{d}] = {sympy.ccode(red[d])};\n'
        return code[:-1]

    def node_code(self):
        code = ''
        code += f'dim_t idxs[{self.dim}];\n'
        code += f'node_idxs(i, idxs);\n'
        for i in range(self.dim):
            code += f'out[{i}] = static_cast<Scalar>(idxs[{i}]) / order;\n'
        return code[:-1]

    def subelement_node_idxs_code(self):
        code = ''
        parent_idxs = self.idxs()
        for subel in ALL_REF_EL:
            code += '\n'
            code += f'static void subelement_node_idxs(dim_t subelement, dim_t *idxs, meta::type_tag<mesh::RefEl{subel().name}>) {{\n'
            code += f'  switch (subelement) {{\n'
            for i in range(self.num_subelements(subel)):
                child = Element(subel(), self.order)
                child_idxs = child.idxs()
                code += f'  case {i}:\n'
                for j in range(child_idxs.shape[0]):
                    rel_idx = 0
                    for k in range(parent_idxs.shape[0]):
                        if np.all(parent_idxs[k] == self.sub_to_parent_coord(subel, i, child_idxs[j])):
                            rel_idx = k
                            break
                    code += f'    idxs[{j}] = {rel_idx};\n'
                code += f'    break;\n'
            code += f'  default:\n'
            code += f'    break;\n'
            code += f'  }}\n'
            code += f'}}\n'
        return code

    def code(self):
        code = f''
        code += f'template <> struct BasisLagrange<mesh::RefEl{self.name}, {self.order}> {{\n'
        code += f'  using ref_el_t = mesh::RefEl{self.name};\n'
        code += f'  static constexpr dim_t order = {self.order};\n'
        code += f'  static constexpr dim_t num_basis_functions = {len(self.idxs())};\n'
        code += f'  \n'
        code += f'  template <typename Scalar>\n'
        code += f'  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {{\n'
        code += f'    ' + self.eval_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Scalar>\n'
        code += f'  static constexpr void grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {{\n'
        code += f'    ' + self.grad_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {{\n'
        code += f'    ' + self.node_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  static constexpr void node_idxs(dim_t i, dim_t *out) {{\n'
        code += f'    ' + self.idxs_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'\n'
        code += f'  template <typename Element>\n'
        code += f'  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {{\n'
        code += f'    subelement_node_idxs(subelement, idxs, meta::type_tag<Element>{{}});\n'
        code += f'  }}\n'
        code += f'  ' + self.subelement_node_idxs_code().replace('\n', '\n  ')
        code += f'}};\n'
        return code


class Segment(Element):

    REF_EL = RefElSegment

    def __init__(self, order):
        super().__init__(Segment.REF_EL(), order)

    def lagrange(self, point):
        points = [sympy.Integer(i)/self.order for i in range(self.order+1)]
        Y = [sympy.Integer(0) for _ in range(self.order+1)]
        Y[point[0]] = sympy.Integer(1)
        return sympy.interpolating_poly(self.order+1, self.coords[0], points, Y)


class Tria(Element):
    """
    These basis functions are taken from P. Silvester, High-Order Polynomial
    Triangular Finite Elements for Potential Problems, Int. J. Engng Sci. Vol. 7,
    pp 849-861
    """

    REF_EL = RefElTria

    def __init__(self, order):
        super().__init__(Tria.REF_EL(), order)

    def lagrange(self, point):
        i, j = point
        k = self.order - i - j
        def P(m, z):
            i = sympy.Symbol('i', integer=True, positive=True)
            return sympy.Product((self.order*z - i + 1) / i, (i, 1, m))
        l1 = self.coords[0]
        l2 = self.coords[1]
        l3 = 1 - self.coords[0] - self.coords[1]
        bijk = P(i, l1) * P(j, l2) * P(k, l3)
        return bijk.doit()


class Quad(Element):

    REF_EL = RefElQuad

    def __init__(self, order):
        super().__init__(Quad.REF_EL(), order)

    def lagrange(self, point):
        points = [sympy.Integer(i)/self.order for i in range(self.order+1)]
        Y1 = [sympy.Integer(0) for _ in range(self.order+1)]
        Y1[point[0]] = sympy.Integer(1)
        Y2 = [sympy.Integer(0) for _ in range(self.order+1)]
        Y2[point[1]] = sympy.Integer(1)
        poly_x = sympy.interpolating_poly(self.order+1, self.coords[0], points, Y1)
        poly_y = sympy.interpolating_poly(self.order+1, self.coords[1], points, Y2)
        return poly_x * poly_y


class Tetra(Element):

    REF_EL = RefElTetra

    def __init__(self, order):
        super().__init__(Tetra.REF_EL(), order)
    
    def lagrange(self, point):
        i, j, k = point
        l = self.order - i - j - k
        def P(m, z):
            i = sympy.Symbol('i', integer=True, positive=True)
            return sympy.Product((self.order*z - i + 1) / i, (i, 1, m))
        x, y, z = sympy.symbols('x y z', real=True)
        l1 = self.coords[0]
        l2 = self.coords[1]
        l3 = self.coords[2]
        l4 = 1 - l1 - l2 - l3
        bijkl = P(i, l1) * P(j, l2) * P(k, l3) * P(l, l4)
        return bijkl.doit()


class Cube(Element):

    REF_EL = RefElCube

    def __init__(self, order):
        super().__init__(Cube.REF_EL(), order)

    def lagrange(self, point):
        points = [sympy.Integer(i)/self.order for i in range(self.order+1)]
        Y1 = [sympy.Integer(0) for _ in range(self.order+1)]
        Y1[point[0]] = sympy.Integer(1)
        Y2 = [sympy.Integer(0) for _ in range(self.order+1)]
        Y2[point[1]] = sympy.Integer(1)
        Y3 = [sympy.Integer(0) for _ in range(self.order+1)]
        Y3[point[2]] = sympy.Integer(1)
        poly_x = sympy.interpolating_poly(self.order+1, self.coords[0], points, Y1)
        poly_y = sympy.interpolating_poly(self.order+1, self.coords[1], points, Y2)
        poly_z = sympy.interpolating_poly(self.order+1, self.coords[2], points, Y3)
        return poly_x * poly_y * poly_z


def generate_code(element_name, p_max):
    element_types = {
        'segment': Segment,
        'tria': Tria,
        'quad': Quad,
        'tetra': Tetra,
        'cube': Cube
    }
    element_type = element_types[element_name]
    ref_el = element_type.REF_EL()
    guard = f'NUMERIC_MATH_BASIS_LAGRANGE_{ref_el.name.upper()}_HPP_'
    code = f'#ifndef {guard}\n'
    code += f'#define {guard}\n'
    code += f'\n'
    code += f'#include <numeric/meta/meta.hpp>\n'
    code += f'#include <numeric/meta/type_tag.hpp>\n'
    code += f'#include <numeric/mesh/ref_el_point.hpp>\n'
    code += f'#include <numeric/mesh/ref_el_segment.hpp>\n'
    code += f'#include <numeric/mesh/ref_el_tria.hpp>\n'
    code += f'#include <numeric/mesh/ref_el_quad.hpp>\n'
    code += f'#include <numeric/mesh/ref_el_tetra.hpp>\n'
    code += f'#include <numeric/mesh/ref_el_cube.hpp>\n'
    code += f'\n'
    code += f'namespace numeric::math {{\n'
    code += f'\n'
    for p in range(1, p_max+1):
        element = element_type(p)
        code += element.code()
        code += f'\n'
    code += f'}}\n'
    code += f'\n'
    code += f'#endif'
    return code



if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        for element_type in [Segment, Tria, Quad, Tetra, Cube]:
            print(specialized_code(element_type, 3))
    else:
        element_name = sys.argv[1]
        max_order = int(sys.argv[2])
        print(generate_code(element_name, max_order))
