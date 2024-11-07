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
        #self.coords = sympy.symbols(','.join([f'x[{i}]' for i in range(self.dim)])+',', real=True)

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

    def has_interior_idxs(self):
        raise NotImplementedError

    def interior_idxs(self):
        if not self.has_interior_idxs():
            return np.zeros((0, self.dim))
        low_el = Element(self.ref_el, self.order - 2)
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
                print(f'{ref_sub.name} of order {ref_sub.order} has interior indices {ref_intidxs}')
                for i in range(num_subel):
                    sub = self.subelement(subelement_type, i)
                    intidxs = sub.ref_el.to_global(self.order * ref_intidxs)
                    idx_list.append(intidxs)
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
            expr = self.lagrange(pt)
            repl, red = sympy.cse(expr, optimizations='basic')
            code += f'  case {i}:\n'
            for r in repl:
                code += f'    const Scalar {r[0]} = {sympy.ccode(r[1])};\n'
            code += f'    return {sympy.ccode(red[0])};\n'
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
            for r in repl:
                code += f'    const Scalar {r[0]} = {sympy.ccode(r[1])};\n'
            for d in range(self.dim):
                code += f'    out[{d}] = {sympy.ccode(red[d])};\n'
            code += f'    break;\n'
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

    def code(self):
        code = ''
        code += f'template <> struct BasisLagrange<mesh::ElementType::{self.name}, {self.order}> {{\n'
        code += f'  static constexpr mesh::ElementType element = mesh::ElementType::{self.name};\n'
        code += f'  static constexpr dim_t order = {self.order};\n'
        code += f'  static constexpr dim_t num_basis_functions = {len(self.idxs())};\n'
        code += f'  \n'
        code += f'  template <typename Scalar>\n'
        code += f'  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {{\n'
        code += f'    ' + self.lagrange_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Scalar>\n'
        code += f'  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {{\n'
        code += f'    ' + self.grad_lagrange_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
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
        code += f'}}\n'
        return code


class Segment(Element):

    def __init__(self, order):
        super().__init__(RefElSegment(), order)

    def lagrange(self, point):
        points = [sympy.Integer(i)/self.order for i in range(self.order+1)]
        Y = [sympy.Integer(0) for _ in range(self.order+1)]
        Y[point[0]] = sympy.Integer(1)
        return sympy.interpolating_poly(self.order+1, self.coords[0], points, Y)


class Tria(Element):

    def __init__(self, order):
        super().__init__(RefElTria(), order)

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

    def __init__(self, order):
        super().__init__(RefElQuad(), order)

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

    def __init__(self, order):
        super().__init__(RefElTetra(), order)
    
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

    def __init__(self, order):
        super().__init__(RefElCube(), order)


def specialized_code(element_type, p_max):
    code = ''
    for p in range(1, p_max+1):
        element = element_type(p)
        code += element.code()
        code += '\n\n\n'
    return code



if __name__ == '__main__':
    for ref_el_type in ALL_REF_EL:
        ref_el = ref_el_type()
        for order in range(3):
            print(ref_el.name, order)
            el = Element(ref_el, order)
            print(el.idxs())
