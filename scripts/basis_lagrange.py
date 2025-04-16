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

    def interior_idxs(self):
        if isinstance(self.ref_el, RefElPoint):
            return np.zeros((1, self.dim))
        elif isinstance(self.ref_el, RefElSegment):
            if self.order < 2:
                return np.zeros((0, self.dim), dtype=np.int32)
            return np.arange(1, self.order).reshape((self.order-1, self.dim))
        elif isinstance(self.ref_el, RefElTria):
            if self.order < 3:
                return np.zeros((0, self.dim), dtype=np.int32)
            low_el = Element(self.ref_el, self.order - 3)
            return 1 + low_el.idxs()
        elif isinstance(self.ref_el, RefElQuad):
            if self.order < 2:
                return np.zeros((0, self.dim), dtype=np.int32)
            idxs1d = np.arange(1, self.order)
            idxs = np.zeros(((self.order-1)**2, self.dim), dtype=np.int32)
            idxs[:,0] = np.tile(idxs1d, self.order-1)
            idxs[:,1] = np.repeat(idxs1d, self.order-1)
            return idxs
        elif isinstance(self.ref_el, RefElTetra):
            if self.order < 4:
                return np.zeros((0, self.dim), dtype=np.int32)
            low_el = Element(self.ref_el, self.order - 4)
            return 1 + low_el.idxs()
        elif isinstance(self.ref_el, RefElCube):
            if self.order < 2:
                return np.zeros((0, self.dim), dtype=np.int32)
            idxs1d = np.arange(1, self.order)
            idxs = np.zeros(((self.order-1)**3, self.dim), dtype=np.int32)
            idxs[:,0] = np.tile(idxs1d, (self.order-1)**2)
            idxs[:,1] = np.tile(np.repeat(idxs1d, self.order-1), self.order-1)
            idxs[:,2] = np.repeat(idxs1d, (self.order-1)**2)
            return idxs
        else:
            raise NotImplementedError

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

    def eval_basis_code(self):
        points = self.idxs()
        code = ''
        basis_functions = [sympy.simplify(self.lagrange(pt)) for pt in points]
        repl, red = sympy.cse(basis_functions, optimizations='basic')
        for r in repl:
            code += f'const Scalar {r[0]} = {sympy.ccode(r[1])};\n'
        for i,bf in enumerate(red):
            code += f'out[{i}] = {sympy.ccode(bf)};\n'
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

    def grad_basis_code(self):
        points = self.idxs()
        basis_functions = [sympy.simplify(self.lagrange(pt)) for pt in points]
        gradients = sum([[sympy.diff(bf, coord) for coord in self.coords] for bf in basis_functions], start=[])
        repl, red = sympy.cse(gradients, optimizations='basic')
        code = ''
        for r in repl:
            code += f'const Scalar {r[0]} = {sympy.ccode(r[1])};\n'
        dim = len(self.coords)
        for i in range(len(basis_functions)):
            for j in range(dim):
                code += f'out[{i}][{j}] = {sympy.ccode(red[dim*i+j])};\n'
        return code

    def node_code(self):
        code = ''
        code += f'dim_t idxs[{self.dim}];\n'
        code += f'node_idxs(i, idxs);\n'
        for i in range(self.dim):
            code += f'out[{i}] = static_cast<Scalar>(idxs[{i}]) / order;\n'
        return code[:-1]

    def interpolation_nodes_code(self):
        code = ''
        for i,pt in enumerate(self.idxs()):
            for j in range(self.dim):
                code += f'out[{i}][{j}] = static_cast<Scalar>({pt[j]}) / order;\n'
        return code

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
        code += f'  static constexpr dim_t num_interpolation_nodes = {len(self.idxs())};\n'
        code += f'  \n'
        code += f'  template <typename Scalar>\n'
        code += f'  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {{\n'
        code += f'    ' + self.eval_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Scalar>\n'
        code += f'  static constexpr void eval_basis(const Scalar *x, Scalar *out) {{\n'
        code += f'    ' + self.eval_basis_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Scalar>\n'
        code += f'  static constexpr void grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {{\n'
        code += f'    ' + self.grad_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Scalar>\n'
        code += f'  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[{len(self.coords)}]) {{\n'
        code += f'    ' + self.grad_basis_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {{\n'
        code += f'    ' + self.node_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Scalar> static constexpr void interpolation_nodes(Scalar (*out)[{self.dim}]) {{\n'
        code += f'    ' + self.interpolation_nodes_code().replace('\n', '\n    ')
        code += f'\n'
        code += f'  }}\n'
        code += f'  template <typename Scalar> static constexpr void interpolate(const Scalar *node_values, Scalar *coeffs) {{\n'
        code += f'    for (dim_t i = 0 ; i < num_interpolation_nodes ; ++i) {{\n'
        code += f'      coeffs[i] = node_values[i];\n'
        code += f'    }}\n'
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


def generate_code(p_max):
    guard = f'NUMERIC_MATH_BASIS_LAGRANGE_SPECIALIZATION_HPP_'
    code = f'#ifndef {guard}\n'
    code += f'#define {guard}\n'
    code += f'\n'
    code += f'namespace numeric::math {{\n'
    code += f'\n'
    for element_type in [Segment, Tria, Quad, Tetra, Cube]:
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

    max_order = int(sys.argv[1])
    print(generate_code(max_order))
