#!/usr/bin/env python3


import numpy as np


class ReferenceElement:

    def __init__(self, vertices : np.ndarray, idxs: list[int], subelements : dict[type,list[list[int]]]):
        self.vertices : np.ndarray = vertices
        self.idxs : list[int] = idxs
        self.subelements : dict[type,list[list[int]]] = subelements
        self.name : str = 'RefEl'

    def __eq__(self, other):
        return (self.vertices[self.idxs,:] == other.vertices[other.idxs,:]).all()

    @property
    def world_dim(self):
        return self.vertices.shape[1]

    @property
    def num_verts(self):
        return len(self.idxs)

    @property
    def verts(self):
        return self.vertices[self.idxs, :]

    def num_subelements(self, element_type):
        return len(self.subelements.get(element_type, []))

    def subelement(self, element_type, idx):
        local_sub_idxs = self.subelements[element_type][idx]
        global_sub_idxs = [self.idxs[i] for i in local_sub_idxs]
        return element_type(self.vertices, global_sub_idxs)

    def generate_code(self):
        guard = f'NUMERIC_MESH_REF_EL_{self.name.upper()}_HPP_'
        code = f'#ifndef {guard}\n'
        code += f'#define {guard}\n'
        code += f'\n'
        code += f'#include <numeric/meta/meta.hpp>\n'
        for sub in self.subelements.keys():
            code += f'#include <numeric/mesh/ref_el_{sub().name.lower()}.hpp>\n'
        code += f'\n'
        code += f'namespace numeric::mesh {{\n'
        code += f'\n'
        code += f'struct RefEl{self.name} {{\n'
        code += f'  static constexpr dim_t dim = {self.world_dim};\n'
        code += f'  static constexpr dim_t num_nodes = {self.num_verts};\n'
        code += f'  static constexpr char name[] = "{self.name}";\n'
        code += f'  \n'
        code += f'  template <typename Subelement>\n'
        code += f'  static constexpr dim_t num_subelements() {{\n'
        for subel in self.subelements:
            code += f'    if constexpr (meta::is_same_v<Subelement, RefEl{subel().name}>) {{\n'
            code += f'      return {self.num_subelements(subel)};\n'
            code += f'    }}\n'
        code += f'    return 0;\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Scalar>\n'
        code += f'  static constexpr void get_nodes(Scalar (*out)[{max(1,self.world_dim)}]) {{\n'
        for vert in range(self.num_verts):
            for dim in range(self.world_dim):
                code += f'    out[{vert}][{dim}] = {self.verts[vert,dim]};\n'
        code += f'  }}\n'
        code += f'  \n'
        code += f'  template <typename Subelement>\n'
        code += f'  static constexpr void subelement_node_idxs(dim_t idx, dim_t *out) {{\n'
        for subeltype in self.subelements:
            code += f'    if constexpr (meta::is_same_v<Subelement, RefEl{subeltype().name}>) {{\n'
            code += f'      switch (idx) {{\n'
            for i in range(self.num_subelements(subeltype)):
                subel = self.subelement(subeltype, i)
                code += f'      case {i}:\n'
                for j in range(subel.num_verts):
                    code += f'        out[{j}] = {subel.idxs[j]};\n'
                code += f'        break;\n'
            code += f'      }}\n'
            code += f'    }}\n'
        code += f'  }}\n'
        code += f'}};\n'
        code += f'\n'
        code += f'}}\n'
        code += f'\n'
        code += f'#endif'
        return code


class RefElPoint(ReferenceElement):

    VERTICES = np.array([[]], dtype=np.int32)

    def __init__(self, vertices=None, idxs=None):
        if vertices is None:
            vertices = __class__.VERTICES
        if idxs is None:
            idxs = list(range(len(__class__.VERTICES)))
        super().__init__(vertices, idxs, {})
        self.name = 'Point'


class RefElSegment(ReferenceElement):

    VERTICES = np.array([[0], [1]], dtype=np.int32)

    def __init__(self, vertices=None, idxs=None):
        if vertices is None:
            vertices = __class__.VERTICES
        if idxs is None:
            idxs = list(range(len(__class__.VERTICES)))
        super().__init__(vertices, idxs, {RefElPoint:[[0],[1]]})
        self.name = 'Segment'


class RefElTria(ReferenceElement):

    VERTICES = np.array([[0,0], [1,0], [0,1]], dtype=np.int32)

    def __init__(self, vertices=None, idxs=None):
        if vertices is None:
            vertices = __class__.VERTICES
        if idxs is None:
            idxs = list(range(len(__class__.VERTICES)))
        super().__init__(
            vertices, idxs,
            {
                RefElPoint : [[0], [1], [2]],
                RefElSegment : [[0,1], [1,2], [2,0]]
            }
        )
        self.name = 'Tria'


class RefElQuad(ReferenceElement):

    VERTICES = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.int32)

    def __init__(self, vertices=None, idxs=None):
        if vertices is None:
            vertices = __class__.VERTICES
        if idxs is None:
            idxs = list(range(len(__class__.VERTICES)))
        super().__init__(
            vertices, idxs,
            {
                RefElPoint : [[0], [1], [2], [3]], 
                RefElSegment : [[0,1], [1,2], [3,2], [0,3]]
            }
        )
        self.name = 'Quad'


class RefElTetra(ReferenceElement):

    VERTICES = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.int32)

    def __init__(self, vertices=None, idxs=None):
        if vertices is None:
            vertices = __class__.VERTICES
        if idxs is None:
            idxs = list(range(len(__class__.VERTICES)))
        super().__init__(
            vertices, idxs,
            {
                RefElPoint : [[0], [1], [2], [3]],
                RefElSegment : [[0,1], [1,2], [2,0], [0,3], [1,3], [2,3]],
                #RefElTria : [[0,1,3], [1,2,3], [0,3,2], [0,1,2]]
                RefElTria : [[0,1,3], [2,3,1], [0,3,2], [0,2,1]]
            }
        )
        self.name = 'Tetra'


class RefElCube(ReferenceElement):

    VERTICES = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]], dtype=np.int32)

    def __init__(self, vertices=None, idxs=None):
        if vertices is None:
            vertices = __class__.VERTICES
        if idxs is None:
            idxs = list(range(len(__class__.VERTICES)))
        super().__init__(
            vertices, idxs,
            {
                RefElPoint : [[0], [1], [2], [3], [4], [5], [6], [7]],
                RefElSegment : [[0,1], [1,2], [3,2], [0,3], [4,5], [5,6], [7,6], [4,7], [0,4], [1,5], [2,6], [3,7]],
                RefElQuad : [[0,3,7,4], [1,2,6,5], [0,1,5,4], [3,2,6,7], [0,1,2,3], [4,5,6,7]]
            }
        )
        self.name = 'Cube'


if __name__ == '__main__':
    import sys


    element_types = {
        'ref_el_point' : RefElPoint,
        'ref_el_segment' : RefElSegment,
        'ref_el_tria' : RefElTria,
        'ref_el_quad' : RefElQuad,
        'ref_el_tetra' : RefElTetra,
        'ref_el_cube' : RefElCube
    }

    def print_subelements(element, indent=4):
        for sub_type in element_types.values():
            num_sub = element.num_subelements(sub_type)
            for i in range(num_sub):
                sub = element.subelement(sub_type, i)
                print(' '*indent + sub_type.__name__, i)
                print(' '*(indent+4) + str(sub.idxs))
                print((' '*(indent+4) + str(sub.verts)).replace('\n', '\n'+' '*(indent+4)))
                print_subelements(sub, indent+4)
    
    if len(sys.argv) < 2:
        for element_type in element_types.values():
            element = element_type()
            print(element_type.__name__)
            print(element.verts)
            print_subelements(element)
    else:
        element_type = element_types[sys.argv[1]]
        element = element_type()
        print(element.generate_code())
