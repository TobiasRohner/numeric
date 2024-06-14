#!/usr/bin/env python3

import mpmath as mp
import sys
import os

mp.mp.dps = 30


def extract_table(data, name):
    varloc = data.find(name)
    if varloc < 0:
        return []
    begin = data.find('[', varloc)
    end = data.find(']', varloc)
    values = [mp.mpf(val[:-1]) for val in data[begin:end+1].split()[1:]]
    return values


def extract_qr_tria(data, order):
    xs = extract_table(data, f'xs{order}')
    ys = extract_table(data, f'ys{order}')
    ws = extract_table(data, f'ws{order}')
    scale = 1 / mp.sqrt(3)
    xo = [0.5*(x+1)-0.5*scale*(y+scale) for x,y in zip(xs,ys)]
    yo = [scale*(y+scale) for y in ys]
    wo = [0.5*scale*w for w in ws]
    return xo, yo, wo


def read_qrs_tria():
    folder = os.path.dirname(os.path.abspath(sys.argv[0]))
    fname = os.path.join(folder, 'triasymq/tables/triasymq/data_post.m')
    with open(fname, 'r') as f:
        data = f.read()
    qrs = []
    for order in range(1, 50+1):
        qr = extract_qr_tria(data, order)
        qrs.append((order, qr))
    return qrs


def qr_to_cpp_array(qr):
    return '{\n'+',\n'.join([
            '  {\n'+',\n'.join([
                '    '+mp.nstr(val, n=30, strip_zeros=False, show_zero_exponent=True, min_fixed=1, max_fixed=1)
                for val in comp
            ])+'\n  }'
            for comp in qr
        ])+'\n};'


def generate_qr_tria_cpp():
    src = '#include <cstddef>\n#include <tuple>\n\nnamespace numeric::math::quad::detail {\n\n'
    qrs = read_qrs_tria()
    for order, qr in qrs:
        src += f'static constexpr double qr_tria_{order}[3][{len(qr[0])}] = '
        src += qr_to_cpp_array(qr)
        src += '\n\n'
    src += 'std::tuple<size_t, const double *, const double *, const double *> get_qr_tria(size_t order) {\n'
    src += '  switch (order) {\n'
    for order, qr in qrs:
        src += f'    case {order}:\n'
        src += f'      return {{ {len(qr[0])}, qr_tria_{order}[0], qr_tria_{order}[1], qr_tria_{order}[2] }};\n'
    src += '    default:\n'
    src += '      return { 0, nullptr, nullptr, nullptr };\n'
    src += '  }\n'
    src += '}\n\n}'
    return src



print(generate_qr_tria_cpp())
