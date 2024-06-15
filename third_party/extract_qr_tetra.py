#!/usr/bin/env python3

import mpmath as mp
import sys
import os
import re

mp.mp.dps = 30


def extract_tables(data):
    expr = re.compile('\s*([xyzw]s)\(\s*(\d+)\)\s*=\s*([\+\-]?\d\.\d*D[\+\-]\d+)')
    arrays = {'xs':[], 'ys':[], 'zs':[], 'ws':[]}
    for line in data.split('\n'):
        res = expr.search(line)
        if res is not None:
            arr = res[1]
            idx = int(res[2])
            val = mp.mpf(res[3].replace('D','E'))
            if idx == 1:
                arrays[arr].append([])
            arrays[arr][-1].append(val)
    return arrays['xs'], arrays['ys'], arrays['zs'], arrays['ws']


def transform_to_standard(xs, ys, zs, ws):
    xt = [x+1 for x in xs]
    yt = [y+1/mp.sqrt(3) for y in ys]
    zt = [z+1/mp.sqrt(6) for z in zs]
    xo = [y/mp.sqrt(3)-z/(2*mp.sqrt(6)) for y,z in zip(yt,zt)]
    yo = [x/2-y/(2*mp.sqrt(3))-z/(2*mp.sqrt(6)) for x,y,z in zip(xt,yt,zt)]
    zo = [mp.sqrt(3)*z/(2*mp.sqrt(2)) for z in zt]
    wtot = sum(ws)
    wo = [w/(6*wtot) for w in ws]
    return xo, yo, zo, wo


def read_qrs_tetra():
    folder = os.path.dirname(os.path.abspath(sys.argv[0]))
    fname = os.path.join(folder, 'triasymq/tetraarbq.f')
    with open(fname, 'r') as f:
        data = f.read()
    all_xs, all_ys, all_zs, all_ws = extract_tables(data)
    qrs = []
    for i,(xs,ys,zs,ws) in enumerate(zip(all_xs, all_ys, all_zs, all_ws)):
        order = i + 1
        qrs.append((order, transform_to_standard(xs, ys, zs, ws)))
    return qrs


def qr_to_cpp_array(qr):
    return '{\n'+',\n'.join([
            '  {\n'+',\n'.join([
                '    '+mp.nstr(val, n=30, strip_zeros=False, show_zero_exponent=True, min_fixed=1, max_fixed=1)
                for val in comp
            ])+'\n  }'
            for comp in qr
        ])+'\n};'


def generate_qr_tetra_cpp():
    src = '#include <cstddef>\n#include <tuple>\n\nnamespace numeric::math::quad::detail {\n\n'
    qrs = read_qrs_tetra()
    for order, qr in qrs:
        src += f'static constexpr double qr_tetra_{order}[4][{len(qr[0])}] = '
        src += qr_to_cpp_array(qr)
        src += '\n\n'
    src += 'std::tuple<size_t, const double *, const double *, const double *, const double *> get_qr_tetra(size_t order) {\n'
    src += '  switch (order) {\n'
    for order, qr in qrs:
        src += f'    case {order}:\n'
        src += f'      return {{ {len(qr[0])}, qr_tetra_{order}[0], qr_tetra_{order}[1], qr_tetra_{order}[2], qr_tetra_{order}[3] }};\n'
    src += '    default:\n'
    src += '      return { 0, nullptr, nullptr, nullptr, nullptr };\n'
    src += '  }\n'
    src += '}\n\n}'
    return src


print(generate_qr_tetra_cpp())
