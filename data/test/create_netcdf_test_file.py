#!/usr/bin/env python3

import netCDF4 as nc
import numpy as np


with nc.Dataset('test_file.nc', 'w') as f:
    f.createDimension('N', 5)
    var_float = f.createVariable('var_float', np.float32, ('N','N','N'))
    var_float[:] = np.random.random((5,5,5))
    var_int = f.createVariable('var_int', np.int32, ('N',))
    var_int[:] = np.random.randint(0, 10, (5,), dtype=np.int32)
    grp_a = f.createGroup('grp_a')
    grp_a.createDimension('M', 10)
    var_float = grp_a.createVariable('var_float', np.float32, ('M','N'))
    var_float[:] = np.random.random((10,5))
    var_int = grp_a.createVariable('var_int', np.int32, ('M',))
    var_int[:] = np.random.randint(0, 10, (10,), dtype=np.int32)
