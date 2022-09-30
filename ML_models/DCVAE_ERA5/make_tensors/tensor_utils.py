# Utility functions for creating and manipulating tensors

import os
import iris
import iris.util
import iris.coord_systems
import tensorflow as tf
import numpy as np

# Want a smooth transition at the edge of the mask, so
#  set masked points near data points to close to the data value
def extrapolate_step(cb, cm, scale=1.0):
    ss = cb.data * 0
    sn = cb.data * 0
    count = cb.data * 0
    for ad in ([1, 0], [-1, 0], [0, 1], [0, -1]):
        sn = np.roll(cb.data, (ad[0], ad[1]), (0, 1))
        sn[0, :] = 0
        sn[-1, :] = 0
        sn[:, 0] = 0
        sn[:, -1] = 0
        ss[sn != 0] += sn[sn != 0]
        count[sn != 0] += 1
    ss[count != 0] /= count[count != 0]
    result = cb.copy()
    result.data[cm.data == 0] = ss[cm.data == 0]
    return result


def extrapolate_missing(cb, nsteps=10, scale=1.0):
    cr = cb.copy()
    for step in range(nsteps):
        cr = extrapolate_step(cr, cb, scale=scale)
    return cr


def cList_to_tensor(cL, sst_mask, extrapolate=True):
    d1 = normalise(cL[0], "mean_sea_level_pressure")
    d2 = normalise(cL[1], "sea_surface_temperature")
    d2.data[np.where(sst_mask == True)] = 0
    if extrapolate:
        d2 = extrapolate_missing(d2, nsteps=100, scale=1.0)
    d3 = normalise(cL[2], "2m_temperature")
    d4 = normalise(cL[3], "cbrt_precipitation")
    ic = np.stack((d1.data, d2.data, d3.data, d4.data), axis=2)
    ict = tf.convert_to_tensor(ic.data, np.float32)
    return ict


def tensor_to_cList(tensor, plotCube, sst_mask):
    d1 = plotCube.copy()
    d1.data = np.squeeze(tensor[:, :, 0].numpy())
    d1 = unnormalise(d1, "mean_sea_level_pressure")
    d1.var_name = "mean_sea_level_pressure"
    d2 = plotCube.copy()
    d2.data = np.squeeze(tensor[:, :, 1].numpy())
    d2 = unnormalise(d2, "sea_surface_temperature")
    d2.data = np.ma.masked_where(sst_mask, d2.data, copy=False)
    d2.var_name = "sea_surface_temperature"
    d3 = plotCube.copy()
    d3.data = np.squeeze(tensor[:, :, 2].numpy())
    d3 = unnormalise(d3, "2m_temperature")
    d3.var_name = "2m_temperature"
    d4 = plotCube.copy()
    d4.data = np.squeeze(tensor[:, :, 3].numpy())
    d4 = unnormalise(d4, "cbrt_precipitation")
    d4.var_name = "cbrt_precipitation"
    return [d1, d2, d3, d4]


nPar = {
    "mean_sea_level_pressure": (-750, 750),
    "total_precipitation": (-0.01, 0.01),
    "cbrt_precipitation": (-0.1, 0.1),
    "2m_temperature": (-20, 10),
    "sea_surface_temperature": (-3, 3),
}


def normalise(cube, variable):
    cb = cube.copy()
    if not variable in nPar:
        raise Exception("Unsupported variable " + variable)
    cb.data -= nPar[variable][0]
    cb.data /= nPar[variable][1] - nPar[variable][0]
    return cb


def unnormalise(cube, variable):
    cb = cube.copy()
    if not variable in nPar:
        raise Exception("Unsupported variable " + variable)
    cb.data *= nPar[variable][1] - nPar[variable][0]
    cb.data += nPar[variable][0]
    return cb
