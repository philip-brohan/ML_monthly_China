# Functions to load ERA5 monthly data

import os
import iris
import iris.util
import iris.cube
import iris.analysis
import iris.coord_systems
import iris.fileformats
import numpy as np
from calendar import monthrange

# Define a standard-cube to work with
# Same projection as the raw data, except that the grid is centered on 35N, 100E,
#  and trimmed to 200x320 to be multiply divisible by 2 for easy use
#  in a hierarchical CNN.
cs_ERA5 = iris.coord_systems.RotatedGeogCS(90, 180, 0)
x_coord = iris.coords.DimCoord(
    np.arange(60, 140, 0.25),
    standard_name="longitude",
    units="degrees",
    coord_system=cs_ERA5,
)
x_coord.guess_bounds()
y_coord = iris.coords.DimCoord(
    np.arange(10, 60, 0.25),
    standard_name="latitude",
    units="degrees",
    coord_system=cs_ERA5,
)
y_coord.guess_bounds()
dummy_data = np.zeros((len(y_coord.points), len(x_coord.points)))
sCube = iris.cube.Cube(dummy_data, dim_coords_and_dims=[(x_coord, 1), (y_coord, 0)])


# Also want a land mask for plotting:
lm_plot = iris.load_cube(
    "%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv("DATADIR")
)
lm_plot = lm_plot.extract(
    iris.Constraint(latitude=lambda cell: 10 <= cell <= 60)
    & iris.Constraint(longitude=lambda cell: 60 <= cell <= 140)
)

# And a land-mask for ERA5 SST grid
fname = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
    os.getenv("SCRATCH"),
    1959,
    "sea_surface_temperature",
)
if not os.path.isfile(fname):
    raise Exception("No data file %s" % fname)
ftt = iris.Constraint(time=lambda cell: cell.point.month == 1)
lm_ERA5 = iris.load_cube(fname, ftt)
lm_ERA5.coord("latitude").coord_system = cs_ERA5
lm_ERA5.coord("longitude").coord_system = cs_ERA5
lm_ERA5 = lm_ERA5.regrid(sCube, iris.analysis.Nearest())
lm_ERA5.data.data[np.where(lm_ERA5.data.mask == True)] = 0
lm_ERA5.data.data[np.where(lm_ERA5.data.mask == False)] = 1


def load_variable(variable, year, month):
    fname = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
        os.getenv("SCRATCH"),
        year,
        variable,
    )
    if not os.path.isfile(fname):
        raise Exception("No data file %s" % fname)
    ftt = iris.Constraint(time=lambda cell: cell.point.month == month)
    varC = iris.load_cube(fname, ftt)
    if len(varC.data.shape) == 3:
        varC = varC.extract(iris.Constraint(expver=1))
    varC.coord("latitude").coord_system = cs_ERA5
    varC.coord("longitude").coord_system = cs_ERA5
    varC = varC.regrid(sCube, iris.analysis.Nearest())
    return varC


def load_cList(year, month):
    res = []
    for variable in (
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "2m_temperature",
        "total_precipitation",
    ):
        var = load_variable(variable, year, month)
        clim = load_climatology(variable, month)
        var.data -= clim.data
        res.append(var)
    return res


def load_climatology(variable, month):
    fname = "%s/ERA5/monthly/climatology/%s_%02d.nc" % (
        os.getenv("SCRATCH"),
        variable,
        month,
    )
    if not os.path.isfile(fname):
        raise Exception("No climatology file %s" % fname)
    c = iris.load_cube(fname)
    c.long_name = variable
    c.coord("latitude").coord_system = cs_ERA5
    c.coord("longitude").coord_system = cs_ERA5
    return c


def load_sd_climatology(variable, month):
    fname = "%s/ERA5/monthly/sd_climatology/%s_%02d.nc" % (
        os.getenv("SCRATCH"),
        variable,
        month,
    )
    if not os.path.isfile(fname):
        raise Exception("No sd climatology file %s" % fname)
    c = iris.load_cube(fname)
    c.long_name = variable
    c.coord("latitude").coord_system = cs_ERA5
    c.coord("longitude").coord_system = cs_ERA5
    return c


def get_range(variable, month, anomalies=True):
    sdc = load_sd_climatology(variable, month)
    if anomalies:
        v= sdc.data*2
        dmax = np.percentile(v.data[v.mask == False], 95)
        dmin = dmax*-1
    else:
        dc = load_climatology(variable, month)
        v = dc.data + (sdc.data * 2)
        dmax = np.percentile(v.data[v.mask == False], 95)
        v = dc.data - (sdc.data * 2)
        dmin = np.percentile(v.data[v.mask == False], 5)
    return (dmin, dmax)
