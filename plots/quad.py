#!/usr/bin/env python

# plot ERA5 monthly fields, for all four variables, for the China region

import os
import sys
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import iris
import iris.analysis
import cmocean

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")

sys.path.append("%s/../get_data" % os.path.dirname(__file__))
from ERA5_monthly_load import load_variable
from ERA5_monthly_load import load_climatology
from ERA5_monthly_load import lm_plot
from ERA5_monthly_load import get_range

sys.path.append("%s/." % os.path.dirname(__file__))
from plot_variable import plotFieldAxes

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument(
    "--actuals",
    help="Show actuals (rather than anomalies)",
    dest="actuals",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--opdir", help="Output directory", type=str, required=False, default="."
)
parser.add_argument(
    "--opfile", help="Output file name", type=str, required=False, default=None
)
args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

if args.opfile is None:
    args.opfile = "Quad_%04d-%02d.png" % (args.year, args.month)

fig = Figure(
    figsize=(30, 22),
    dpi=100,
    facecolor=(0.5, 0.5, 0.5, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)
font = {
    "family": "sans-serif",
    "sans-serif": "Arial",
    "weight": "normal",
    "size": 20,
}
matplotlib.rc("font", **font)
axb = fig.add_axes([0, 0, 1, 1])
axb.add_patch(
    Rectangle(
        (0, 0),
        1,
        1,
        facecolor=(0.95, 0.95, 0.95, 1),
        fill=True,
        zorder=1,
    )
)


# Top left - PRMSL
var = load_variable("mean_sea_level_pressure", args.year, args.month)
if args.actuals:
    (dmin, dmax) = get_range("mean_sea_level_pressure", args.month, anomalies=False)
else:
    clim = load_climatology("mean_sea_level_pressure", args.month)
    var.data -= clim.data
    (dmin, dmax) = get_range("mean_sea_level_pressure", args.month, anomalies=True)
var /= 100
dmin /= 100
dmax /= 100
ax_prmsl = fig.add_axes([0.025 / 2, 0.125 / 2 + 0.5, 0.95 / 2, 0.85 / 2])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.diff,
)
ax_prmsl_cb = fig.add_axes([0.125 / 2, 0.05 / 2 + 0.5, 0.75 / 2, 0.05 / 2])
ax_prmsl_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_img, ax=ax_prmsl_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Bottom left - SST
var = load_variable("sea_surface_temperature", args.year, args.month)
if args.actuals:
    (dmin, dmax) = get_range("sea_surface_temperature", args.month, anomalies=False)
    var -= 273.15
    dmin -= 273.15
    dmax -= 273.15
else:
    clim = load_climatology("sea_surface_temperature", args.month)
    var.data -= clim.data
    (dmin, dmax) = get_range("sea_surface_temperature", args.month, anomalies=True)
ax_sst = fig.add_axes([0.025 / 2, 0.125 / 2, 0.95 / 2, 0.85 / 2])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_sst_cb = fig.add_axes([0.125 / 2, 0.05 / 2, 0.75 / 2, 0.05 / 2])
ax_sst_cb.set_axis_off()
cb = fig.colorbar(
    SST_img, ax=ax_sst_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top right - PRATE
var = load_variable("cbrt_precipitation", args.year, args.month)
if args.actuals:
    (dmin, dmax) = get_range("cbrt_precipitation", args.month, anomalies=False)
    dmin = 0
    cmap = cmocean.cm.rain
else:
    clim = load_climatology("cbrt_precipitation", args.month)
    var.data -= clim.data
    (dmin, dmax) = get_range("cbrt_precipitation", args.month, anomalies=True)
    cmap = cmocean.cm.tarn
ax_prate = fig.add_axes([0.025 / 2 + 0.5, 0.125 / 2 + 0.5, 0.95 / 2, 0.85 / 2])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmap,
)
ax_prate_cb = fig.add_axes([0.125 / 2 + 0.5, 0.05 / 2 + 0.5, 0.75 / 2, 0.05 / 2])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)
# Bottom left - T2m
var = load_variable("2m_temperature", args.year, args.month)
if args.actuals:
    (dmin, dmax) = get_range("2m_temperature", args.month, anomalies=False)
    var -= 273.15
    dmin -= 273.15
    dmax -= 273.15
else:
    clim = load_climatology("2m_temperature", args.month)
    var.data -= clim.data
    (dmin, dmax) = get_range("2m_temperature", args.month, anomalies=True)
ax_tmp2m = fig.add_axes([0.025 / 2 + 0.5, 0.125 / 2, 0.95 / 2, 0.85 / 2])
ax_tmp2m.set_axis_off()
TMP2m_img = plotFieldAxes(
    ax_tmp2m,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_tmp2m_cb = fig.add_axes([0.125 / 2 + 0.5, 0.05 / 2, 0.75 / 2, 0.05 / 2])
ax_tmp2m_cb.set_axis_off()
cb = fig.colorbar(
    TMP2m_img, ax=ax_tmp2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

# Output as png
fig.savefig("%s/%s" % (args.opdir, args.opfile))
