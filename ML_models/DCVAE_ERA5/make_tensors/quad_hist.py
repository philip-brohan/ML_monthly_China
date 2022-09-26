#!/usr/bin/env python

# plot histograms of the normalised tensor data

import os
import sys
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import iris
import iris.analysis
import numpy as np
import cmocean

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from ERA5_monthly_load import load_variable
from ERA5_monthly_load import load_climatology
from ERA5_monthly_load import lm_plot
from ERA5_monthly_load import get_range

sys.path.append("%s/../../../plots" % os.path.dirname(__file__))
from plot_variable import plotFieldAxes

sys.path.append("%s/." % os.path.dirname(__file__))
from tensor_utils import normalise

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
    args.opfile = "Hist.png"

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
clim = load_climatology("mean_sea_level_pressure", args.month)
var.data -= clim.data
d1 = normalise(var, "mean_sea_level_pressure")
ax_prmsl = fig.add_axes(
    [0.025 / 2 + 0.025, 0.125 / 2 + 0.5, 0.95 / 2 - 0.025, 0.85 / 2]
)
ax_prmsl.set_xlabel("PRMSL")
PRMSL_hist = ax_prmsl.hist(d1.data.flatten(), bins=100, range=(0, 1), zorder=50)

# Bottom left - SST
var = load_variable("sea_surface_temperature", args.year, args.month)
clim = load_climatology("sea_surface_temperature", args.month)
var.data -= clim.data
d1 = normalise(var, "sea_surface_temperature")
ax_sst = fig.add_axes([0.025 / 2 + 0.025, 0.125 / 2, 0.95 / 2 - 0.025, 0.85 / 2])
ax_sst.set_xlabel("SST")
SST_hist = ax_sst.hist(d1.data.flatten(), bins=100, range=(0, 1))

# Top right - PRATE
var = load_variable("total_precipitation", args.year, args.month)
clim = load_climatology("total_precipitation", args.month)
var.data -= clim.data
d1 = normalise(var, "total_precipitation")
ax_prate = fig.add_axes(
    [0.025 / 2 + 0.5 + 0.025, 0.125 / 2 + 0.5, 0.95 / 2 - 0.025, 0.85 / 2]
)
ax_prate.set_xlabel("PRATE")
PRATE_hist = ax_prate.hist(d1.data.flatten(), bins=100, range=(0, 1))

# Bottom left - T2m
var = load_variable("2m_temperature", args.year, args.month)
clim = load_climatology("2m_temperature", args.month)
var.data -= clim.data
d1 = normalise(var, "2m_temperature")
ax_tmp2m = fig.add_axes(
    [0.025 / 2 + 0.5 + 0.025, 0.125 / 2, 0.95 / 2 - 0.025, 0.85 / 2]
)
ax_tmp2m.set_xlabel("T2M")
T2M_hist = ax_tmp2m.hist(d1.data.flatten(), bins=100, range=(0, 1))

if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

# Output as png
fig.savefig("%s/%s" % (args.opdir, args.opfile))
