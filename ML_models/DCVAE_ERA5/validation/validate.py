#!/usr/bin/env python

# Plot a validation figure for the autoencoder.

# For each variable:
#  1) Input field
#  2) Autoencoder output
#  3) scatter plot
#

import os
import sys
import numpy as np
import tensorflow as tf
import iris
import iris.fileformats
import iris.analysis
import cmocean

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=100)
parser.add_argument("--year", help="Test year", type=int, required=False, default=1969)
parser.add_argument("--month", help="Test month", type=int, required=False, default=3)
parser.add_argument(
    "--anomalies",
    help="Make monthly anomalies",
    dest="anomalies",
    default=False,
    action="store_true",
)
args = parser.parse_args()

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from ERA5_monthly_load import load_cList
from ERA5_monthly_load import sCube
from ERA5_monthly_load import lm_plot
from ERA5_monthly_load import lm_ERA5
from ERA5_monthly_load import load_climatology
from ERA5_monthly_load import get_range

sys.path.append("%s/../make_tensors" % os.path.dirname(__file__))
from tensor_utils import cList_to_tensor
from tensor_utils import normalise
from tensor_utils import unnormalise

sys.path.append("%s/../../../plots" % os.path.dirname(__file__))
from plot_variable import plotFieldAxes
from plot_variable import plotScatterAxes

# Load and standardise data
qd = load_cList(
    args.year,
    args.month,
)
ic = cList_to_tensor(qd, lm_ERA5.data.mask, extrapolate=True)

# Load the model specification
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("%s/models/Epoch_%04d") % (
    LSCRATCH,
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
load_status.assert_existing_objects_matched()

# Get autoencoded tensors
encoded = autoencoder.call(tf.reshape(ic, [1, 200, 320, 4]))

# Make the plot
fig = Figure(
    figsize=(20, 22),
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
axb.set_axis_off()
axb.add_patch(
    Rectangle(
        (0, 0),
        1,
        1,
        facecolor=(1.0, 1.0, 1.0, 1),
        fill=True,
        zorder=1,
    )
)


# Top left - PRMSL original
varx = sCube.copy()
varx.data = np.squeeze(ic[:, :, 0].numpy())
varx = unnormalise(varx, "mean_sea_level_pressure")
if args.anomalies:
    clim = load_climatology("mean_sea_level_pressure", args.month)
    varx.data -= clim.data
    varx /= 100
    (dmin, dmax) = (-5, 5)
else:
    varx /= 100
    (dmin, dmax) = get_range("mean_sea_level_pressure", args.month)
    dmin /= 100
    dmax /= 100
ax_prmsl = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.75, 0.95 / 3, 0.85 / 4])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.diff,
)
ax_prmsl_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.75 + 0.01, 0.75 / 3, 0.05 / 4])
ax_prmsl_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_img, ax=ax_prmsl_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top centre - PRMSL encoded
vary = sCube.copy()
vary.data = np.squeeze(encoded[0, :, :, 0].numpy())
vary = unnormalise(vary, "mean_sea_level_pressure")
if args.anomalies:
    clim = load_climatology("mean_sea_level_pressure", args.month)
    vary.data -= clim.data
    vary /= 100
else:
    vary /= 100
ax_prmsl_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.75, 0.95 / 3, 0.85 / 4])
ax_prmsl_e.set_axis_off()
PRMSL_e_img = plotFieldAxes(
    ax_prmsl_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.diff,
)
ax_prmsl_e_cb = fig.add_axes(
    [0.125 / 3 + 1 / 3, 0.05 / 4 + 0.75 + 0.01, 0.75 / 3, 0.05 / 4]
)
ax_prmsl_e_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_e_img,
    ax=ax_prmsl_e_cb,
    location="bottom",
    orientation="horizontal",
    fraction=1.0,
)

# Top right - PRMSL scatter
ax_prmsl_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.75, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_prmsl_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# 2nd left - PRATE original
varx.data = np.squeeze(ic[:, :, 3].numpy())
varx = unnormalise(varx, "total_precipitation")
if args.anomalies:
    clim = load_climatology("total_precipitation", args.month)
    varx.data -= clim.data
    varx *= 1000
    (dmin, dmax) = (-5, 5)
    pcmap = cmocean.cm.tarn
else:
    varx *= 1000
    (dmin, dmax) = get_range("total_precipitation", args.month)
    dmin = 0
    dmax *= 1000
    pcmap = cmocean.cm.rain
ax_prate = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=pcmap,
)
ax_prate_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.5 + 0.01, 0.75 / 3, 0.05 / 4])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - PRATE encoded
vary.data = np.squeeze(encoded[0, :, :, 3].numpy())
vary = unnormalise(vary, "total_precipitation")
if args.anomalies:
    clim = load_climatology("total_precipitation", args.month)
    vary.data -= clim.data
    vary *= 1000
else:
    vary *= 1000
ax_prate_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_prate_e.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=pcmap,
)
ax_prate_e_cb = fig.add_axes(
    [0.125 / 3 + 1 / 3, 0.05 / 4 + 0.5 + 0.01, 0.75 / 3, 0.05 / 4]
)
ax_prate_e_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_e_img,
    ax=ax_prate_e_cb,
    location="bottom",
    orientation="horizontal",
    fraction=1.0,
)

# 2nd right - PRATE scatter
ax_prate_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.5, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_prate_s, varx, vary, vMin=min(dmin, 0.001), vMax=dmax, bins=None)


# 3rd left - T2m original
varx.data = np.squeeze(ic[:, :, 2].numpy())
varx = unnormalise(varx, "2m_temperature")
if args.anomalies:
    clim = load_climatology("2m_temperature", args.month)
    varx.data -= clim.data
    (dmin, dmax) = (-5, 5)
else:
    varx -= 273.15
    (dmin, dmax) = get_range("2m_temperature", args.month)
    dmin -= 273.15 + 2
    dmax -= 273.15 - 2
ax_t2m = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.25, 0.95 / 3, 0.85 / 4])
ax_t2m.set_axis_off()
T2m_img = plotFieldAxes(
    ax_t2m,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_t2m_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.25 + 0.01, 0.75 / 3, 0.05 / 4])
ax_t2m_cb.set_axis_off()
cb = fig.colorbar(
    T2m_img, ax=ax_t2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd centre - T2m encoded
vary.data = np.squeeze(encoded[0, :, :, 2].numpy())
vary = unnormalise(vary, "2m_temperature")
if args.anomalies:
    clim = load_climatology("2m_temperature", args.month)
    vary.data -= clim.data
else:
    vary -= 273.15
ax_t2m_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.25, 0.95 / 3, 0.85 / 4])
ax_t2m_e.set_axis_off()
T2m_e_img = plotFieldAxes(
    ax_t2m_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_t2m_e_cb = fig.add_axes(
    [0.125 / 3 + 1 / 3, 0.05 / 4 + 0.25 + 0.01, 0.75 / 3, 0.05 / 4]
)
ax_t2m_e_cb.set_axis_off()
cb = fig.colorbar(
    T2m_e_img, ax=ax_t2m_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd right - T2m scatter
ax_t2m_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.25, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_t2m_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# Bottom left - SST original
varx.data = np.squeeze(ic[:, :, 1].numpy())
varx.data = np.ma.masked_where(lm_ERA5.data.mask, varx.data, copy=False)
varx = unnormalise(varx, "sea_surface_temperature")
if args.anomalies:
    clim = load_climatology("sea_surface_temperature", args.month)
    varx.data -= clim.data
    (dmin, dmax) = (-5, 5)
else:
    varx -= 273.15
    (dmin, dmax) = get_range("sea_surface_temperature", args.month)
    dmin -= 273.15 + 2
    dmax -= 273.15 - 2
ax_sst = fig.add_axes([0.025 / 3, 0.125 / 4, 0.95 / 3, 0.85 / 4])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_sst_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.01, 0.75 / 3, 0.05 / 4])
ax_sst_cb.set_axis_off()
cb = fig.colorbar(
    SST_img, ax=ax_sst_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - SST encoded
vary.data = encoded.numpy()[0, :, :, 1]
vary.data = np.ma.masked_where(lm_ERA5.data.mask, vary.data, copy=False)
vary = unnormalise(vary, "sea_surface_temperature")
if args.anomalies:
    clim = load_climatology("sea_surface_temperature", args.month)
    vary.data -= clim.data
else:
    vary -= 273.15
ax_sst_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4, 0.95 / 3, 0.85 / 4])
ax_sst_e.set_axis_off()
SST_e_img = plotFieldAxes(
    ax_sst_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_sst_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4 + 0.01, 0.75 / 3, 0.05 / 4])
ax_sst_e_cb.set_axis_off()
cb = fig.colorbar(
    SST_e_img, ax=ax_sst_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd right - SST scatter
ax_sst_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_sst_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


fig.savefig("comparison.png")
