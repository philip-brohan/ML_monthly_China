#!/usr/bin/env python

# Plot validation statistics for all the test cases

import os
import sys
import numpy as np
import tensorflow as tf
import iris
import iris.fileformats
import iris.analysis
import datetime
from statistics import mean
import cmocean

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=100)
parser.add_argument(
    "--startyear", help="First year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--endyear", help="Last year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--xpoint",
    help="Extract data at this x point",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--ypoint",
    help="Extract data at this y point",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--anomalies",
    help="Make monthly anomalies",
    dest="anomalies",
    default=False,
    action="store_true",
)
args = parser.parse_args()

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from ERA5_monthly_load import lm_ERA5
from ERA5_monthly_load import load_climatology

sys.path.append("%s/../make_tensors" % os.path.dirname(__file__))
from tensor_utils import nPar

# Load the model specification
sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH
from localise import TSOURCE
from autoencoderModel import DCVAE
from makeDataset import getDataset

# Set up the test data
testData = getDataset(
    purpose="test", startyear=args.startyear, endyear=args.endyear, shuffle=False
)
testData = testData.batch(1)

autoencoder = DCVAE()
weights_dir = ("%s/models/Epoch_%04d") % (
    LSCRATCH,
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
load_status.assert_existing_objects_matched()


def unnormalise(value, variable):
    if not variable in nPar:
        raise Exception("Unsupported variable " + variable)
    value *= nPar[variable][1] - nPar[variable][0]
    value += nPar[variable][0]
    return value


def field_to_scalar(field, variable, month, mask, anomalies=False):
    f2 = unnormalise(field, variable)
    if anomalies:
        clim = load_climatology(variable, month)
        f2 -= clim.data
    if args.xpoint is not None and args.ypoint is not None:
        return f2[args.ypoint, args.xpoint]
    if args.xpoint is None and args.ypoint is None:
        if mask is not None:
            return np.mean(f2[np.where(mask == False)])
        return np.mean(f2)
    else:
        raise Exception("Must specify both xpoint and ypoint (or neither).")


# Get target and encoded statistics for one test case
def compute_stats(model, x):
    # get the date from the filename tensor
    fn = x[1].numpy()[0]
    year = int(fn[:4])
    month = int(fn[5:7])
    dtp = datetime.date(year, month, 15)
    # Pass the test field through the autoencoder
    mean, logvar = model.encode(x[0])
    latent = model.reparameterize(mean, logvar)
    generated = model.generate(latent)

    stats = {}
    stats["dtp"] = dtp
    stats["PRMSL_target"] = field_to_scalar(
        x[0][0, :, :, 0].numpy(),
        "mean_sea_level_pressure",
        month,
        None,
        anomalies=args.anomalies,
    )
    stats["PRMSL_model"] = field_to_scalar(
        generated[0, :, :, 0].numpy(),
        "mean_sea_level_pressure",
        month,
        None,
        anomalies=args.anomalies,
    )
    vt = x[0][0, :, :, 1].numpy()
    mask = lm_ERA5.data.mask
    stats["SST_target"] = field_to_scalar(
        vt, "sea_surface_temperature", month, mask, anomalies=args.anomalies
    )
    vm = generated[0, :, :, 1].numpy()
    stats["SST_model"] = field_to_scalar(
        vm, "sea_surface_temperature", month, mask, anomalies=args.anomalies
    )
    vt = x[0][0, :, :, 2].numpy()
    stats["T2M_target"] = field_to_scalar(
        vt, "2m_temperature", month, None, anomalies=args.anomalies
    )
    vm = generated[0, :, :, 2].numpy()
    stats["T2M_model"] = field_to_scalar(
        vm, "2m_temperature", month, None, anomalies=args.anomalies
    )
    vt = x[0][0, :, :, 3].numpy()
    stats["PRATE_target"] = field_to_scalar(
        vt, "total_precipitation", month, None, anomalies=args.anomalies
    )
    vm = generated[0, :, :, 3].numpy()
    stats["PRATE_model"] = field_to_scalar(
        vm, "total_precipitation", month, None, anomalies=args.anomalies
    )
    return stats


all_stats = {}
for case in testData:
    stats = compute_stats(autoencoder, case)
    for key in stats.keys():
        if key in all_stats:
            all_stats[key].append(stats[key])
        else:
            all_stats[key] = [stats[key]]


def to_monthly_anomalies(ts, monthly):
    climatology = []
    for m in range(13):
        climatology.append([])
    for idx in range(len(ts)):
        climatology[ts[idx].month].append(monthly[idx])
    for m in range(1, 13):
        climatology[m] = mean(climatology[m])
    result = []
    for idx in range(len(ts)):
        result.append(monthly[idx] - climatology[ts[idx].month])
    return result


# Plot sizes
tsh = 2
ssh = 2
tsw = 4
ssw = 2
hpad = 0.5
wpad = 0.5
fig_h = tsh * 2 + hpad * 3
fig_w = tsw * 2 + ssw * 2 + wpad * 5
# Make the plot
fig = Figure(
    figsize=(fig_w, fig_h),
    dpi=300,
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
    "size": 8,
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


def plot_var(ts, t, m, xp, yp, label):
    ymin = min(min(t), min(m))
    ymax = max(max(t), max(m))
    ypad = (ymax - ymin) * 0.1
    if ypad == 0:
        ypad = 1
    ax_ts = fig.add_axes(
        [
            (wpad / fig_w) * (2 * xp - 1) + (tsw + ssw) * (xp - 1) / fig_w,
            (hpad / fig_h) * yp + (tsh / fig_h) * (yp - 1),
            tsw / fig_w,
            tsh / fig_h,
        ],
        xlim=(
            ts[0] - datetime.timedelta(days=15),
            ts[-1] + datetime.timedelta(days=15),
        ),
        ylim=(ymin - ypad, ymax + ypad),
    )
    ax_ts.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
    ax_ts.text(
        ts[0] - datetime.timedelta(days=15),
        ymax + ypad,
        label,
        ha="left",
        va="top",
        bbox=dict(boxstyle="square,pad=0.5", fc=(1, 1, 1, 1)),
        zorder=100,
    )
    ax_ts.add_line(Line2D(ts, t, linewidth=1, color=(0, 0, 0, 1), alpha=1.0, zorder=50))
    ax_ts.add_line(Line2D(ts, m, linewidth=1, color=(1, 0, 0, 1), alpha=1.0, zorder=60))
    ax_sc = fig.add_axes(
        [
            (wpad / fig_w) * (2 * xp) + (tsw * xp + ssw * (xp - 1)) / fig_w,
            (hpad / fig_h) * yp + (tsh / fig_h) * (yp - 1),
            ssw / fig_w,
            tsh / fig_h,
        ],
        xlim=(ymin - ypad, ymax + ypad),
        ylim=(ymin - ypad, ymax + ypad),
    )
    ax_sc.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
    ax_sc.scatter(t, m, s=1, color=(1, 0, 0, 1), zorder=60)
    ax_sc.add_line(
        Line2D(
            (ymin - ypad, ymax + ypad),
            (ymin - ypad, ymax + ypad),
            linewidth=1,
            color=(0, 0, 0, 1),
            alpha=0.2,
            zorder=10,
        )
    )


# Top left - PRMSL
tsx = all_stats["dtp"]
ty = [x / 100 for x in all_stats["PRMSL_target"]]
my = [x / 100 for x in all_stats["PRMSL_model"]]
plot_var(tsx, ty, my, 1, 2, "PRMSL")

# Centre left - SST
if args.xpoint is None or lm_20CR.data.mask[args.ypoint, args.xpoint] == False:
    offset = 273.15
    if args.anomalies:
        offset = 0
    ty = [x - offset for x in all_stats["SST_target"]]
    my = [x - offset for x in all_stats["SST_model"]]
    plot_var(tsx, ty, my, 1, 1, "SST")

# Bottom left - T2M
if args.xpoint is None or dm_hukg.data.mask[args.ypoint, args.xpoint] == False:
    offset = 273.15
    if args.anomalies:
        offset = 0
    ty = [x - offset for x in all_stats["T2M_target"]]
    my = [x - offset for x in all_stats["T2M_model"]]
    plot_var(tsx, ty, my, 2, 1, "T2M")

# Top right - PRATE
if args.xpoint is None or dm_hukg.data.mask[args.ypoint, args.xpoint] == False:
    ty = [x * 1000 for x in all_stats["PRATE_target"]]
    my = [x * 1000 for x in all_stats["PRATE_model"]]
    plot_var(tsx, ty, my, 2, 2, "PRATE")


fig.savefig("multi.png")
