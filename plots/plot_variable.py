# Functions to plot ERA5 monthly data for the China region

import os
import numpy as np

import iris
import matplotlib

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import cmocean


def plotFieldAxes(
    ax_map,
    field,
    vMax=None,
    vMin=None,
    lMask=None,
    cMap=cmocean.cm.balance,
):

    if vMax is None:
        vMax = np.max(field.data)
    if vMin is None:
        vMin = np.min(field.data)

    lons = field.coord("longitude").points
    lats = field.coord("latitude").points
    ax_map.set_ylim(min(lats), max(lats))
    ax_map.set_xlim(min(lons), max(lons))
    ax_map.set_axis_off()
    ax_map.set_aspect("equal", adjustable="box", anchor="C")
    ax_map.add_patch(
        Rectangle(
            (min(lons), min(lats)),
            max(lons) - min(lons),
            max(lats) - min(lats),
            facecolor=(0.9, 0.9, 0.9, 1),
            fill=True,
            zorder=1,
        )
    )
    # Plot the field
    T_img = ax_map.pcolorfast(
        lons,
        lats,
        field.data,
        cmap=cMap,
        vmin=vMin,
        vmax=vMax,
        alpha=1.0,
        zorder=10,
    )

    # Overlay the land mask
    if lMask is not None:
        mask_img = ax_map.pcolorfast(
            lMask.coord("longitude").points,
            lMask.coord("latitude").points,
            lMask.data,
            cmap=matplotlib.colors.ListedColormap(
                ((0.4, 0.4, 0.4, 0), (0.4, 0.4, 0.4, 0.3))
            ),
            vmin=0,
            vmax=1,
            alpha=1.0,
            zorder=100,
        )

    return T_img


def plotScatterAxes(
    ax, var_in, var_out, vMax=None, vMin=None, xlabel="", ylabel="", bins="log"
):
    if vMax is None:
        vMax = max(np.max(var_in.data), np.max(var_out.data))
    if vMin is None:
        vMin = min(np.min(var_in.data), np.min(var_out.data))
    ax.set_xlim(vMin, vMax)
    ax.set_ylim(vMin, vMax)
    ax.hexbin(
        x=var_in.data.flatten(),
        y=var_out.data.flatten(),
        cmap=cmocean.cm.ice_r,
        bins=bins,
        gridsize=50,
        mincnt=1,
    )
    ax.add_line(
        Line2D(
            xdata=(vMin, vMax),
            ydata=(vMin, vMax),
            linestyle="solid",
            linewidth=0.5,
            color=(0.5, 0.5, 0.5, 1),
            zorder=100,
        )
    )
    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.grid(color="black", alpha=0.2, linestyle="-", linewidth=0.5)
