#!/usr/bin/env python

# We need to normalise the data - to map values on the range 0-1
# Estimate scale parameters corresponding to 0 and 1.
# (Max and min values of data, with a bit of wiggle room).

import os
import sys
import iris
import iris.analysis
import numpy as np
import argparse

sys.path.append("%s/." % os.path.dirname(__file__))
from ERA5_monthly_load import load_variable

parser = argparse.ArgumentParser()
parser.add_argument("--variable", help="Variable name", type=str, required=True)
args = parser.parse_args()

pc = plot_cube()
smax = -1000000.0
smin = 1000000.0
for year in range(1981, 2011):
    for month in range(1, 13):
        var = load_variable(args.variable, year, month)
        vmax = np.amax(var.data)
        vmin = np.amin(var.data)
        smax = max(smax, vmax)
        smin = min(smin, vmin)

print(smin, smax)
