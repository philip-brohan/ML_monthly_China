#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lat", help="Latitude", type=float, required=True)
parser.add_argument("--lon", help="Longitude", type=float, required=True)
args = parser.parse_args()

x_grid_point = int((args.lon - 59.825) / 0.25)
y_grid_point = int((args.lat - 9.825) / 0.25)

print("x grid point: %4d" % x_grid_point)
print("y grid point: %4d" % y_grid_point)
