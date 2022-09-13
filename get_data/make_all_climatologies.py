#!/usr/bin/env python

# Make climatologies and sd climatologies

import os
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

for month in range(1, 13):
    for var in [
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "2m_temperature",
        "total_precipitation",
    ]:
        opfile = "%s/ERA5/monthly/climatology/%s_%02d.nc" % (
            os.getenv("SCRATCH"),
            var,
            month,
        )
        if not os.path.isfile(opfile):
            print(
                ("./make_climatology_for_month.py --month=%d --variable=%s")
                % (
                    month,
                    var,
                )
            )
        opfile = "%s/ERA5/monthly/sd_climatology/%s_%02d.nc" % (
            os.getenv("SCRATCH"),
            var,
            month,
        )
        if not os.path.isfile(opfile):
            print(
                ("./make_sd_climatology_for_month.py --month=%d --variable=%s")
                % (
                    month,
                    var,
                )
            )
