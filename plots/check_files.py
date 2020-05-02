#!/usr/bin/env python

"""
Turn the MAESPA input file into a CABLE netcdf file. Aim to swap MAESPA data
for the raw data later when I have more time...

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (04.08.2018)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt

fname = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_Hvrd/EucFACE_amb_out.nc"
ds = xr.open_dataset(fname)
#plt.plot(ds.Tair[:,0,0])
plt.plot(ds.WatTable[:,0,0])
plt.savefig("check_met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_Hvrd_WatTable.pdf", bbox_inches='tight', pad_inches=0.1)
