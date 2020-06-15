#!/usr/bin/env python

"""
Run add gamma to restart file in order to run Hvrd's plant water stress function
"""

__author__    = "MU Mengyuan"

import os
import sys
import glob
import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

def main(restart_fname):

    # create file and write global attributes
    f = nc.Dataset(restart_fname, "r+", format="NETCDF4") #append to add

    #ndim = 1
    #f.createDimension('mp', ndim)

    gamma = f.createVariable('gamma', 'f4', ('mp',))
    gamma.units = "-"
    gamma.long_name = "Parameter in root efficiency function (Lai and Katul 2000)"
    #gamma._FillValue = -1.e+33
    #gamma.missing_value = -1.e+33
    gamma[:] = 0.03

    f.close()

if __name__ == "__main__":

    restart_fname = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_31layers/run_heatwave_prescribe_soilmoist/restart_files/met_LAI_vrt_31uni_2013-2018-1-17_litter_gw-ssat-bom_Hvrd/EucFACE_amb_restart.nc"

    main(restart_fname)
