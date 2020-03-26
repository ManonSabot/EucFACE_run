#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-10-06"
__changefrom__ = 'plot_eucface_swc_cable_vs_obs.py'

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import griddata
import scipy.stats as stats
from sklearn.metrics import mean_squared_error

def main(fmets):

    mets1 = nc.Dataset(fmets[0], 'r')
    mets2 = nc.Dataset(fmets[1], 'r')
    mets3 = nc.Dataset(fmets[2], 'r')
    mets4 = nc.Dataset(fmets[3], 'r')
    mets5 = nc.Dataset(fmets[4], 'r')
    mets6 = nc.Dataset(fmets[5], 'r')

    soil = pd.DataFrame(mets1.variables['sand_vec'][:,0,0],columns=['sand'])
    soil['silt'] = mets1.variables['silt_vec'][:,0,0]
    soil['clay'] = mets1.variables['clay_vec'][:,0,0]

    soil['sand'] = (soil['sand'] + mets2.variables['sand_vec'][:,0,0]
                    + mets3.variables['sand_vec'][:,0,0]
                    + mets4.variables['sand_vec'][:,0,0]
                    + mets5.variables['sand_vec'][:,0,0]
                    + mets6.variables['sand_vec'][:,0,0])/ 6.
    soil['silt'] = (soil['silt'] + mets2.variables['silt_vec'][:,0,0]
                    + mets3.variables['silt_vec'][:,0,0]
                    + mets4.variables['silt_vec'][:,0,0]
                    + mets5.variables['silt_vec'][:,0,0]
                    + mets6.variables['silt_vec'][:,0,0])/ 6.
    soil['clay'] = (soil['clay'] + mets2.variables['clay_vec'][:,0,0]
                    + mets3.variables['clay_vec'][:,0,0]
                    + mets4.variables['clay_vec'][:,0,0]
                    + mets5.variables['clay_vec'][:,0,0]
                    + mets6.variables['clay_vec'][:,0,0])/ 6.

    plt.plot(soil['sand'], c="orange")
    plt.plot(soil['silt'], c="brown")
    plt.plot(soil['clay'], c="blue")
    
    plt.show()

if __name__ == "__main__":

    fmets = ["/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI_vrt_swilt-watr-ssat_SM_31uni/EucFACE_met_R1.nc",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI_vrt_swilt-watr-ssat_SM_31uni/EucFACE_met_R2.nc",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI_vrt_swilt-watr-ssat_SM_31uni/EucFACE_met_R3.nc",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI_vrt_swilt-watr-ssat_SM_31uni/EucFACE_met_R4.nc",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI_vrt_swilt-watr-ssat_SM_31uni/EucFACE_met_R5.nc",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI_vrt_swilt-watr-ssat_SM_31uni/EucFACE_met_R6.nc"]

    main(fmets)
