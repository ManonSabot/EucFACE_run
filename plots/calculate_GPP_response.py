#!/usr/bin/env python

"""
Plot visual benchmark (average seasonal cycle) of old vs new model runs.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (18.10.2017)"
__email__ = "mdekauwe@gmail.com"

import netCDF4 as nc
import matplotlib.pyplot as plt
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import os

def main(amb_fname, ele_fname):

    df_a = read_cable_file(amb_fname)
    df_e = read_cable_file(ele_fname)

    df_a = df_a[df_a.YEAR < 2017]
    df_e = df_e[df_e.YEAR < 2017]
    GPP_amb = df_a.groupby("YEAR").GPP.sum()
    GPP_ele = df_e.groupby("YEAR").GPP.sum()

    print(GPP_amb)
    print(GPP_ele)
    GPP_response = ((GPP_ele/GPP_amb)-1.0)*100.
    print(GPP_response)

def read_cable_file(fname):

    f = nc.Dataset(fname)
    time = nc.num2date(f.variables['time'][:],
                        f.variables['time'].units)
    df = pd.DataFrame(f.variables['GPP'][:,0,0], columns=['GPP'])
    df['Qle'] = f.variables['Qle'][:,0,0]
    df['LAI'] = f.variables['LAI'][:,0,0]
    df['TVeg'] = f.variables['TVeg'][:,0,0]
    df['ESoil'] = f.variables['ESoil'][:,0,0]
    df['CO2air'] = f.variables['CO2air'][:,0]

    df['dates'] = time
    df = df.set_index('dates')
    df['YEAR'] = df.index.year

    UMOL_TO_MOL = 1E-6
    MOL_C_TO_GRAMS_C = 12.0

    # umol/m2/s -> g/C/30min
    df['GPP'] *= UMOL_TO_MOL * MOL_C_TO_GRAMS_C * 1800.0

    return df


if __name__ == "__main__":

    amb_fname = "outputs/EucFACE_amb_out.nc"
    ele_fname = "outputs/EucFACE_ele_out.nc"
    main(amb_fname, ele_fname)
