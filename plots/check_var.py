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
import datetime

def main(fname, var_name):

    df = read_cable_var(fname, var_name)

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax1 = fig.add_subplot(111)
    ax1.plot(df.index, df.cable.cumsum(), c="green", lw=2.0, ls="-")
    ax1.set_title(var_name, fontsize=12)

    plt.show()


def read_cable_var(output_file, var_name):

    """
    read transpiration or soil evaporation from CABLE output
    """

    print("carry on read_cable_var")
    cable = nc.Dataset(output_file, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    var = pd.DataFrame(cable.variables[var_name][:,0,0]*1800., columns=['cable'])

    var['Date'] = Time
    var = var.set_index('Date')
    var = var.resample("D").agg('sum')
    var.index = var.index - pd.datetime(2011,12,31)
    var.index = var.index.days
    var = var.sort_values(by=['Date'])
    print(var)

    return var

if __name__ == "__main__":

    #fname = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_GW-wb_SM-fix_or_fix_fw-hie-exp/EucFACE_amb_out.nc"
    fname = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_bch-hyds-top1/fix_hie_exp/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=40-hyds^-20_fw-hie-exp_fix/EucFACE_amb_out.nc"
    var_name = 'Qs'#"Qrecharge"
    main(fname, var_name)
