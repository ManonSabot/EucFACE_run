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

def main(amb_fname, ele_fname, met_fname):

    df_a = read_cable_file(amb_fname)
    df_e = read_cable_file(ele_fname)
    df_m = read_cable_met_file(amb_fname)

    df_a = df_a.between_time("06:00", "19:00")
    df_e = df_e.between_time("06:00", "19:00")
    df_m = df_m.between_time("06:00", "19:00")

    cable_tcana = df_a.CanT - 273.15
    cable_tcane = df_e.CanT - 273.15
    met_tair = df_m.Tair - 273.15

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

    ax1 = fig.add_subplot(1,1,1)

    ax1.plot(met_tair-cable_tcana, color="blue", label="Amb")
    ax1.axhline(y=np.mean(met_tair-cable_tcana), ls="-", color="k")
    ax1.axhline(y=np.mean(met_tair-cable_tcana) +\
                np.std(met_tair-cable_tcana), ls="--", color="k")
    ax1.axhline(y=np.mean(met_tair-cable_tcana) -\
                np.std(met_tair-cable_tcana), ls="--", color="k")
    ax1.set_ylabel("Tair-Tcanopy (deg C)")

    plot_fname = "Tair_minus_Tcan.pdf"
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig.savefig(os.path.join(plot_dir, plot_fname), bbox_inches='tight',
                pad_inches=0.1)

def read_cable_met_file(fname):

    f = nc.Dataset(fname)
    time = nc.num2date(f.variables['time'][:],
                        f.variables['time'].units)
    df = pd.DataFrame(f.variables['Tair'][:,0,0], columns=['Tair'])

    df['dates'] = time
    df = df.set_index('dates')

    return df

def read_cable_file(fname):

    f = nc.Dataset(fname)
    time = nc.num2date(f.variables['time'][:],
                        f.variables['time'].units)
    df = pd.DataFrame(f.variables['CanT'][:,0,0], columns=['CanT'])

    df['dates'] = time
    df = df.set_index('dates')

    return df

def resample_to_seasonal_cycle(df, OBS=False):

    UMOL_TO_MOL = 1E-6
    MOL_C_TO_GRAMS_C = 12.0
    SEC_2_DAY = 86400.

    # umol/m2/s -> g/C/d
    df['GPP'] *= UMOL_TO_MOL * MOL_C_TO_GRAMS_C * SEC_2_DAY

    # kg/m2/s -> mm/d
    df['TVeg'] *= SEC_2_DAY
    df['ESoil'] *= SEC_2_DAY

    method = {'GPP':'mean', 'CO2air':'mean', 'Qle':'mean', 'LAI':'mean',
              'TVeg':'mean', 'ESoil':'mean'}
    df = df.resample("M").agg(method).groupby(lambda x: x.month).mean()
    df['month'] = np.arange(1,13)

    return df

if __name__ == "__main__":

    amb_fname = "outputs/EucFACE_amb_out.nc"
    ele_fname = "outputs/EucFACE_ele_out.nc"
    met_fname = "met/EucFACE_met_amb.nc"
    main(amb_fname, ele_fname, met_fname)
