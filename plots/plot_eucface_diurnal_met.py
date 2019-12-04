#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-11-13"
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

def main(fobs, fcable1, fcable2):

# _________________________ CABLE ___________________________
    cable1 = nc.Dataset(fcable1, 'r')
    Time  = nc.num2date(cable1.variables['time'][:],cable1.variables['time'].units)
    # LWdown SWdown Tair PSurf Qair Rainf Wind
    var1 = pd.DataFrame(cable1.variables['LWdown'][:,0,0],columns=['LWdown'])
    var1['SWdown'] = cable1.variables['SWdown'][:,0,0]
    var1['Tair']   = cable1.variables['Tair'][:,0,0]
    var1['PSurf']  = cable1.variables['PSurf'][:,0,0]
    var1['Qair']   = cable1.variables['Qair'][:,0,0]
    var1['Rainf']  = cable1.variables['Rainf'][:,0,0]
    var1['Wind']   = cable1.variables['Wind'][:,0,0]
    var1['dates'] = Time
    var1 = var1.set_index('dates')
    var1 = var1.resample("H").mean()

    cable2 = nc.Dataset(fcable2, 'r')
    Time  = nc.num2date(cable2.variables['time'][:],cable2.variables['time'].units)
    # LWdown SWdown Tair PSurf Qair Rainf Wind
    var2 = pd.DataFrame(cable2.variables['LWdown'][:,0,0],columns=['LWdown'])
    var2['SWdown'] = cable2.variables['SWdown'][:,0,0]
    var2['Tair']   = cable2.variables['Tair'][:,0,0]
    var2['PSurf']  = cable2.variables['PSurf'][:,0,0]
    var2['Qair']   = cable2.variables['Qair'][:,0,0]
    var2['Rainf']  = cable2.variables['Rainf'][:,0,0]
    var2['Wind']   = cable2.variables['Wind'][:,0,0]
    var2['dates'] = Time
    var2 = var2.set_index('dates')
    var2 = var2.resample("H").mean()

    '''
    t = lambda x : x.hour
    print(var.index.month)
    a = var.index.month == 1
    b = var.index.month == 2
    c = var.index.month == 12 #and var.index.month == 1 and var.index.month == 2
    '''

    var1 = var1[var1.index.month == 1]
    var1 = var1.groupby(var1.index.hour).mean()

    var2 = var2[var2.index.month == 1]
    var2 = var2.groupby(var2.index.hour).mean()


    fig = plt.figure(figsize=[10,10])
    fig.subplots_adjust(hspace=0.35)
    fig.subplots_adjust(wspace=0.35)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black
    plt.rcParams['text.color'] = almost_black
    plt.rcParams['axes.edgecolor'] = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)

    ax1.plot(var1['LWdown'],c="orange",label="EucFACE")
    ax1.plot(var2['LWdown'],c="green",label="Cumberland")
    ax2.plot(var1['SWdown'],c="orange",label="EucFACE")
    ax2.plot(var2['SWdown'],c="green",label="Cumberland")
    ax3.plot(var1['Tair'],c="orange",label="EucFACE")
    ax3.plot(var2['Tair'],c="green",label="Cumberland")
    ax4.plot(var1['PSurf'],c="orange",label="EucFACE")
    ax4.plot(var2['PSurf'],c="green",label="Cumberland")
    ax5.plot(var1['Qair'],c="orange",label="EucFACE")
    ax5.plot(var2['Qair'],c="green",label="Cumberland")
    ax6.plot(var1['Rainf'],c="orange",label="EucFACE")
    ax6.plot(var2['Rainf'],c="green",label="Cumberland")
    ax7.plot(var1['Wind'],c="orange",label="EucFACE")
    ax7.plot(var2['Wind'],c="green",label="Cumberland")

    ax1.set_title("LWdown")
    ax2.set_title("SWdown")
    ax3.set_title("Tair")
    ax4.set_title("PSurf")
    ax5.set_title("Qair")
    ax6.set_title("Rainf")
    ax7.set_title("Wind")

    ax1.legend()
    fig.savefig("EucFACE_vs_CumberlandPlains_met.png",bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":

    fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    #fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/ctl_met_LAI_vrt_SM_swilt-watr_31uni_HDM_or-off-litter_Hvrd/EucFACE_amb_out.nc"
    fcable1 ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_6layers/met/met_only/EucFACE_met_amb.nc"
    fcable2 ="/srv/ccrc/data25/z5218916/data/Cumberland_OzFlux/CumberlandPlainsOzFlux2.0_met.nc"

    main(fobs, fcable1, fcable2)
