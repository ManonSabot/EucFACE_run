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

    Time  = nc.num2date(mets1.variables['time'][:],mets1.variables['time'].units)

    LAI = pd.DataFrame(mets1.variables['LAI'][:,0,0],columns=['LAI_R2'])
    LAI["LAI_R3"] = mets2.variables['LAI'][:,0,0]
    LAI["LAI_R6"] = mets3.variables['LAI'][:,0,0]
    LAI["LAI_ave"] = (LAI["LAI_R2"]+LAI["LAI_R3"]+LAI["LAI_R6"])/3.
    LAI['dates'] = Time
    LAI = LAI.set_index('dates')
    LAI = LAI.resample("D").agg('mean')
    LAI.index = LAI.index - pd.datetime(2011,12,31)
    LAI.index = LAI.index.days

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    fig = plt.figure(figsize=[15,15])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color'] = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor'] = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.plot(LAI.index, LAI["LAI_R2"].values)
    ax2.plot(LAI.index, LAI["LAI_R3"].values)
    ax3.plot(LAI.index, LAI["LAI_R6"].values)
    ax4.plot(LAI.index, LAI["LAI_ave"].values)

    plt.setp(ax1.get_xticklabels(), visible=True)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_title("R2")
    ax1.set_ylabel("LAI")
    ax1.axis('tight')
    ax1.set_ylim(0.,2.5)
    ax1.set_xlim(367,2739)
    ax1.legend()

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_title("R3")
    ax2.set_ylabel("LAI")
    ax2.axis('tight')
    ax2.set_ylim(0.,2.5)
    ax2.set_xlim(367,2739)
    ax2.legend()

    plt.setp(ax3.get_xticklabels(), visible=True)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_title("R6")
    ax3.set_ylabel("LAI")
    ax3.axis('tight')
    ax3.set_ylim(0.,2.5)
    ax3.set_xlim(367,2739)
    ax3.legend()

    plt.setp(ax4.get_xticklabels(), visible=True)
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_title("average")
    ax4.set_ylabel("LAI")
    ax4.axis('tight')
    ax4.set_ylim(0.,2.5)
    ax4.set_xlim(367,2739)
    ax4.legend()

    fig.savefig("LAI_amb.png", bbox_inches='tight', pad_inches=0.1)

    print(LAI.mean())
    print(LAI.max())
    print(LAI.min())

if __name__ == "__main__":

    fmets = ["/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI_6/EucFACE_met_R2.nc",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI_6/EucFACE_met_R3.nc",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI_6/EucFACE_met_R6.nc"]

    main(fmets)
