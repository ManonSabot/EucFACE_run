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
import glob
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


def plot_ET(fctl, flit, fbest, ring):

    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)

    TVeg_ctl   = read_cable_var(fctl, "TVeg")
    ESoil_ctl  = read_cable_var(fctl, "ESoil")

    TVeg_lit   = read_cable_var(flit, "TVeg")
    ESoil_lit  = read_cable_var(flit, "ESoil")

    TVeg_best  = read_cable_var(fbest, "TVeg")
    ESoil_best = read_cable_var(fbest, "ESoil")

    fig = plt.figure(figsize=[9,12])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 12
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

    ax1  = fig.add_subplot(311)
    ax2  = fig.add_subplot(312)
    ax3  = fig.add_subplot(313)

    x = TVeg_ctl.index

    ax1.plot(x, TVeg_ctl['cable'].rolling(window=7).mean(),     c="green", lw=1.0, ls="-", label="Trans") #.rolling(window=5).mean() .rolling(window=7).mean()
    ax1.plot(x, ESoil_ctl['cable'].rolling(window=7).mean(),    c="orange", lw=1.0, ls="-", label="ESoil") #.rolling(window=7).mean()
    ax1.scatter(subs_Trans.index, subs_Trans['obs'], marker='o', c='',edgecolors="green", s = 4., label="Trans Obs") # subs['EfloorPred'] 'blue'
    ax1.scatter(subs_Esoil.index, subs_Esoil['obs'], marker='o', c='',edgecolors="orange", s = 4., label="ESoil Obs") # subs['EfloorPred'] 'red'

    ax2.plot(x, TVeg_lit['cable'].rolling(window=7).mean(),     c="green", lw=1.0, ls="-", label="Trans") #.rolling(window=5).mean() .rolling(window=7).mean()
    ax2.plot(x, ESoil_lit['cable'].rolling(window=7).mean(),    c="orange", lw=1.0, ls="-", label="ESoil") #.rolling(window=7).mean()
    ax2.scatter(subs_Trans.index, subs_Trans['obs'], marker='o', c='',edgecolors="green", s = 4., label="Trans Obs") # subs['EfloorPred'] 'blue'
    ax2.scatter(subs_Esoil.index, subs_Esoil['obs'], marker='o', c='',edgecolors="orange", s = 4., label="ESoil Obs") # subs['EfloorPred'] 'red'

    ax3.plot(x, TVeg_best['cable'].rolling(window=7).mean(),     c="green", lw=1.0, ls="-", label="Trans") #.rolling(window=5).mean() .rolling(window=7).mean()
    ax3.plot(x, ESoil_best['cable'].rolling(window=7).mean(),    c="orange", lw=1.0, ls="-", label="ESoil") #.rolling(window=7).mean()
    ax3.scatter(subs_Trans.index, subs_Trans['obs'], marker='o', c='',edgecolors="green", s = 4., label="Trans Obs") # subs['EfloorPred'] 'blue'
    ax3.scatter(subs_Esoil.index, subs_Esoil['obs'], marker='o', c='',edgecolors="orange", s = 4., label="ESoil Obs") # subs['EfloorPred'] 'red'

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax1.get_xticklabels(), visible=True)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.set_ylabel("Trans, Esoil ($mm d^{-1}$)")
    ax1.axis('tight')
    ax1.set_ylim(0.,4.0)
    ax1.set_xlim(367,1098)
    ax1.legend()

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax2.set_ylabel("Trans, Esoil ($mm d^{-1}$)")
    ax2.axis('tight')
    ax2.set_ylim(0.,4.0)
    ax2.set_xlim(367,1098)

    plt.setp(ax3.get_xticklabels(), visible=True)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax3.set_ylabel("Trans, Esoil ($mm d^{-1}$)")
    ax3.axis('tight')
    ax3.set_ylim(0.,4.0)
    ax3.set_xlim(367,1098)

    fig.savefig("../plots/EucFACE_ET_ctl-lit-best" , bbox_inches='tight', pad_inches=0.1)


def read_cable_var(fcable, var_name):

    """
    read a var from CABLE output
    """

    print("carry on read_cable_var")
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    if var_name in ["TVeg", "ESoil", "Rainf"]:
        var = pd.DataFrame(cable.variables[var_name][:,0,0]*1800., columns=['cable'])
    else:
        var = pd.DataFrame(cable.variables[var_name][:,0,0], columns=['cable'])
    var['Date'] = Time
    var = var.set_index('Date')
    if var_name in ["TVeg", "ESoil", "Rainf"]:
        var = var.resample("D").agg('sum')
    elif var_name in ["Fwsoil"]:
        var = var.resample("D").agg('mean')
    var.index = var.index - pd.datetime(2011,12,31)
    var.index = var.index.days
    var = var.sort_values(by=['Date'])

    return var

def read_obs_esoil(ring):

    fobs_Esoil = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_underET.csv"

    est_esoil = pd.read_csv(fobs_Esoil, usecols = ['Ring','Date','wuTP'])
    est_esoil['Date'] = pd.to_datetime(est_esoil['Date'],format="%d/%m/%Y",infer_datetime_format=False)
    est_esoil['Date'] = est_esoil['Date'] - pd.datetime(2011,12,31)
    est_esoil['Date'] = est_esoil['Date'].dt.days
    est_esoil = est_esoil.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
       subs = est_esoil[(est_esoil['Ring'].isin(['R2','R3','R6'])) & (est_esoil.Date > 366)]
    elif ring == 'ele':
       subs = est_esoil[(est_esoil['Ring'].isin(['R1','R4','R5'])) & (est_esoil.Date > 366)]
    else:
       subs = est_esoil[(est_esoil['Ring'].isin([ring]))  & (est_esoil.Date > 366)]

    subs = subs.groupby(by=["Date"]).mean()
    subs['wuTP']   = subs['wuTP'].clip(lower=0.)
    subs['wuTP']   = subs['wuTP'].replace(0., float('nan'))
    subs = subs.rename({'wuTP' : 'obs'}, axis='columns')

    return subs

def read_obs_trans(ring):

    fobs_Trans = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_sapflow.csv"

    est_trans = pd.read_csv(fobs_Trans, usecols = ['Ring','Date','volRing'])
    est_trans['Date'] = pd.to_datetime(est_trans['Date'],format="%d/%m/%Y",infer_datetime_format=False)
    est_trans['Date'] = est_trans['Date'] - pd.datetime(2011,12,31)
    est_trans['Date'] = est_trans['Date'].dt.days
    est_trans = est_trans.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
       subs = est_trans[(est_trans['Ring'].isin(['R2','R3','R6'])) & (est_trans.Date > 366)]
    elif ring == 'ele':
       subs = est_trans[(est_trans['Ring'].isin(['R1','R4','R5'])) & (est_trans.Date > 366)]
    else:
       subs = est_trans[(est_trans['Ring'].isin([ring]))  & (est_trans.Date > 366)]

    subs = subs.groupby(by=["Date"]).mean()
    subs['volRing']   = subs['volRing'].clip(lower=0.)
    subs['volRing']   = subs['volRing'].replace(0., float('nan'))
    subs = subs.rename({'volRing' : 'obs'}, axis='columns')

    return subs

def read_obs_swc(ring):

    fobs   = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    tdr = pd.read_csv(fobs, usecols = ['Ring','Date','swc.tdr'])
    tdr['Date'] = pd.to_datetime(tdr['Date'],format="%Y-%m-%d",infer_datetime_format=False)
    tdr['Date'] = tdr['Date'] - pd.datetime(2011,12,31)
    tdr['Date'] = tdr['Date'].dt.days
    tdr = tdr.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
        subset = tdr[(tdr['Ring'].isin(['R2','R3','R6'])) & (tdr.Date > 366)]
    elif ring == 'ele':
        subset = tdr[(tdr['Ring'].isin(['R1','R4','R5'])) & (tdr.Date > 366)]
    else:
        subset = tdr[(tdr['Ring'].isin([ring]))  & (tdr.Date > 366)]

    subset = subset.groupby(by=["Date"]).mean()/100.
    subset = subset.rename({'swc.tdr' : 'obs'}, axis='columns')
    return subset
