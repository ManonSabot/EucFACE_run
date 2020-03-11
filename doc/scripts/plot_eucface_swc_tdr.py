#!/usr/bin/env python

"""
Calculate tdr SM, fwsoil, and fluxes

Include functions :

    plot_tdr
    plot_Fwsoil
    plot_ET
    plot_Rain
    plot_Rain_Fwsoil
    plot_ET_3
    plot_EF_SM
    plot_EF_SM_HW
    plot_Rain_Fwsoil_Trans
    plot_Rain_Fwsoil_Trans_EF_SM

"""
__author__ = "MU Mengyuan"
__version__ = "2020-03-10"

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import datetime as dt
import netCDF4 as nc
from matplotlib import cm
from matplotlib import ticker
from scipy.interpolate import griddata
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from plot_eucface_get_var import *

def plot_tdr(fcable, case_name, ring, layer):

    subset = read_obs_swc_tdr(ring)

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    if layer == "6":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.5-0.022-0.058-0.154) )/0.5
    elif layer == "31uni":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.05 )/0.5

    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.sort_values(by=['dates'])

    swilt = np.zeros(len(SoilMoist))
    sfc = np.zeros(len(SoilMoist))
    ssat = np.zeros(len(SoilMoist))

    if layer == "6":
        swilt[:] = ( cable.variables['swilt'][0]*0.022 + cable.variables['swilt'][1]*0.058 \
                   + cable.variables['swilt'][2]*0.154 + cable.variables['swilt'][3]*(0.5-0.022-0.058-0.154) )/0.5
        sfc[:] = ( cable.variables['sfc'][0]*0.022   + cable.variables['sfc'][1]*0.058 \
                   + cable.variables['sfc'][2]*0.154 + cable.variables['sfc'][3]*(0.5-0.022-0.058-0.154) )/0.5
        ssat[:] = ( cable.variables['ssat'][0]*0.022 + cable.variables['ssat'][1]*0.058 \
                   + cable.variables['ssat'][2]*0.154+ cable.variables['ssat'][3]*(0.5-0.022-0.058-0.154) )/0.5
    elif layer == "31uni":
        swilt[:] =(cable.variables['swilt'][0]*0.15 + cable.variables['swilt'][1]*0.15 \
                  + cable.variables['swilt'][2]*0.15 + cable.variables['swilt'][3]*0.05 )/0.5
        sfc[:] =(cable.variables['sfc'][0]*0.15 + cable.variables['sfc'][1]*0.15 \
                + cable.variables['sfc'][2]*0.15 + cable.variables['sfc'][3]*0.05 )/0.5
        ssat[:] =(cable.variables['ssat'][0]*0.15 + cable.variables['ssat'][1]*0.15 \
                 + cable.variables['ssat'][2]*0.15 + cable.variables['ssat'][3]*0.05 )/0.5

# ____________________ Plot obs _______________________
    fig = plt.figure(figsize=[9,4])

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

    ax = fig.add_subplot(111)

    x   = SoilMoist.index

    ax.plot(subset.index, subset.values,   c="green", lw=1.0, ls="-", label="tdr")
    ax.plot(x, SoilMoist.values,c="orange", lw=1.0, ls="-", label="swc")
    '''
    tmp1 = SoilMoist['SoilMoist'].loc[SoilMoist.index.isin(subset.index)]
    tmp2 = subset.loc[subset.index.isin(SoilMoist.index)]
    mask = np.isnan(tmp2)
    print(mask)
    tmp1 = tmp1[mask == False]
    tmp2 = tmp2[mask == False]

    cor_tdr = stats.pearsonr(tmp1,tmp2)
    mse_tdr = mean_squared_error(tmp2, tmp1)
    ax.set_title("r = % 5.3f , MSE = % 5.3f" %(cor_tdr[0], np.sqrt(mse_tdr)))
    print("-----------------------------------------------")
    print(mse_tdr)
    '''
    ax.plot(x, swilt,           c="black", lw=1.0, ls="-", label="swilt")
    ax.plot(x, sfc,             c="black", lw=1.0, ls="-.", label="sfc")
    ax.plot(x, ssat,            c="black", lw=1.0, ls=":", label="ssat")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax.set_ylabel("VWC (m3/m3)")
    ax.axis('tight')
    ax.set_ylim(0,0.5)
    ax.set_xlim(367,2739)
    ax.legend()

    fig.savefig("../plots/EucFACE_tdr_%s_%s" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)

def plot_Fwsoil(fcbl_def, fcbl_fw_def, fcbl_fw_hie, ring):

    fw1 = read_cable_var(fcbl_def, "Fwsoil")
    fw2 = read_cable_var(fcbl_fw_def, "Fwsoil")
    fw3 = read_cable_var(fcbl_fw_hie, "Fwsoil")

    fig = plt.figure(figsize=[15,10])

    ax  = fig.add_subplot(111)

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

    x = fw1.index

    ax.plot(x, fw1["cable"],   c="orange", lw=1.0, ls="-", label="Default_fw-std")
    ax.plot(x, fw2["cable"],   c="blue", lw=1.0, ls="-", label="Best_fw-std")
    ax.plot(x, fw3["cable"],   c="forestgreen", lw=1.0, ls="-", label="Best_fw-hie")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax.set_ylabel("β")
    ax.axis('tight')
    ax.set_ylim(0.,1.1)
    ax.set_xlim(367,2739)
    ax.legend()

    fig.savefig("../plots/EucFACE_fwsoil_comp_%s" % ring, bbox_inches='tight', pad_inches=0.1)

def plot_ET(fcable, case_name, ring):

    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)

    TVeg  = read_cable_var(fcable, "TVeg")
    ESoil = read_cable_var(fcable, "ESoil")

    fig = plt.figure(figsize=[9,5])
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

    ax  = fig.add_subplot(111)

    x = TVeg.index

    ax.plot(x, TVeg['cable'].rolling(window=7).mean(),     c="green", lw=1.0, ls="-", label="Trans") #.rolling(window=5).mean() .rolling(window=7).mean()
    ax.plot(x, ESoil['cable'].rolling(window=7).mean(),    c="orange", lw=1.0, ls="-", label="ESoil") #.rolling(window=7).mean()
    ax.scatter(subs_Trans.index, subs_Trans['obs'], marker='o', c='',edgecolors="green", s = 4., label="Trans Obs") # subs['EfloorPred'] 'blue'
    ax.scatter(subs_Esoil.index, subs_Esoil['obs'], marker='o', c='',edgecolors="orange", s = 4., label="ESoil Obs") # subs['EfloorPred'] 'red'

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax.set_ylabel("Trans, Esoil ($mm d^{-1}$)")
    ax.axis('tight')
    ax.set_ylim(0.,3.0)
    ax.set_xlim(367,1098)
    #ax.legend()

    fig.savefig("../plots/EucFACE_ET_%s_%s" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)

def plot_Rain(fcable, case_name, ring):

    Rain  = read_cable_var(fcable, "Rainf")
    fig   = plt.figure(figsize=[15,10])

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

    ax    = fig.add_subplot(111)
    x     = Rain.index
    width = 1.

    ax.bar(x, Rain['cable'], width, color='royalblue', label='Obs')

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.set_ylabel("Rain (mm/day)")
    ax.axis('tight')
    ax.set_ylim(0.,150.)
    ax.set_xlim(367,2739)

    fig.savefig("../plots/EucFACE_Rainfall", bbox_inches='tight', pad_inches=0.1)

def plot_Rain_Fwsoil(fcbl_def, fcbl_fw_def, fcbl_fw_hie, ring):

    fw1 = read_cable_var(fcbl_def, "Fwsoil")
    fw2 = read_cable_var(fcbl_fw_def, "Fwsoil")
    fw3 = read_cable_var(fcbl_fw_hie, "Fwsoil")
    Rain= read_cable_var(fcbl_def, "Rainf")

    fig = plt.figure(figsize=[15,10])

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

    ax1  = fig.add_subplot(211)
    ax2  = fig.add_subplot(212)

    x    = Rain.index
    width= 1.

    ax1.bar(x, Rain['cable'], width, color='royalblue', label='Obs')

    ax2.plot(x, fw1['cable'],   c="orange", lw=1.0, ls="-", label="Default_fw-std")
    ax2.plot(x, fw2['cable'],   c="blue", lw=1.0, ls="-", label="Best_fw-std")
    ax2.plot(x, fw3['cable'],   c="forestgreen", lw=1.0, ls="-", label="Best_fw-hie")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")
    ax1.set_ylabel("Rain (mm/day)")
    ax1.axis('tight')
    ax1.set_ylim(0.,150.)
    ax1.set_xlim(367,2739)#,1098)

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("β")
    ax2.axis('tight')
    ax2.set_ylim(0.,1.1)
    ax2.set_xlim(367,2739)#,1098)
    ax2.legend()

    fig.savefig("../plots/EucFACE_Rain_Fwsoil_%s" % ring, bbox_inches='tight', pad_inches=0.1)

def plot_ET_3(fctl, flit, fbest, ring):

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

def plot_EF_SM(fstd, fhvrd, fexp, fwatpot, ring, layer):

    lh1 = read_cable_var(fstd, "Qle")
    lh2 = read_cable_var(fhvrd, "Qle")
    lh3 = read_cable_var(fexp, "Qle")
    lh4 = read_cable_var(fwatpot, "Qle")

    r1 = read_cable_var(fstd, "Qh") + read_cable_var(fstd, "Qle")
    r2 = read_cable_var(fhvrd, "Qh") + read_cable_var(fhvrd, "Qle")
    r3 = read_cable_var(fexp, "Qh") + read_cable_var(fexp, "Qle")
    r4 = read_cable_var(fwatpot, "Qh") + read_cable_var(fwatpot, "Qle")

    r1["cable"] = np.where(r1["cable"].values < 1., lh1['cable'].values, r1["cable"].values)
    r2["cable"] = np.where(r2["cable"].values < 1., lh2['cable'].values, r2["cable"].values)
    r3["cable"] = np.where(r3["cable"].values < 1., lh3['cable'].values, r3["cable"].values)
    r4["cable"] = np.where(r4["cable"].values < 1., lh4['cable'].values, r4["cable"].values)

    EF1 = pd.DataFrame(lh1['cable'].values/r1['cable'].values, columns=['EF'])
    EF1["Date"] = lh1.index
    EF1 = EF1.set_index('Date')
    EF1["EF"]= np.where(EF1["EF"].values> 10.0, 10., EF1["EF"].values)

    EF2 = pd.DataFrame(lh2['cable'].values/r2['cable'].values, columns=['EF'])
    EF2["Date"] = lh2.index
    EF2 = EF2.set_index('Date')
    EF2["EF"]= np.where(EF2["EF"].values> 10.0, 10., EF2["EF"].values)

    EF3 = pd.DataFrame(lh3['cable'].values/r3['cable'].values, columns=['EF'])
    EF3["Date"] = lh3.index
    EF3 = EF3.set_index('Date')
    EF3["EF"]= np.where(EF3["EF"].values> 10.0, 10., EF3["EF"].values)

    EF4 = pd.DataFrame(lh4['cable'].values/r4['cable'].values, columns=['EF'])
    EF4["Date"] = lh4.index
    EF4 = EF4.set_index('Date')
    EF4["EF"]= np.where(EF4["EF"].values> 10.0, 10., EF4["EF"].values)

    sm1 = read_SM_top_mid_bot(fstd, ring, layer)
    #print(sm1)
    sm2 = read_SM_top_mid_bot(fhvrd, ring, layer)
    sm3 = read_SM_top_mid_bot(fexp, ring, layer)
    sm4 = read_SM_top_mid_bot(fwatpot, ring, "6")

    fig = plt.figure(figsize=[15,17])

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

    ax1  = fig.add_subplot(511)
    ax2  = fig.add_subplot(512)
    ax3  = fig.add_subplot(513)
    ax4  = fig.add_subplot(514)
    ax5  = fig.add_subplot(515)

    day_start = 1828
    x    = lh1.index[lh1.index >= day_start]
    width= 1.
    print(EF1['EF'])
    ax1.plot(x, EF1['EF'][lh1.index >= day_start],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax1.plot(x, EF2['EF'][lh1.index >= day_start],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax1.plot(x, EF3['EF'][lh1.index >= day_start],   c="green", lw=1.0, ls="-", label="β-exp")
    ax1.plot(x, EF4['EF'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="Ctl-β-std")
    print("-------------------")
    print(sm1['SM_top'])
    print(lh1)
    ax2.plot(x, sm1['SM_top'][lh1.index >= day_start],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax2.plot(x, sm2['SM_top'][lh1.index >= day_start],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax2.plot(x, sm3['SM_top'][lh1.index >= day_start],   c="green", lw=1.0, ls="-", label="β-exp")
    ax2.plot(x, sm4['SM_top'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="Ctl-β-std")
    #
    # ax3.plot(x, sm1['SM_mid'][lh1.index >= day_start],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    # ax3.plot(x, sm2['SM_mid'][lh1.index >= day_start],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    # ax3.plot(x, sm3['SM_mid'][lh1.index >= day_start],   c="green", lw=1.0, ls="-", label="β-exp")
    # ax3.plot(x, sm4['SM_mid'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="Ctl-β-std")
    #
    # ax4.plot(x, sm1['SM_bot'][lh1.index >= day_start],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    # ax4.plot(x, sm2['SM_bot'][lh1.index >= day_start],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    # ax4.plot(x, sm3['SM_bot'][lh1.index >= day_start],   c="green", lw=1.0, ls="-", label="β-exp")
    # ax4.plot(x, sm4['SM_bot'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="Ctl-β-std")

    ax5.plot(x, sm1['SM_all'][lh1.index >= day_start],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax5.plot(x, sm2['SM_all'][lh1.index >= day_start],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax5.plot(x, sm3['SM_all'][lh1.index >= day_start],   c="green", lw=1.0, ls="-", label="β-exp")
    ax5.plot(x, sm4['SM_all'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="Ctl-β-std")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")
    ax1.set_ylabel("Evaporative Fraction (-)")
    ax1.axis('tight')
    #ax1.set_ylim(0.,120.)
    #ax1.set_xlim(367,2739)#,1098)
    ax1.set_xlim(day_start,2739)

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("Top soil moisture  (m$3$ m$-3$)")
    ax2.axis('tight')
    ax2.set_ylim(0.,0.4)
    #ax2.set_xlim(367,2739)#,1098)
    ax2.set_xlim(day_start,2739)

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Middle soil moisture  (m$3$ m$-3$)")
    ax3.axis('tight')
    ax3.set_ylim(0.,0.4)
    #ax3.set_xlim(367,2739)#,1098)
    ax3.set_xlim(day_start,2739)

    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("Bottom soil moisture  (m$3$ m$-3$)")
    ax4.axis('tight')
    ax4.set_ylim(0.,0.4)
    #ax4.set_xlim(367,2739)#,1098)
    ax4.set_xlim(day_start,2739)

    plt.setp(ax5.get_xticklabels(), visible=True)
    ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax5.set_ylabel("soil moisture  (m$3$ m$-3$)")
    ax5.axis('tight')
    ax5.set_ylim(0.,0.4)
    #ax5.set_xlim(367,2739)#,1098)
    ax5.set_xlim(day_start,2739)
    ax5.legend()

    fig.savefig("../plots/EucFACE_EF_SM" , bbox_inches='tight', pad_inches=0.1)

def find_Heatwave(fcable, ring, layer):

    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    # Air temperature
    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')
    Tair = Tair.resample("D").agg('max')
    #Tair.index = Tair.index - pd.datetime(2011,12,31)
    #Tair.index = Tair.index.days

    # Precipitation
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    #Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    #Rainf.index = Rainf.index.days

    Qle = read_cable_var(fcable, "Qle")
    Qh  = read_cable_var(fcable, "Qh")
    Rnet= read_cable_var(fcable, "Qle") + read_cable_var(fcable, "Qh")

    #Rnet["cable"] = np.where(Rnet["cable"].values < 1., Qle['cable'].values, Rnet["cable"].values)
    EF = pd.DataFrame(Qle['cable'].values/Rnet['cable'].values, columns=['EF'])
    #EF['EF'] = np.where(EF["EF"].values >10.0, 10., EF["EF"].values)
    SM = read_SM_top_mid_bot(fcable, ring, layer)

    # exclude rainday and the after two days of rain
    day = np.zeros((len(Tair)), dtype=bool)

    for i in np.arange(0,len(Tair)):
        if (Tair.values[i] >= 35.): # and Rainf.values[i] == 0.):
            day[i]   = True

    # calculate heatwave event
    HW = [] # create empty list

    i = 0
    while i < len(Tair)-2:
        HW_event = []
        if (np.all([day[i:i+3]])):
            # consistent 3 days > 35 degree
            for j in np.arange(i-2,i+3):

                event = ( Tair.index[j], Tair['Tair'].values[j], Rainf['Rainf'].values[j],
                          Qle['cable'].values[j], Qh['cable'].values[j],
                          EF['EF'].values[j], SM['SM_top'].values[j], SM['SM_mid'].values[j],
                          SM['SM_bot'].values[j], SM['SM_all'].values[j], SM['SM_15m'].values[j])
                HW_event.append(event)
            i = i + 3

            while day[i]:
                # consistent more days > 35 degree
                event = ( Tair.index[i], Tair['Tair'].values[i], Rainf['Rainf'].values[i],
                          Qle['cable'].values[i], Qh['cable'].values[i],
                          EF['EF'].values[i], SM['SM_top'].values[i], SM['SM_mid'].values[i],
                          SM['SM_bot'].values[i], SM['SM_all'].values[i], SM['SM_15m'].values[j] )
                HW_event.append(event)
                i += 1

            # post 2 days
            event = ( Tair.index[i], Tair['Tair'].values[i], Rainf['Rainf'].values[i],
                      Qle['cable'].values[i], Qh['cable'].values[i],
                      EF['EF'].values[i], SM['SM_top'].values[i], SM['SM_mid'].values[i],
                      SM['SM_bot'].values[i], SM['SM_all'].values[i], SM['SM_15m'].values[j] )
            HW_event.append(event)

            event = ( Tair.index[i+1], Tair['Tair'].values[i+1], Rainf['Rainf'].values[i+1],
                      Qle['cable'].values[i+1], Qh['cable'].values[i+1],
                      EF['EF'].values[i+1], SM['SM_top'].values[i+1], SM['SM_mid'].values[i+1],
                      SM['SM_bot'].values[i+1], SM['SM_all'].values[i+1], SM['SM_15m'].values[j] )
            HW_event.append(event)

            HW.append(HW_event)
        else:
            i += 1

    # The variable HW is a nested list, in Python accessing a nested list cannot\
    # be done by multi-dimensional slicing, i.e.: HW[1,2], instead one  would   \
    # write HW[1][2].
    # HW[:][0] does not work because HW[:] returns HW.

    return HW

def find_Heatwave_hourly(fcable, ring, layer):

    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    # Air temperature
    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')

    Tair_daily = Tair.resample("D").agg('max')

    # Precipitation
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0]*1800.,columns=['Rainf'])
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')

    Qle          = pd.DataFrame(cable.variables['Qle'][:,0,0],columns=['cable'])
    Qle['dates'] = Time
    Qle          = Qle.set_index('dates')

    Qh           = pd.DataFrame(cable.variables['Qh'][:,0,0],columns=['cable'])
    Qh['dates']  = Time
    Qh           = Qh.set_index('dates')

    Rnet         = Qle + Qh

    #print(Rnet)

    #Rnet["cable"] = np.where(Rnet["cable"].values < 1., Qle['cable'].values, Rnet["cable"].values)
    EF          = pd.DataFrame(Qle['cable'].values/Rnet['cable'].values, columns=['EF'])
    EF['dates'] = Time
    EF          = EF.set_index('dates')
    #EF['EF'] = np.where(EF["EF"].values >10.0, 10., EF["EF"].values)
    SM = read_SM_top_mid_bot_hourly(fcable, ring, layer)

    #print(SM)

    # exclude rainday and the after two days of rain
    day = np.zeros((len(Tair_daily)), dtype=bool)

    for i in np.arange(0,len(Tair_daily)):
        if (Tair_daily.values[i] >= 35.): # and Rainf.values[i] == 0.):
            day[i]   = True

    # calculate heatwave event
    HW = [] # create empty list

    i = 0

    while i < len(Tair_daily)-2:

        HW_event = []

        if (np.all([day[i:i+3]])):

            day_start = Tair_daily.index[i-2]
            i = i + 3

            while day[i]:

                i += 1

            else:
                day_end = Tair_daily.index[i+2] # the third day after heatwave

                #print(np.all([Tair.index >= day_start,  Tair.index < day_end],axis=0))
                #print(day_start)
                #print(day_end)
                Tair_event  = Tair[np.all([Tair.index >= day_start,  Tair.index < day_end],axis=0)]
                Rainf_event = Rainf[np.all([Tair.index >= day_start, Tair.index < day_end],axis=0)]
                Qle_event   = Qle[np.all([Tair.index >= day_start,   Tair.index < day_end],axis=0)]
                Qh_event    = Qh[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]
                EF_event    = EF[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]
                SM_event    = SM[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]

                for hour_num in np.arange(len(Tair_event)):
                    hour_in_event = ( Tair_event.index[hour_num],
                                      Tair_event['Tair'].values[hour_num],
                                      Rainf_event['Rainf'].values[hour_num],
                                      Qle_event['cable'].values[hour_num],
                                      Qh_event['cable'].values[hour_num],
                                      EF_event['EF'].values[hour_num],
                                      SM_event['SM_top'].values[hour_num],
                                      SM_event['SM_mid'].values[hour_num],
                                      SM_event['SM_bot'].values[hour_num],
                                      SM_event['SM_all'].values[hour_num],
                                      SM_event['SM_15m'].values[hour_num] )
                    HW_event.append(hour_in_event)

            HW.append(HW_event)
        else:
            i += 1
    #print(HW[0])

    return HW

def plot_single_HW_event(time_scale, case_labels, i, date, Tair, Rainf, Qle, Qh, EF, SM_top, SM_mid, SM_bot, SM_all, SM_15m):

    if time_scale == "daily":
        fig = plt.figure(figsize=[12,20])
    elif time_scale == "hourly":
        fig = plt.figure(figsize=[14,20])

    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
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

    ax1  = fig.add_subplot(411)
    ax2  = fig.add_subplot(412)
    ax3  = fig.add_subplot(413)
    ax4  = fig.add_subplot(414)
    #ax6  = fig.add_subplot(515)

    x      = date
    colors = cm.rainbow(np.linspace(0,1,len(case_labels)))
    ax5 = ax1.twinx()

    if time_scale == "daily":
        width  = 0.6
    elif time_scale == "hourly":
        width  = 1/48


    ax1.plot(x, Tair,   c="black", lw=1.5, ls="-", label="Air Temperature")#.rolling(window=30).mean()
    if time_scale == "daily":
        ax1.set_ylabel('Max Air Temperature (°C)')
        ax1.set_ylim(20, 45)
    elif time_scale == "hourly":
        ax1.set_ylabel('Air Temperature (°C)')
        ax1.set_ylim(10, 45)

    ax5.set_ylabel('Rainfall (mm d$^{-1}$)')
    ax5.bar(x, Rainf,  width, color='royalblue', alpha = 0.5, label='Rainfall')
    if time_scale == "daily":
        ax5.set_ylim(0., 30.)
    elif time_scale == "hourly":
        ax5.set_ylim(0., 20.)

    for case_num in np.arange(len(case_labels)):
        print(case_num)
        ax2.plot(x, EF[case_num, :],  c=colors[case_num], lw=1.5, ls="-", label=case_labels[case_num])#.rolling(window=30).mean()
        ax3.plot(x, Qle[case_num, :], c=colors[case_num], lw=1.5, ls="-", label=case_labels[case_num])#.rolling(window=30).mean()
        ax3.plot(x, Qh[case_num, :],  c=colors[case_num], lw=1.5, ls="-.") #, label=case_labels)#.rolling(window=30).mean()
        ax4.plot(x, SM_15m[case_num, :],  c=colors[case_num], lw=1.5, ls="-", label=case_labels[case_num])#.rolling(window=30).mean()
        #ax4.plot(x, SM_all[case_num, :],  c=colors[case_num], lw=1.5, ls="-", label=case_labels[case_num])#.rolling(window=30).mean()

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim(date[0],date[-1])

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel("Evaporative Fraction (-)")
    ax2.axis('tight')
    ax2.set_xlim(date[0],date[-1])
    if time_scale == "daily":
        ax2.set_ylim(0.,1.8)
    elif time_scale == "hourly":
        ax2.set_ylim(0,10.)

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel('Latent, Sensible Heat (W m$^{-2}$)')
    ax3.axis('tight')
    ax3.set_xlim(date[0],date[-1])
    if time_scale == "daily":
        ax3.set_ylim(-50.,220)

    plt.setp(ax4.get_xticklabels(), visible=True)
    ax4.set_ylabel("VWC in top 1.5m  (m$^{3}$ m$^{-3}$)")
    ax4.axis('tight')
    ax4.legend()
    ax4.set_xlim(date[0],date[-1])
    if time_scale == "daily":
        ax4.set_ylim(0.18,0.32)
        plt.suptitle('Heatwave in %s ~ %s ' % (str(date[2]), str(date[-3])))
    elif time_scale == "hourly":
        ax4.set_ylim(0.,0.4)
        #plt.suptitle('Heatwave in %s ~ %s ' % (str(date[2]), str(date[-3])))
    '''
    plt.setp(ax4.get_xticklabels(), visible=True)
    ax4.set_ylabel("VWC (m$^{3}$ m$^{-3}$)")
    ax4.axis('tight')
    #ax4.set_ylim(0.,0.4)
    ax4.legend()
    '''
    fig.savefig("../plots/EucFACE_Heatwave_%s" % str(i) , bbox_inches='tight', pad_inches=0.02)

def plot_EF_SM_HW(fcables, ring, layers, case_labels, time_scale):

    # =========== Calc HW events ==========
    # save all cases and all heatwave events
    # struction : 1st-D  2st-D  3st-D  4st-D
    #             case   event  day    variables

    HW_all   = []
    case_sum = len(fcables)

    for case_num in np.arange(case_sum):
        if time_scale == "daily":
            HW = find_Heatwave(fcables[case_num], ring, layers[case_num])
        elif time_scale == "hourly":
            HW = find_Heatwave_hourly(fcables[case_num], ring, layers[case_num])
        HW_all.append(HW)
    #print(HW_all)
    #print(HW_all[0][1])

    # ============ Read vars ==============
    event_sum = len(HW_all[0])

    for event_num in np.arange(event_sum):

        day_sum = len(HW_all[0][event_num])
        if time_scale == "daily":
            date   = np.zeros(day_sum, dtype='datetime64[D]')
        elif time_scale == "hourly":
            date   = np.zeros(day_sum, dtype='datetime64[ns]')
        Tair   = np.zeros(day_sum)
        Rainf  = np.zeros(day_sum)
        Qle    = np.zeros([case_sum,day_sum])
        Qh     = np.zeros([case_sum,day_sum])
        EF     = np.zeros([case_sum,day_sum])
        SM_top = np.zeros([case_sum,day_sum])
        SM_mid = np.zeros([case_sum,day_sum])
        SM_bot = np.zeros([case_sum,day_sum])
        SM_all = np.zeros([case_sum,day_sum])
        SM_15m = np.zeros([case_sum,day_sum])

        # loop days in one event
        for day_num in np.arange(day_sum):
            date[day_num]      = HW_all[0][event_num][day_num][0].to_datetime64()
            #print(date[day_num])
            Tair[day_num]      = HW_all[0][event_num][day_num][1]
            Rainf[day_num]     = HW_all[0][event_num][day_num][2]
            #print(date)
            for case_num in np.arange(case_sum):

                Qle[case_num,day_num]     =  HW_all[case_num][event_num][day_num][3]
                Qh[case_num,day_num]      =  HW_all[case_num][event_num][day_num][4]
                EF[case_num,day_num]      =  HW_all[case_num][event_num][day_num][5]
                SM_top[case_num,day_num]  =  HW_all[case_num][event_num][day_num][6]
                SM_mid[case_num,day_num]  =  HW_all[case_num][event_num][day_num][7]
                SM_bot[case_num,day_num]  =  HW_all[case_num][event_num][day_num][8]
                SM_all[case_num,day_num]  =  HW_all[case_num][event_num][day_num][9]
                SM_15m[case_num,day_num]  =  HW_all[case_num][event_num][day_num][10]

        plot_single_HW_event(time_scale, case_labels, event_num, date, Tair, Rainf, Qle, Qh, EF, SM_top, SM_mid, SM_bot, SM_all, SM_15m)

def plot_Rain_Fwsoil_Trans(fstd, fhvrd, fexp, fwatpot, ring):

    Rain= read_cable_var(fstd, "Rainf")

    fw1 = read_cable_var(fstd, "Fwsoil")
    fw2 = read_cable_var(fhvrd, "Fwsoil")
    fw3 = read_cable_var(fexp, "Fwsoil")
    fw4 = read_cable_var(fwatpot, "Fwsoil")

    t1 = read_cable_var(fstd, "TVeg")
    t2 = read_cable_var(fhvrd, "TVeg")
    t3 = read_cable_var(fexp, "TVeg")
    t4 = read_cable_var(fwatpot, "TVeg")

    fig = plt.figure(figsize=[15,10])

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

    day_start = 1828
    x    = Rain.index[Rain.index >= day_start]
    width= 1.

    ax1.plot(x, Rain['cable'][Rain.index >= day_start].rolling(window=30).sum(), width, color='royalblue', label='Obs') # bar   .cumsum()

    ax2.plot(x, fw1['cable'][fw1.index >= day_start].rolling(window=30).mean(),   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax2.plot(x, fw2['cable'][fw2.index >= day_start].rolling(window=30).mean(),   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax2.plot(x, fw3['cable'][fw3.index >= day_start].rolling(window=30).mean(),   c="green", lw=1.0, ls="-", label="β-exp")
    ax2.plot(x, fw4['cable'][fw4.index >= day_start].rolling(window=30).mean(),   c="red", lw=1.0, ls="-", label="β-watpot")

    ax3.plot(x, t1['cable'][t1.index >= day_start].rolling(window=30).sum(),   c="orange", lw=1.0, ls="-", label="β-std")
    ax3.plot(x, t2['cable'][t2.index >= day_start].rolling(window=30).sum(),   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax3.plot(x, t3['cable'][t3.index >= day_start].rolling(window=30).sum(),   c="green", lw=1.0, ls="-", label="β-exp")
    ax3.plot(x, t4['cable'][t4.index >= day_start].rolling(window=30).sum(),   c="red", lw=1.0, ls="-", label="β-watpot")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")
    ax1.set_ylabel("Rainfall (mm mon$^{-1}$)")
    ax1.axis('tight')
    #ax1.set_ylim(0.,120.)
    #ax1.set_xlim(367,2739)#,1098)
    ax1.set_xlim(day_start,2739)

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("β")
    ax2.axis('tight')
    ax2.set_ylim(0.,1.1)
    #ax2.set_xlim(367,2739)#,1098)
    ax2.set_xlim(day_start,2739)
    ax2.legend()

    plt.setp(ax3.get_xticklabels(), visible=True)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Transpiration ($mm$ $mon^{-1}$)")
    ax3.axis('tight')
    #ax3.set_ylim(0.,2.5)
    #ax3.set_ylim(0.,1000.)
    #ax3.set_xlim(367,2739)#,1098)
    ax3.set_xlim(day_start,2739)
    ax3.legend()
    fig.savefig("../plots/EucFACE_Rain_Fwsoil_Trans" , bbox_inches='tight', pad_inches=0.1)

def plot_Rain_Fwsoil_Trans_Esoil_EF_SM(fcables, ring, layers, case_labels):

    fig = plt.figure(figsize=[15,20])

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

    colors = cm.tab20(np.linspace(0,1,len(case_labels)))

    ax1  = fig.add_subplot(511)
    ax2  = fig.add_subplot(512)
    ax3  = fig.add_subplot(513)
    ax4  = fig.add_subplot(514)
    ax5  = fig.add_subplot(515)

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    day_start = 1828

    case_sum = len(fcables)

    for case_num in np.arange(case_sum):

        Rain  = read_cable_var(fcables[case_num], "Rainf")
        fw    = read_cable_var(fcables[case_num], "Fwsoil")
        Trans = read_cable_var(fcables[case_num], "TVeg")
        Esoil = read_cable_var(fcables[case_num], "ESoil")
        Qle   = read_cable_var(fcables[case_num], "Qle")
        Rnet  = read_cable_var(fcables[case_num], "Qh") + \
                read_cable_var(fcables[case_num], "Qle")

        Rnet = np.where(Rnet["cable"].values < 5., Qle['cable'].values, Rnet["cable"].values)

        EF   = pd.DataFrame(Qle['cable'].values/Rnet, columns=['EF'])
        EF["Date"] = Qle.index
        EF   = EF.set_index('Date')
        #mean_val = np.where(np.any([EF1["EF"].values> 1.0, EF1["EF"].values< 0.0], axis=0), float('nan'), EF1["EF"].values)
        #EF["EF"]= np.where(EF["EF"].values> 10.0, 10., EF["EF"].values)

        sm = read_SM_top_mid_bot(fcables[case_num], ring, layers[case_num])

        x    = fw.index[fw.index >= day_start]

        ax1.set_ylabel('VWC in top 1.5m (m$^{3}$ m$^{-3}$)')
        ax1.plot(x, sm['SM_15m'][Qle.index >= day_start].rolling(window=30).mean(),   c=colors[case_num], lw=1.0, ls="-", label=case_labels[case_num])#.rolling(window=30).mean()
        ax2.plot(x, Esoil['cable'][Esoil.index >= day_start].rolling(window=30).sum(),c=colors[case_num], lw=1.0, ls="-", label=case_labels[case_num])
        ax3.plot(x, Trans['cable'][Trans.index >= day_start].rolling(window=30).sum(),c=colors[case_num], lw=1.0, ls="-", label=case_labels[case_num])
        ax4.plot(x, fw['cable'][fw.index >= day_start].rolling(window=30).mean(),     c=colors[case_num], lw=1.0, ls="-", label=case_labels[case_num])#.rolling(window=30).mean()
        ax5.plot(x, EF['EF'][Qle.index >= day_start].rolling(window=30).mean(),       c=colors[case_num], lw=1.0, ls="-", label=case_labels[case_num])#.rolling(window=30).mean()


    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.axis('tight')
    ax1.set_xlim(day_start,2739)

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("Soil Evaporation (mm mon$^{-1}$)")
    ax2.axis('tight')
    ax4.set_ylim(0.,65.)
    ax2.set_xlim(day_start,2739)

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Transpiration (mm mon$^{-1}$)")
    ax3.axis('tight')
    ax4.set_ylim(0.,65.)
    ax3.set_xlim(day_start,2739)

    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("β")
    ax4.axis('tight')
    ax4.set_ylim(0.,1.1)
    #ax4.set_xlim(367,2739)#,1098)
    ax4.set_xlim(day_start,2739)
    ax4.legend()

    plt.setp(ax5.get_xticklabels(), visible=True)
    ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    #ax5.yaxis.tick_left()
    #ax5.yaxis.set_label_position("left")
    ax5.set_ylabel("Evaporative Fraction (-)")
    ax5.axis('tight')
    ax5.set_xlim(day_start,2739)

    fig.savefig("../plots/EucFACE_Rain_Fwsoil_Trans_EF_SM" , bbox_inches='tight', pad_inches=0.1)
