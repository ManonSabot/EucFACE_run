#!/usr/bin/env python

"""
draw drought plots

Include functions :

    plot_EF_SM
    plot_Fwsoil_Trans
    plot_Rain_Fwsoil_Trans
    plot_Rain_Fwsoil_Trans_Esoil_EF_SM
    plot_Fwsoil_days

"""
__author__ = "MU Mengyuan"
__version__ = "2020-03-19"

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import datetime as dt
import netCDF4 as nc
import scipy.stats as stats
import seaborn as sns
from matplotlib import cm
from matplotlib import ticker
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
from plot_eucface_get_var import *

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

def plot_Fwsoil_Trans(fcables, ring, case_labels):

    subs_Trans = read_obs_trans(ring)

    fig = plt.figure(figsize=[10,8])

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

    ax1  = fig.add_subplot(211)
    ax2  = fig.add_subplot(212)
    #day_start = 1828
    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    colors = ["pink","orange","blue","green"]
    case_sum = len(case_labels)

    ax2.plot(subs_Trans.index, subs_Trans['obs'].rolling(window=10).mean(),  c='red', label="Obs")#,edgecolors="red", s = 4., marker='o')#.rolling(window=30).sum()
    #scatter
    for case_num in np.arange(case_sum):
        fw    = read_cable_var(fcables[case_num], "Fwsoil")
        Trans = read_cable_var(fcables[case_num], "TVeg")

        x    = fw.index

        ax1.plot(x, fw['cable'].rolling(window=10).mean(),   c=colors[case_num], lw=1.0, ls="-", label=case_labels[case_num])#.rolling(window=30).mean()
        ax2.plot(x, Trans['cable'].rolling(window=10).mean(), c=colors[case_num], lw=1.0, ls="-", label=case_labels[case_num])

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_ylabel("β")
    ax1.axis('tight')
    ax1.set_ylim(0.,1.1)
    ax1.set_xlim(367,978)
    #ax1.legend()

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("Transpiration (mm d$^{-1}$)")
    ax2.axis('tight')
    ax2.set_ylim(0.,3.)
    ax2.set_xlim(367,978)#(367,1098)
    ax2.legend()
    fig.savefig("../plots/EucFACE_Fwsoil_Trans-Martin" , bbox_inches='tight', pad_inches=0.1)

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

def plot_Fwsoil_days_bar(fcables, case_labels):
    """
    Calculate from beta figure two metrics: #1 over only drought periods and
    #2 over whole length of run. Calculate number of days where the average
    beta is below 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1. Then plot a simple
    bar chart with the results for each experiment.
    """

    fig = plt.figure(figsize=[12,8])

    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
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

    ax  = fig.add_subplot(111)

    case_sum   = len(fcables)
    intval     = np.arange(0.8,0.0,-0.1)
    intval_sum = len(intval)
    drought = True
    if drought:
        day_start  = 1828 - 367# first day of 2017
        day_end    = 2558 - 367 # first day of 2019
        tot_year   = 2 #6
    else:
        day_start  = 367 - 367 # first day of 2013
        day_end    = 2558 - 367 # first day of 2019
        tot_year   = 6

    offset     = 0.5
    fw_days    = np.zeros([case_sum,intval_sum])
    width      = 0.1
    print(fw_days)
    for case_num in np.arange(case_sum):
        print(fcables[case_num])
        fw = read_cable_var(fcables[case_num], "Fwsoil")

        for intval_num in np.arange(intval_sum):
            print(fw)

            print(fw[day_start:day_end])

            tmp = np.where(fw.values[day_start:day_end] <= intval[intval_num], 1., 0.)

            fw_days[case_num, intval_num] = sum(tmp)/tot_year
        '''
        x = np.arange( case_num + 0.15, case_num + 0.9, 0.1)
        print(x)
        print(fw_days[case_num,:])
        ax.bar( x , fw_days[case_num,:], width, edgecolor= "black", color=colors[case_num], label=case_labels[case_num])

    cleaner_dates = [ "0.8","0.6","0.4","0.2",
                      "0.8","0.6","0.4","0.2",
                      "0.8","0.6","0.4","0.2",
                      "0.8","0.6","0.4","0.2",
                      "0.8","0.6","0.4","0.2",
                      "0.8","0.6","0.4","0.2",
                      "0.8","0.6","0.4","0.2"]

    xtickslocs    = [  0.15, 0.35, 0.55, 0.75,
                       1.15, 1.35, 1.55, 1.75,
                       2.15, 2.35, 2.55, 2.75,
                       3.15, 3.35, 3.55, 3.75,
                       4.15, 4.35, 4.55, 4.75,
                       5.15, 5.35, 5.55, 5.75,
                       6.15, 6.35, 6.55, 6.75 ]
        '''

    x      = np.arange(intval_sum)
    width  = 1./(case_sum+2.)

    for case_num in np.arange(case_sum):
        offset = (case_num+1+0.5)*width
        ax.bar(x + offset, fw_days[case_num,:], width, color=colors[case_num], label=case_labels[case_num])

    cleaner_dates = [ "0.8","0.7","0.6","0.5","0.4","0.3","0.2","0.1"]

    xtickslocs    = [   0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5]


    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    if drought:
        ax.set_title("2017-2018 drought")
    else:
        ax.set_title("2013-2018")#
    ax.set_ylabel("days per year")
    ax.set_xlabel("β")
    ax.axis('tight')
    ax.set_ylim(0,300)
    ax.set_xlim(-0.2,7)
    ax.legend()

    if drought:
        fig.savefig("../plots/EucFACE_Fwsoil_days_2017-2018_drought" , bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("../plots/EucFACE_Fwsoil_days_2013-2018" , bbox_inches='tight', pad_inches=0.1)

def plot_Fwsoil_boxplot(fcables, case_labels):

    """
    box-whisker of fwsoil
    """

    day_start_drought = 1828 # first day of 2017
    day_start_all     = 367  # first day of 2013
    day_end           = 2558 # first day of 2019

    day_drought  = day_end - day_start_drought + 1
    day_all      = day_end - day_start_all + 1
    case_sum     = len(fcables)
    fw           = pd.DataFrame(np.zeros((day_drought+day_all)*case_sum),columns=['fwsoil'])
    fw['year']   = [''] * ((day_drought+day_all)*case_sum)
    fw['exp']    = [''] * ((day_drought+day_all)*case_sum)

    s = 0

    for case_num in np.arange(case_sum):

        cable = nc.Dataset(fcables[case_num], 'r')
        Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

        Fwsoil          = pd.DataFrame(cable.variables['Fwsoil'][:,0,0],columns=['fwsoil'])
        Fwsoil['dates'] = Time
        Fwsoil          = Fwsoil.set_index('dates')
        Fwsoil          = Fwsoil.resample("D").agg('mean')
        Fwsoil.index    = Fwsoil.index - pd.datetime(2011,12,31)
        Fwsoil.index    = Fwsoil.index.days

        e  = s+day_drought
        print(Fwsoil[np.all([Fwsoil.index >= day_start_drought, Fwsoil.index <=day_end],axis=0)]['fwsoil'].values)
        print(fw['year'].iloc[s:e] )
        fw['fwsoil'].iloc[s:e] = Fwsoil[np.all([Fwsoil.index >= day_start_drought, Fwsoil.index <=day_end],axis=0)]['fwsoil'].values
        fw['year'].iloc[s:e]   = ['drought'] * day_drought
        fw['exp'].iloc[s:e]    = [ case_labels[case_num]] * day_drought
        s  = e
        e  = s+day_all
        fw['fwsoil'].iloc[s:e] = Fwsoil[np.all([Fwsoil.index >= day_start_all, Fwsoil.index <=day_end],axis=0)]['fwsoil'].values
        fw['year'].iloc[s:e]   = ['all'] * day_all
        fw['exp'].iloc[s:e]    = [ case_labels[case_num]] * day_all
        s  =  e

    print(fw)

    fig = plt.figure(figsize=[12,9])
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

    ax  = fig.add_subplot(111)

    #ax.boxplot(Qle, widths = 0.4, showfliers=False)# c=colors[case_num], label=case_labels[case_num])

    # define outlier properties
    flierprops = dict(marker='o', markersize=3, markerfacecolor="black")
    ax = sns.boxplot(x="exp", y="fwsoil", hue="year", data=fw, palette="Set3",
                     order=case_labels, flierprops=flierprops, width=0.6,
                     hue_order=['drought','all'])

    ax.set_ylabel("β")
    ax.set_xlabel("simulations")
    ax.axis('tight')
    #ax1.set_xlim(date[0],date[-1])
    ax.set_ylim(0.,1.1)
    ax.axhline(y=np.median(fw[np.all([fw.year=='drought',fw.exp=='Ctl'],axis=0)]['fwsoil'].values) , ls="--")

    plt.legend()#loc="upper right"

    fig.savefig("../plots/EucFACE_Fwsoil_boxplot" , bbox_inches='tight', pad_inches=0.1)
