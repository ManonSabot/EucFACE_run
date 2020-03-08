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
from plot_eucface_get_var import *

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

def plot_EF_SM(fstd, fhvrd, fexp, fwatpot, ring, layer):

    lh1 = read_cable_var(fstd, "Qle")
    lh2 = read_cable_var(fhvrd, "Qle")
    lh3 = read_cable_var(fexp, "Qle")
    lh4 = read_cable_var(fwatpot, "Qle")

    r1 = read_cable_var(fstd, "Rnet")
    r2 = read_cable_var(fhvrd, "Rnet")
    r3 = read_cable_var(fexp, "Rnet")
    r4 = read_cable_var(fwatpot, "Rnet")

    EF1 = pd.DataFrame(lh1['cable'].values/r1['cable'].values, columns=['EF'])
    EF1["Date"] = lh1.index
    EF1 = EF1.set_index('Date')
    EF1["EF"]= np.where(np.any([EF1["EF"].values> 1.0, EF1["EF"].values< 0.0], axis=0), float('nan'), EF1["EF"].values)

    EF2 = pd.DataFrame(lh2['cable'].values/r2['cable'].values, columns=['EF'])
    EF2["Date"] = lh2.index
    EF2 = EF2.set_index('Date')
    EF2["EF"] = np.where(np.any([EF2["EF"].values> 1.0, EF2["EF"].values< 0.0], axis=0), float('nan'), EF2["EF"].values)

    EF3 = pd.DataFrame(lh3['cable'].values/r3['cable'].values, columns=['EF'])
    EF3["Date"] = lh3.index
    EF3 = EF3.set_index('Date')
    EF3["EF"] = np.where(np.any([EF3["EF"].values> 1.0, EF3["EF"].values< 0.0], axis=0), float('nan'), EF3["EF"].values)

    EF4 = pd.DataFrame(lh4['cable'].values/r4['cable'].values, columns=['EF'])
    EF4["Date"] = lh4.index
    EF4 = EF4.set_index('Date')
    EF4["EF"] = np.where(np.any([EF4["EF"].values> 1.0, EF4["EF"].values< 0.0], axis=0), float('nan'), EF4["EF"].values)

    sm1 = read_SM_top_mid_bot(fstd, ring, layer)
    #print(sm1)
    sm2 = read_SM_top_mid_bot(fhvrd, ring, layer)
    sm3 = read_SM_top_mid_bot(fexp, ring, layer)
    sm4 = read_SM_top_mid_bot(fwatpot, ring, layer)

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
    ax1.plot(x, EF4['EF'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="β-watpot")
    print("-------------------")
    print(sm1['SM_top'])
    print(lh1)
    ax2.plot(x, sm1['SM_top'][lh1.index >= day_start],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax2.plot(x, sm2['SM_top'][lh1.index >= day_start],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax2.plot(x, sm3['SM_top'][lh1.index >= day_start],   c="green", lw=1.0, ls="-", label="β-exp")
    ax2.plot(x, sm4['SM_top'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="β-watpot")

    ax3.plot(x, sm1['SM_mid'][lh1.index >= day_start],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax3.plot(x, sm2['SM_mid'][lh1.index >= day_start],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax3.plot(x, sm3['SM_mid'][lh1.index >= day_start],   c="green", lw=1.0, ls="-", label="β-exp")
    ax3.plot(x, sm4['SM_mid'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="β-watpot")

    ax4.plot(x, sm1['SM_bot'][lh1.index >= day_start],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax4.plot(x, sm2['SM_bot'][lh1.index >= day_start],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax4.plot(x, sm3['SM_bot'][lh1.index >= day_start],   c="green", lw=1.0, ls="-", label="β-exp")
    ax4.plot(x, sm4['SM_bot'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="β-watpot")

    ax5.plot(x, sm1['SM_all'][lh1.index >= day_start],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax5.plot(x, sm2['SM_all'][lh1.index >= day_start],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax5.plot(x, sm3['SM_all'][lh1.index >= day_start],   c="green", lw=1.0, ls="-", label="β-exp")
    ax5.plot(x, sm4['SM_all'][lh1.index >= day_start],   c="red", lw=1.0, ls="-", label="β-watpot")

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

def plot_EF_SM_HW(fstd, fhvrd, fexp, fwatpot, ring, layer):

    HW1 = find_Heatwave(fstd, ring, layer)
    HW2 = find_Heatwave(fhvrd, ring, layer)
    HW3 = find_Heatwave(fexp, ring, layer)
    HW4 = find_Heatwave(fwatpot, ring, layer)

    # loop event
    for i in np.arange(len(HW1)):
        date   = np.zeros(len(HW1[i]), dtype='datetime64[D]')
        Tair   = np.zeros(len(HW1[i]))
        Rainf  = np.zeros(len(HW1[i]))
        EF     = np.zeros([4,len(HW1[i])])
        SM_top = np.zeros([4,len(HW1[i])])
        SM_mid = np.zeros([4,len(HW1[i])])
        SM_bot = np.zeros([4,len(HW1[i])])
        SM_all = np.zeros([4,len(HW1[i])])

        # loop days in one event
        for j in np.arange(len(HW1[i])):
            print(HW1[i][j][0].to_datetime64())
            date[j]    = HW1[i][j][0].to_datetime64()
            Tair[j]    = HW1[i][j][1]
            Rainf[j]   = HW1[i][j][2]

            EF[0,j]      = HW1[i][j][3]
            SM_top[0,j]  = HW1[i][j][4]
            SM_mid[0,j]  = HW1[i][j][5]
            SM_bot[0,j]  = HW1[i][j][6]
            SM_all[0,j]  = HW1[i][j][7]

            EF[1,j]      = HW2[i][j][3]
            SM_top[1,j]  = HW2[i][j][4]
            SM_mid[1,j]  = HW2[i][j][5]
            SM_bot[1,j]  = HW2[i][j][6]
            SM_all[1,j]  = HW2[i][j][7]

            EF[2,j]      = HW3[i][j][3]
            SM_top[2,j]  = HW3[i][j][4]
            SM_mid[2,j]  = HW3[i][j][5]
            SM_bot[2,j]  = HW3[i][j][6]
            SM_all[2,j]  = HW3[i][j][7]

            EF[3,j]      = HW4[i][j][3]
            SM_top[3,j]  = HW4[i][j][4]
            SM_mid[3,j]  = HW4[i][j][5]
            SM_bot[3,j]  = HW4[i][j][6]
            SM_all[3,j]  = HW4[i][j][7]

            plot_HW_event(i, date, Tair, Rainf, EF, SM_top, SM_mid, SM_bot, SM_all)

def plot_HW_event(i, date, Tair, Rainf, EF, SM_top, SM_mid, SM_bot, SM_all):

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
    ax3  = fig.add_subplot(512)
    ax4  = fig.add_subplot(513)
    ax5  = fig.add_subplot(514)
    ax6  = fig.add_subplot(515)

    x     = date
    width = 1.
    ax1.set_ylabel('Rainfall (mm d$^-1$)')
    ax1.bar(x , Rainf,  width, color='blue', label='Rainfall')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Max Air Temperature (°C)')
    ax2.plot(x, Tair,   c="red", lw=1.0, ls="-", label="Max Air Temperature")#.rolling(window=30).mean()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    '''
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")
    ax1.set_ylabel("Evaporative Fraction (-)")
    # ax1.yaxis.set_label_position("left")
    # ax1.set_ylabel("Evaporative Fraction (-)")
    ax1.axis('tight')
    #ax1.set_ylim(0.,120.)
    #ax1.set_xlim(367,2739)#,1098)
    ax1.set_xlim(day_start,2739)
    ax1.legend()
    '''

    ax3.plot(x, EF[0,:],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax3.plot(x, EF[1,:],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax3.plot(x, EF[2,:],   c="green", lw=1.0, ls="-", label="β-exp")
    ax3.plot(x, EF[3,:],   c="red", lw=1.0, ls="-", label="β-watpot")

    ax4.plot(x, SM_top[0,:],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax4.plot(x, SM_top[1,:],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax4.plot(x, SM_top[2,:],   c="green", lw=1.0, ls="-", label="β-exp")
    ax4.plot(x, SM_top[3,:],   c="red", lw=1.0, ls="-", label="β-watpot")

    ax5.plot(x, SM_mid[0,:],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax5.plot(x, SM_mid[1,:],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax5.plot(x, SM_mid[2,:],   c="green", lw=1.0, ls="-", label="β-exp")
    ax5.plot(x, SM_mid[3,:],   c="red", lw=1.0, ls="-", label="β-watpot")

    ax6.plot(x, SM_all[0,:],   c="orange", lw=1.0, ls="-", label="β-std")#.rolling(window=30).mean()
    ax6.plot(x, SM_all[1,:],   c="blue", lw=1.0, ls="-", label="β-hvrd")
    ax6.plot(x, SM_all[2,:],   c="green", lw=1.0, ls="-", label="β-exp")
    ax6.plot(x, SM_all[3,:],   c="red", lw=1.0, ls="-", label="β-watpot")

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel("Evaporative Fraction (-)")
    ax3.axis('tight')
    ax3.set_ylim(0.,1.1)

    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set_ylabel("Middle soil moisture  (m$3$ m$-3$)")
    ax4.axis('tight')
    ax4.set_ylim(0.,0.4)

    plt.setp(ax5.get_xticklabels(), visible=False)
    ax5.set_ylabel("Bottom soil moisture  (m$3$ m$-3$)")
    ax5.axis('tight')
    ax5.set_ylim(0.,0.4)

    plt.setp(ax6.get_xticklabels(), visible=True)
    ax6.set_ylabel("soil moisture  (m$3$ m$-3$)")
    ax6.axis('tight')
    ax6.set_ylim(0.,0.4)
    ax6.legend()

    fig.savefig("../plots/EucFACE_HW_event_%s" % str(i) , bbox_inches='tight', pad_inches=0.1)


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
    Rnet= read_cable_var(fcable, "Rnet")

    #EF["Date"] = lh4.index
    #EF4 = EF4.set_index('Date')
    EF = pd.DataFrame(Qle['cable'].values/Rnet['cable'].values, columns=['EF'])
    EF = np.where(np.any([EF["EF"].values> 1.0, EF["EF"].values< 0.0], axis=0), float('nan'), EF["EF"].values)
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

                print(SM['SM_top'].values[j])
                print(SM['SM_mid'].values[j])
                print(SM['SM_bot'].values[j])
                print(SM['SM_all'].values[j])
                event = ( Tair.index[j], Tair['Tair'].values[j], Rainf['Rainf'].values[j],   \
                          EF[j], SM['SM_top'].values[j], SM['SM_mid'].values[j],\
                          SM['SM_bot'].values[j], SM['SM_all'].values[j] )
                HW_event.append(event)
            i = i + 3

            while day[i]:
                # consistent more days > 35 degree
                event = ( Tair.index[i], Tair['Tair'].values[i], Rainf['Rainf'].values[i], \
                          EF[i], SM['SM_top'].values[i], SM['SM_mid'].values[i],\
                          SM['SM_bot'].values[i], SM['SM_all'].values[i] )
                HW_event.append(event)
                i += 1

            # post 2 days
            event = ( Tair.index[i], Tair['Tair'].values[i], Rainf['Rainf'].values[i], \
                      EF[i], SM['SM_top'].values[i], SM['SM_mid'].values[i],\
                      SM['SM_bot'].values[i], SM['SM_all'].values[i] )
            HW_event.append(event)

            event = ( Tair.index[i+1], Tair['Tair'].values[i+1], Rainf['Rainf'].values[i+1], \
                      EF[i+1], SM['SM_top'].values[i+1], SM['SM_mid'].values[i+1],\
                      SM['SM_bot'].values[i+1], SM['SM_all'].values[i+1] )
            HW_event.append(event)

            HW.append(HW_event)
        else:
            i += 1
    print(type(HW[0][0][0]))
    #for i in np.arange(len(HW)):
    #    print(len(HW[i]))
    # The variable HW is a nested list, in Python accessing a nested list cannot\
    # be done by multi-dimensional slicing, i.e.: HW[1,2], instead one  would   \
    # write HW[1][2].
    # HW[:][0] does not work because HW[:] returns HW.

    return HW

if __name__ == "__main__":

    case_name = ["met_LAI_6",
                 "met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter",
                 "met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_hie-exp"]

    rings = ["amb"] #["R1","R2","R3","R4","R5","R6","amb","ele"]
    term  = "Qh" #"Qle" # "Qle" #
    for ring in rings:
        fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[0], ring)
        v1,lct1 = main(fobs, fcable, case_name[0], ring, term)

        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[1], ring)
        v2,lct2 = main(fobs, fcable, case_name[1], ring, term)

        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[2], ring)
        v3,lct3 = main(fobs, fcable, case_name[2], ring, term)

        fig = plt.figure(figsize=[10,7])
        fig.subplots_adjust(hspace=0.1)
        fig.subplots_adjust(wspace=0.05)
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = "Helvetica"
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['font.size'] = 16
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16

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

        ax1 = fig.add_subplot(111)

        #ax1.scatter(lct1, v1, marker='o', c='',edgecolors='orange',label="CTL")
        #ax1.scatter(lct1, v2, marker='o', c='',edgecolors='green',label="NEW")


        #lct = [0,1,2]
        #qh_mean1 = np.mean(v1, axis=0)
        #qh_mean2 = np.mean(v2, axis=0)
        #for i in np.arange(0,8):
        '''
        ax1.plot(lct1[1,:], v1[1,:], c='orange', ls= '-', label="HW1_Ctl")
        ax1.plot(lct2[1,:], v2[1,:], c='royalblue', ls= '-',  label="HW1_Best_β-std")
        ax1.plot(lct3[1,:], v3[1,:], c='forestgreen', ls= '-',  label="HW1_Best_β-site")
        ax1.plot(lct1[6,:], v1[6,:], c='orange', ls= '--', label="HW2_Ctl")
        ax1.plot(lct2[6,:], v2[6,:], c='royalblue', ls= '--',  label="HW2_Best_β-std")
        ax1.plot(lct3[6,:], v3[6,:], c='forestgreen', ls= '--',  label="HW2_Best_β-site")
        ax1.plot(lct1[7,:], v1[7,:], c='orange', ls= ':', label="HW3_Ctl")
        ax1.plot(lct2[7,:], v2[7,:], c='royalblue', ls= ':',  label="HW3_Best_β-std")
        ax1.plot(lct3[7,:], v3[7,:], c='forestgreen', ls= ':',  label="HW3_Best_β-site")
        '''
        ax1.plot(lct1[0,:], v1[0,:], c='orange', ls= ':', label="HW1-CTL")
        ax1.plot(lct2[0,:], v2[0,:], c='green', ls= ':',  label="HW1-NEW")
        ax1.plot(lct1[1,:], v1[1,:], c='orange', ls= '-', label="HW2-CTL")
        ax1.plot(lct2[1,:], v2[1,:], c='green', ls= '-',  label="HW2-NEW")
        ax1.plot(lct1[2,:], v1[2,:], c='orange', ls= '--', label="HW3-CTL")
        ax1.plot(lct2[2,:], v2[2,:], c='green', ls= '--',  label="HW3-NEW")
        ax1.plot(lct1[3,:], v1[3,:], c='orange', ls= '-.', label="HW4-CTL")
        ax1.plot(lct2[3,:], v2[3,:], c='green', ls= '-.',  label="HW4-NEW")
        ax1.plot(lct1[4,:], v1[4,:], c='red', ls= ':', label="HW5-CTL")
        ax1.plot(lct2[4,:], v2[4,:], c='blue', ls= ':',  label="HW5-NEW")
        ax1.plot(lct1[5,:], v1[5,:], c='red', ls= '-', label="HW6-CTL")
        ax1.plot(lct2[5,:], v2[5,:], c='blue', ls= '-',  label="HW6-NEW")
        ax1.plot(lct1[6,:], v1[6,:], c='red', ls= '--', label="HW7-CTL")
        ax1.plot(lct2[6,:], v2[6,:], c='blue', ls= '--',  label="HW7-NEW")
        ax1.plot(lct1[7,:], v1[7,:], c='red', ls= '-.', label="HW8-CTL")
        ax1.plot(lct2[7,:], v2[7,:], c='blue', ls= '-.',  label="HW8-NEW")

        ax1.set_xlim(0.5,5.5)

        ax1.set_title('')
        cleaner_dates = ["1","2","3","4","5"]
        xtickslocs = [1,2,3,4,5]
        ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        if term == "Qle":
            ax1.set_ylabel("Latent Heat $(W m^{-2})$")
            ax1.set_ylim(0,250)
        elif term == "Qh":
            ax1.set_ylabel("Sensible Heat $(W m^{-2})$")
            ax1.set_ylim(-100,100)
        ax1.set_xlabel("Days During Heatwave")
        ax1.legend()

        '''
        soilmoist_tdr = np.mean(soilmoist_tdr_rn1, axis=0)
        soilmoist_max1 = np.max(soilmoist_rn1, axis=0)
        soilmoist_max2 = np.max(soilmoist_rn2, axis=0)
        soilmoist_max_tdr = np.max(soilmoist_tdr_rn1, axis=0)
        soilmoist_min1 = np.min(soilmoist_rn1, axis=0)
        soilmoist_min2 = np.min(soilmoist_rn2, axis=0)
        soilmoist_min_tdr = np.min(soilmoist_tdr_rn1, axis=0)
        evap1 = np.mean(evap_rn1, axis=0)
        evap2 = np.mean(evap_rn2, axis=0)
        evap_max1 = np.max(evap_rn1, axis=0)
        evap_max2 = np.max(evap_rn2, axis=0)
        evap_min1 = np.max(evap_rn1, axis=0)
        evap_min2 = np.max(evap_rn2, axis=0)
        ax1.plot(lct, soilmoist1, c='orange', label="CTL")
        ax1.plot(lct, soilmoist2, c='green', label="NEW")
        ax1.plot(lct, soilmoist_tdr, c='blue', label="OBS")
        ax1.fill_between(lct, soilmoist_min1, soilmoist_max1, alpha=0.5, edgecolor='', facecolor='orange')
        ax1.fill_between(lct, soilmoist_min2, soilmoist_max2, alpha=0.5, edgecolor='', facecolor='green')
        ax1.fill_between(lct, soilmoist_min_tdr, soilmoist_max_tdr, alpha=0.5, edgecolor='', facecolor='blue')

        ax2.plot(lct, evap1, c='orange', label="CTL")
        ax2.plot(lct, evap2, c='green', label="NEW")
        ax2.fill_between(lct, evap_min1, evap_max1, alpha=0.5, edgecolor='', facecolor='orange')
        ax2.fill_between(lct, evap_min2, evap_max2, alpha=0.5, edgecolor='', facecolor='green')
        '''
        '''
        ax1.scatter(lct, soilmoist1, marker='o', c='',edgecolors='orange', label="CTL") # s=2.,
        ax1.scatter(lct, soilmoist2, marker='o', c='',edgecolors='green', label="NEW")
        ax1.scatter(lct, soilmoist_tdr, marker='o', c='',edgecolors='blue', label="OBS")

        ax2.scatter(lct, evap1, marker='o', c='',edgecolors='orange',label="CTL")
        ax2.scatter(lct, evap2, marker='o', c='',edgecolors='green',label="NEW")
        '''
        #ax1.set_xlim(-0.5,2.5)
        # ax1.set_ylim(-0.1,1.)

        fig.savefig("EucFACE_%s_during_HW_%s_1.png" % (term,ring),bbox_inches='tight')#, pad_inches=0.1)
        #ax1.set_ylim(-100,200)
        #fig.savefig("EucFACE_Rnet-G_during_HW_%s.png" % (ring),bbox_inches='tight')#, pad_inches=0.1)

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
    ax1.set_ylabel("Rainfall ($mm$ $mon^{-1}$)")
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
