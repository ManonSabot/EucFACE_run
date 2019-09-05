#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-8-27"
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

def main(fobs, fcable, case_name):

    tdr = pd.read_csv(fobs, usecols = ['Ring','Date','swc.tdr'])
    tdr['Date'] = pd.to_datetime(tdr['Date'],format="%Y-%m-%d",infer_datetime_format=False)
    tdr['Date'] = tdr['Date'] - pd.datetime(2011,12,31)
    tdr['Date'] = tdr['Date'].dt.days
    tdr = tdr.sort_values(by=['Date'])
    subset = tdr[(tdr['Ring'].isin(['R2','R3','R6'])) & (tdr.Date > 366)]
    #['R1','R4','R5']
    subset = subset.groupby(by=["Date"]).mean()/100.
    subset = subset.xs('swc.tdr', axis=1, drop_level=True)

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])
    SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                             + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                             + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                             + cable.variables['SoilMoist'][:,3,0,0]*(0.5-0.022-0.058-0.154) )/0.5

    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.sort_values(by=['dates'])

    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    Rainf.index = Rainf.index.days

    TVeg = pd.DataFrame(cable.variables['TVeg'][:,0,0],columns=['TVeg'])
    TVeg = TVeg*1800.
    TVeg['dates'] = Time
    TVeg = TVeg.set_index('dates')
    TVeg = TVeg.resample("D").agg('sum')
    TVeg.index = TVeg.index - pd.datetime(2011,12,31)
    TVeg.index = TVeg.index.days

    ESoil = pd.DataFrame(cable.variables['ESoil'][:,0,0],columns=['ESoil'])
    ESoil = ESoil*1800.
    ESoil['dates'] = Time
    ESoil = ESoil.set_index('dates')
    ESoil = ESoil.resample("D").agg('sum')
    ESoil.index = ESoil.index - pd.datetime(2011,12,31)
    ESoil.index = ESoil.index.days

    Qrf = pd.DataFrame(cable.variables['Qs'][:,0,0],columns=['Qrf'])
    Qrf['Qrf'] = (cable.variables['Qs'][:,0,0]+ cable.variables['Qsb'][:,0,0])*1800.
    print(Qrf)
    Qrf['dates'] = Time
    Qrf = Qrf.set_index('dates')
    Qrf = Qrf.resample("D").agg('sum')
    Qrf.index = Qrf.index - pd.datetime(2011,12,31)
    Qrf.index = Qrf.index.days

    wilt_point = np.zeros(len(Rainf))
    wilt_point[:] = (0.06343579*0.022+ 0.06343579*0.058 + 0.05908652*0.154 + 0.1029432*(0.5-0.022-0.058-0.154) )/0.5

# ____________________ Plot obs _______________________
    fig = plt.figure(figsize=[15,10])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 10
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

    ax1 = fig.add_subplot(411) #(nrows=2, ncols=2, index=1)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    width = 1.
    x = np.arange(len(Rainf.index))
    print(x)
    print(width)
    #print(Rainf.values)
    rects1 = ax1.bar(x, Qrf['Qrf'], width, color='royalblue', label='Qrf')
    rects2 = ax2.bar(x, Rainf['Rainf'], width, color='royalblue', label='Obs')
    #bar_plot = ax1.bar(x, Rainf.values, color='royalblue', label='rain')
    ax3.plot(x, subset.values,   c="green", lw=1.0, ls="-", label="tdr")
    ax3.plot(x, SoilMoist.values,c="orange", lw=1.0, ls="-", label="swc")
    ax3.plot(x, wilt_point,    c="black", lw=1.0, ls="-", label="swilt")
    ax4.plot(x, TVeg.values,     c="green", lw=1.0, ls="-", label="TVeg")
    ax4.plot(x, ESoil.values,    c="orange", lw=1.0, ls="-", label="ESoil")
     
    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [1,365,730,1095,1461,1826,2191]

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.set_ylabel("CABLE Runoff (mm)")
    ax1.axis('tight')
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax2.set_ylabel("Obs Rain (mm)")
    ax2.axis('tight')
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax3.set_ylabel("VWC (m3/m3)")
    ax3.axis('tight')

    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax4.set_ylabel("TVeg, ESoil (m3/m3)")
    ax4.axis('tight')

    fig.savefig("EucFACE_Prcp-SW_tdr_GW_Or_Hvrd_Nzdpt_6l_amb_%s.png" % (case_name), bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    case_name = "hyds"
    fobs = "/short/w35/mm3972/data/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    fcable ="/g/data/w35/mm3972/cable/EucFACE/EucFACE_run/outputs/hyds_test/6layer_hyds_test/no_zdepth/%s/EucFACE_amb_out.nc" % (case_name)
    main(fobs, fcable, case_name)
