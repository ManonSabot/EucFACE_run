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
    SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.02 \
                             + cable.variables['SoilMoist'][:,1,0,0]*0.05 \
                             + cable.variables['SoilMoist'][:,2,0,0]*0.06 \
                             + cable.variables['SoilMoist'][:,3,0,0]*0.13 \
                             + cable.variables['SoilMoist'][:,3,0,0]*(0.5-0.02-0.05-0.06-0.13) )/0.5

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

    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0],columns=['Tair'])
    Tair = Tair - 273.15
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')
    Tair = Tair.resample("D").agg('mean')
    Tair.index = Tair.index - pd.datetime(2011,12,31)
    Tair.index = Tair.index.days

    VegT = pd.DataFrame(cable.variables['VegT'][:,0,0],columns=['VegT'])
    VegT = VegT - 273.15
    VegT['dates'] = Time
    VegT = VegT.set_index('dates')
    VegT = VegT.resample("D").agg('mean')
    VegT.index = VegT.index - pd.datetime(2011,12,31)
    VegT.index = VegT.index.days

    Qair = pd.DataFrame(cable.variables['Qair'][:,0,0],columns=['Qair'])
    Qair['dates'] = Time
    Qair = Qair.set_index('dates')
    Qair = Qair.resample("D").agg('mean')
    Qair.index = Qair.index - pd.datetime(2011,12,31)
    Qair.index = Qair.index.days

    Wind = pd.DataFrame(cable.variables['Wind'][:,0,0],columns=['Wind'])
    Wind['dates'] = Time
    Wind = Wind.set_index('dates')
    Wind = Wind.resample("D").agg('mean')
    Wind.index = Wind.index - pd.datetime(2011,12,31)
    Wind.index = Wind.index.days

    Rnet = pd.DataFrame(cable.variables['Rnet'][:,0,0],columns=['Rnet'])
    Rnet['dates'] = Time
    Rnet = Rnet.set_index('dates')
    Rnet = Rnet.resample("D").agg('mean')
    Rnet.index = Rnet.index - pd.datetime(2011,12,31)
    Rnet.index = Rnet.index.days

    Fwsoil = pd.DataFrame(cable.variables['Fwsoil'][:,0,0],columns=['Fwsoil'])
    Fwsoil['dates'] = Time
    Fwsoil = Fwsoil.set_index('dates')
    Fwsoil = Fwsoil.resample("D").agg('mean')
    Fwsoil.index = Fwsoil.index - pd.datetime(2011,12,31)
    Fwsoil.index = Fwsoil.index.days

    swilt = np.zeros(len(Rainf))
    swilt[:] =(cable.variables['swilt'][0]*0.02 + cable.variables['swilt'][1]*0.05 \
             + cable.variables['swilt'][2]*0.06 + cable.variables['swilt'][3]*0.13 \
             + cable.variables['swilt'][4]*(0.5-0.02-0.05-0.06-0.13) )/0.5
    sfc = np.zeros(len(Rainf))
    sfc[:] =(cable.variables['sfc'][0]*0.02 + cable.variables['sfc'][1]*0.05 \
           + cable.variables['sfc'][2]*0.06 + cable.variables['sfc'][3]*0.13 \
           + cable.variables['sfc'][4]*(0.5-0.02-0.05-0.06-0.13) )/0.5
    ssat = np.zeros(len(Rainf))
    ssat[:] =(cable.variables['ssat'][0]*0.02 + cable.variables['ssat'][1]*0.05 \
            + cable.variables['ssat'][2]*0.06 + cable.variables['ssat'][3]*0.13 \
            + cable.variables['ssat'][4]*(0.5-0.02-0.05-0.06-0.13) )/0.5

# ____________________ Plot obs _______________________
    fig = plt.figure(figsize=[15,10])#,constrained_layout=True)
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

    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)
    ax8 = fig.add_subplot(428)

    width = 1.
    x = np.arange(len(Rainf.index))
    print(x)
    print(width)
    #print(Rainf.values)
    #rects1 = ax1.bar(x, Qrf['Qrf'], width, color='royalblue', label='Qrf')

    ax1.plot(x, subset.values,   c="green", lw=1.0, ls="-", label="tdr")
    ax1.plot(x, SoilMoist.values,c="orange", lw=1.0, ls="-", label="swc")
    ax1.plot(x, swilt,           c="black", lw=1.0, ls="-", label="swilt")
    ax1.plot(x, sfc,             c="black", lw=1.0, ls="-.", label="sfc")
    ax1.plot(x, ssat,            c="black", lw=1.0, ls=":", label="ssat")
    ax3.plot(x, Fwsoil.values,   c="forestgreen", lw=1.0, ls="-", label="Fwsoil")
    ax5.plot(x, TVeg['TVeg'].rolling(window=7).mean(),     c="green", lw=1.0, ls="-", label="TVeg")
    ax5.plot(x, ESoil['ESoil'].rolling(window=7).mean(),    c="orange", lw=1.0, ls="-", label="ESoil")
    ax7.plot(x, Tair['Tair'].rolling(window=7).mean(),     c="red",    lw=1.0, ls="-", label="Tair")
    ax7.plot(x, VegT['VegT'].rolling(window=7).mean(),     c="orange", lw=1.0, ls="-", label="VegT")

    rects2 = ax2.bar(x, Rainf['Rainf'], width, color='royalblue', label='Obs')
    ax4.plot(x, Qair['Qair'].rolling(window=7).mean(),     c="royalblue", lw=1.0, ls="-", label="Qair")
    ax6.plot(x, Wind['Wind'].rolling(window=7).mean(),     c="darkgoldenrod", lw=1.0, ls="-", label="Wind")
    ax8.plot(x, Rnet['Rnet'].rolling(window=7).mean(),     c="crimson", lw=1.0, ls="-", label="Rnet")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [1,365,730,1095,1461,1826,2191]

    #plt.setp(ax1.get_xticklabels(), visible=False)
    #ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    #ax1.set_ylabel("CABLE Runoff (mm)")
    #ax1.axis('tight')
    #ax1.set_xlim(0,2374)

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.set_ylabel("VWC (m3/m3)")
    ax1.axis('tight')
    ax1.set_xlim(0,2374)
    ax1.legend()

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax3.set_ylabel("Fwsoil (-)")
    ax3.axis('tight')
    ax3.set_xlim(0,2374)

    plt.setp(ax5.get_xticklabels(), visible=False)
    ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax5.set_ylabel("TVeg, ESoil (mm/day)")
    ax5.axis('tight')
    ax5.set_xlim(0,2374)
    ax5.legend()
    
    ax7.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax7.set_ylabel("Tair, VegT (°C)")
    ax7.axis('tight')
    ax7.set_xlim(0,2374)
    ax7.legend()

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Rain (mm/day)")
    ax2.axis('tight')
    ax2.set_xlim(0,2374)

    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel("Qair (kg/kg)")
    ax4.axis('tight')
    ax4.set_xlim(0,2374)

    plt.setp(ax6.get_xticklabels(), visible=False)
    ax6.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax6.yaxis.tick_right()
    ax6.yaxis.set_label_position("right")
    ax6.set_ylabel("Wind (m/s)")
    ax6.axis('tight')
    ax6.set_xlim(0,2374)

    ax8.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax8.yaxis.tick_right()
    ax8.yaxis.set_label_position("right")
    ax8.set_ylabel("Rnet (W/m²)")
    ax8.axis('tight')
    ax8.set_xlim(0,2374)

    fig.savefig("EucFACE_Prcp-SW_tdr_GW_Or_Hvrd_Nzdpt_13l_amb_%s.png" % (case_name), bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    #case = ['bch1.5','cnsd2','ctl','css1.5','hyds10','sfc2','ssat2','sucs2','swilt2']
    #case = ["swilt0.5_ssat1.5","sfc0.5","sfc0.5_ssat0.5","sfc1.5_ssat1.5","sfc2","ssat0.5","ssat2","swilt","swilt0.5"]
    #case = ["sfc0.5","sfc0.5_ssat0.5","sfc1.5_ssat1.5","sfc2","ssat0.5","ssat2","swilt","swilt0.5","swilt0.5_ssat1.5","swilt0.5_ssat1.5_top5-layer","swilt0.5_ssat1.5_top5-layer_hyds10"]
    case = ["bch1.5","cnsd2","css1.5","ctl","froot","froot_swilt0.5_ssat1.5_top5-layer_hyds10","hyds10",\
            "sfc0.5","sfc0.5_ssat0.5","sfc1.5_ssat1.5","sfc2","ssat0.5","ssat2","sucs2","swilt0.5","swilt0.5_ssat1.5",\
            "swilt0.5_ssat0.75_top5-layer","swilt0.5_ssat0.75_all-layer","swilt0.5_ssat1.5_top5-layer",\
            "swilt0.5_ssat1.5_top5-layer_hyds10","swilt2"]
    for case_name in case:
        fobs = "/short/w35/mm3972/data/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
        fcable ="/g/data/w35/mm3972/cable/EucFACE/EucFACE_run/outputs/13-layer/sensitivity_test/para_test/%s/EucFACE_amb_out.nc" % (case_name)
        main(fobs, fcable, case_name)
