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

def main(fobs, fcable, case_name, ring, layer, cable_version):

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
    subset = subset.xs('swc.tdr', axis=1, drop_level=True)
    #print(subset)
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


    '''
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    Rainf.index = Rainf.index.days

    Fwsoil = pd.DataFrame(cable.variables['Fwsoil'][:,0,0],columns=['Fwsoil'])
    Fwsoil['dates'] = Time
    Fwsoil = Fwsoil.set_index('dates')
    Fwsoil = Fwsoil.resample("D").agg('mean')
    Fwsoil.index = Fwsoil.index - pd.datetime(2011,12,31)
    Fwsoil.index = Fwsoil.index.days
    '''

    swilt      = np.zeros(len(SoilMoist))
    sfc        = np.zeros(len(SoilMoist))
    ssat       = np.zeros(len(SoilMoist))
    effctv_sat = np.zeros(len(SoilMoist))

    if cable_version == "pore_scale_model":
        Watr = cable.variables['Watr'][0]
    elif cable_version == "Mark_latest":
        Watr = 0.02355

    for i in np.arange(len(SoilMoist)):

        swilt[i]= cable.variables['swilt'][0]
        sfc[i]  = cable.variables['sfc'][0]
        ssat[i] = cable.variables['ssat'][0]
        print(Watr)
        print(swilt[i])
        print(sfc[i])
        print(ssat[i])
        print(SoilMoist.values[i])
        effctv_sat[i] = (SoilMoist.values[i] - Watr)/(ssat[i] - Watr)

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

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    width = 1.
    x = SoilMoist.index
    #np.arange(len(Rainf.index))
    #print(x)
    #print(Rainf.values)

    ax1.plot(subset.index, subset.values,   c="green", lw=1.0, ls="-", label="tdr")
    ax1.plot(x, SoilMoist.values,c="orange", lw=1.0, ls="-", label="swc")
    ax1.plot(x, swilt,           c="black", lw=1.0, ls="-", label="swilt")
    ax1.plot(x, sfc,             c="black", lw=1.0, ls="-.", label="sfc")
    ax1.plot(x, ssat,            c="black", lw=1.0, ls=":", label="ssat")
    ax2.plot(x, effctv_sat,      c="forestgreen", lw=1.0, ls="-", label="Fwsoil")
    ax3.plot(x, TVeg['TVeg'].rolling(window=7).mean(),     c="green", lw=1.0, ls="-", label="TVeg")
    ax3.plot(x, ESoil['ESoil'].rolling(window=7).mean(),    c="orange", lw=1.0, ls="-", label="ESoil")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.set_ylabel("VWC (m3/m3)")
    ax1.axis('tight')
    ax1.set_ylim(0,0.5)
    ax1.set_xlim(367,2739)
    ax1.legend()

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax2.set_ylabel("Effective Saturation (-)")
    ax2.axis('tight')
    ax2.set_ylim(0.,1.)
    ax2.set_xlim(367,2739)

    #plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax3.set_ylabel("TVeg, ESoil (mm/day)")
    ax3.axis('tight')
    ax3.set_ylim(0.,4.)
    ax3.set_xlim(367,2739)
    ax3.legend()
    fig.savefig("EucFACE_tdr_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    layer =  "6"

    cases = ["met_only_or_test_CABLE-2.2.3_pore_scale_model_l_new_roughness_soil-off"]
          # ["default-met_only"]
    cable_version =  "pore_scale_model"
        # "Mark_latest","pore_scale_model"

    rings = ["amb"] #["R1","R2","R3","R4","R5","R6","amb","ele"]
    for case_name in cases:
        for ring in rings:
            fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
            fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            main(fobs, fcable, case_name, ring, layer, cable_version)
