#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-9-2"
__changefrom__ = 'plot_eucface_swc_cable_vs_obs_obsved_dates-13-layer.py'

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

    neo = pd.read_csv(fobs, usecols = ['Ring','Depth','Date','VWC'])
    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)
    neo['Date'] = neo['Date'] - pd.datetime(2012,12,31)
    neo['Date'] = neo['Date'].dt.days
    neo = neo.sort_values(by=['Date','Depth'])

    print(neo['Depth'].unique())
    subset = neo[neo['Ring'].isin(['R2','R3','R6'])] # isin(['R1','R4','R5'])]
    subset = subset.groupby(by=["Depth","Date"]).mean()
    subset = subset.xs('VWC', axis=1, drop_level=True)
    subset[:] = subset[:]/100.
    date  = subset[(25)].index.values

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')

    Time = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=\
              [ 1.021985, 2.131912, 2.417723, 2.967358, 3.868759, 5.209868,\
                7.078627, 9.562978, 12.75086, 16.73022, 21.58899, 27.41512,\
                34.29655, 42.32122, 51.57708, 62.15205, 74.1341 , 87.61115,\
                102.6711, 119.402 , 137.8918, 158.2283, 180.4995, 204.7933,\
                231.1978, 259.8008, 290.6903, 323.9542, 359.6805, 397.9571,\
                438.8719 ])
    #columns=[1.,4.5,10.,19.5,41,71,101,131,161,191,221,273.5,386])

    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2012,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.stack() # turn multi-columns into one-column
    SoilMoist = SoilMoist.reset_index() # remove index 'dates'
    SoilMoist = SoilMoist.rename(index=str, columns={"level_1": "Depth"})
    SoilMoist = SoilMoist.sort_values(by=['Depth','dates'])

    date_start_cable = pd.datetime(2013,1,1) - pd.datetime(2012,12,31)
    date_end_cable   = pd.datetime(2019,6,30) - pd.datetime(2012,12,31)
    date_start_cable = date_start_cable.days
    date_end_cable   = date_end_cable.days

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    # add the 12 depths to 0
    X_cable     = np.arange(date_start_cable,date_end_cable,1) # 2013-1-1 to 2019-6-30
    Y_cable     = [25,50,75,100,125,150,200,250,300,350,400,450]
    grid_X_cable, grid_Y_cable = np.meshgrid(X_cable,Y_cable)

    # interpolate
    grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='linear') #'cubic')#'linear')#'nearest')
    print(grid_cable.shape)

# ____________________ Plot obs _______________________
    fig = plt.figure(figsize=[30,15],constrained_layout=True)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    almost_black = '#262626'
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black
    plt.rcParams['text.color'] = almost_black
    plt.rcParams['axes.edgecolor'] = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black
    cmap = plt.cm.viridis_r

    ax1 = fig.add_subplot(431)
    ax2 = fig.add_subplot(432)
    ax3 = fig.add_subplot(433)
    ax4 = fig.add_subplot(434)
    ax5 = fig.add_subplot(435)
    ax6 = fig.add_subplot(436)
    ax7 = fig.add_subplot(437)
    ax8 = fig.add_subplot(438)
    ax9 = fig.add_subplot(439)
    ax10= fig.add_subplot(4,3,10)
    ax11= fig.add_subplot(4,3,11)
    ax12= fig.add_subplot(4,3,12)

    ax1.plot(X_cable, grid_cable[0,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax2.plot(X_cable, grid_cable[1,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax3.plot(X_cable, grid_cable[2,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax4.plot(X_cable, grid_cable[3,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax5.plot(X_cable, grid_cable[4,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax6.plot(X_cable, grid_cable[5,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax7.plot(X_cable, grid_cable[6,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax8.plot(X_cable, grid_cable[7,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax9.plot(X_cable, grid_cable[8,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax10.plot(X_cable, grid_cable[9,:], c="green", lw=1.0, ls="-", label="CABLE")
    ax11.plot(X_cable, grid_cable[10,:],c="green", lw=1.0, ls="-", label="CABLE")
    ax12.plot(X_cable, grid_cable[11,:],c="green", lw=1.0, ls="-", label="CABLE")

    ax1.scatter(date, subset[(25)].values, marker='.', label="obs")
    ax2.scatter(date, subset[(50)].values, marker='.', label="obs")
    ax3.scatter(date, subset[(75)].values, marker='.', label="obs")
    ax4.scatter(date, subset[(100)].values, marker='.', label="obs")
    ax5.scatter(date, subset[(125)].values, marker='.', label="obs")
    ax6.scatter(date, subset[(150)].values, marker='.', label="obs")
    ax7.scatter(date, subset[(200)].values, marker='.', label="obs")
    ax8.scatter(date, subset[(250)].values, marker='.', label="obs")
    ax9.scatter(date, subset[(300)].values, marker='.', label="obs")
    ax10.scatter(date,subset[(350)].values, marker='.', label="obs")
    ax11.scatter(date,subset[(400)].values, marker='.', label="obs")
    ax12.scatter(date,subset[(450)].values, marker='.', label="obs")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [1,365,730,1095,1461,1826,2191]

    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_title("25cm")
    ax1.axis('tight')
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_title("50cm")
    ax2.axis('tight')
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_title("75cm")
    ax3.axis('tight')
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_title("100cm")
    ax4.axis('tight')
    ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax5.set_title("125cm")
    ax5.axis('tight')
    ax6.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax6.set_title("150cm")
    ax6.axis('tight')
    ax7.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax7.set_title("200cm")
    ax7.axis('tight')
    ax8.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax8.set_title("250cm")
    ax8.axis('tight')
    ax9.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax9.set_title("300cm")
    ax9.axis('tight')
    ax10.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax10.set_title("350cm")
    ax10.axis('tight')
    ax11.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax11.set_title("400cm")
    ax11.axis('tight')
    ax12.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax12.set_title("450cm")
    ax12.axis('tight')

    ax1.set_xlim([0,2374])
    ax1.set_ylim([0.0,0.4])
    ax2.set_xlim([0,2374])
    ax2.set_ylim([0.0,0.4])
    ax3.set_xlim([0,2374])
    ax3.set_ylim([0.0,0.4])
    ax4.set_xlim([0,2374])
    ax4.set_ylim([0.0,0.4])
    ax5.set_xlim([0,2374])
    ax5.set_ylim([0.0,0.4])
    ax6.set_xlim([0,2374])
    ax6.set_ylim([0.0,0.4])
    ax7.set_xlim([0,2374])
    ax7.set_ylim([0.0,0.4])
    ax8.set_xlim([0,2374])
    ax8.set_ylim([0.0,0.4])
    ax9.set_xlim([0,2374])
    ax9.set_ylim([0.0,0.4])
    ax10.set_xlim([0,2374])
    ax10.set_ylim([0.0,0.4])
    ax11.set_xlim([0,2374])
    ax11.set_ylim([0.0,0.4])
    ax12.set_xlim([0,2374])
    ax12.set_ylim([0.0,0.4])

    ax1.legend()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax6.get_xticklabels(), visible=False)
    plt.setp(ax7.get_xticklabels(), visible=False)
    plt.setp(ax8.get_xticklabels(), visible=False)
    plt.setp(ax9.get_xticklabels(), visible=False)

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    plt.setp(ax6.get_yticklabels(), visible=False)
    plt.setp(ax8.get_yticklabels(), visible=False)
    plt.setp(ax9.get_yticklabels(), visible=False)
    plt.setp(ax11.get_yticklabels(), visible=False)
    plt.setp(ax12.get_yticklabels(), visible=False)

    plt.suptitle('Volumetric Water Content - %s (m3/m3)' %(case_name))
    fig.savefig("EucFACE_Prcp-SW_neo_GW_Or_Hvrd_31l_amb_%s.png" % (case_name), bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    case = ["31-layer_exp"]

    for case_name in case:
        fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/31-layer/%s/EucFACE_amb_out.nc" % (case_name)
        main(fobs, fcable, case_name)
