#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-10-06"
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
import scipy.stats as stats
from sklearn.metrics import mean_squared_error

def main(fobs, fcable, ring, layer):

    neo = pd.read_csv(fobs, usecols = ['Ring','Depth','Date','VWC'])
    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)
    neo['Date'] = neo['Date'] - pd.datetime(2012,12,31)
    neo['Date'] = neo['Date'].dt.days
    neo = neo.sort_values(by=['Date','Depth'])

    print(neo['Depth'].unique())


    print( pd.datetime(2017,4,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2017,5,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2017,6,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2017,7,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2017,8,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2017,9,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2017,10,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2017,11,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2017,12,1) - pd.datetime(2012,12,31))

    print( pd.datetime(2018,1,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,2,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,3,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,4,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,5,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,6,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,7,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,8,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,9,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,10,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,11,1) - pd.datetime(2012,12,31))
    print( pd.datetime(2018,12,1) - pd.datetime(2012,12,31))
    if ring == 'amb':
        subset = neo[neo['Ring'].isin(['R2','R3','R6'])]
    elif ring == 'ele':
        subset = neo[neo['Ring'].isin(['R1','R4','R5'])]
    else:
        subset = neo[neo['Ring'].isin([ring])]

    subset = subset.groupby(by=["Depth","Date"]).mean()
    subset = subset.xs('VWC', axis=1, drop_level=True)
    subset[:] = subset[:]/100.
    #date  = subset[(25)].index.values

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')

    Time = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    if layer == "6":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=[1.1, 5.1, 15.7, 43.85, 118.55, 316.4])
    elif layer == "13":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns = \
                    [1.,4.5,10.,19.5,41,71,101,131,161,191,221,273.5,386])
    elif layer == "31uni":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns = \
                   [7.5,   22.5 , 37.5 , 52.5 , 67.5 , 82.5 , 97.5 , \
                    112.5, 127.5, 142.5, 157.5, 172.5, 187.5, 202.5, \
                    217.5, 232.5, 247.5, 262.5, 277.5, 292.5, 307.5, \
                    322.5, 337.5, 352.5, 367.5, 382.5, 397.5, 412.5, \
                    427.5, 442.5, 457.5 ])
    elif layer == "31exp":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=\
                  [ 1.021985, 2.131912, 2.417723, 2.967358, 3.868759, 5.209868,\
                    7.078627, 9.562978, 12.75086, 16.73022, 21.58899, 27.41512,\
                    34.29655, 42.32122, 51.57708, 62.15205, 74.1341 , 87.61115,\
                    102.6711, 119.402 , 137.8918, 158.2283, 180.4995, 204.7933,\
                    231.1978, 259.8008, 290.6903, 323.9542, 359.6805, 397.9571,\
                    438.8719 ])
    elif layer == "31para":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=\
                    [ 1.000014,  3.47101, 7.782496, 14.73158, 24.11537, 35.73098, \
                      49.37551, 64.84607, 81.93976, 100.4537, 120.185 , 140.9308, \
                      162.4881, 184.6541, 207.2259, 230.    , 252.7742, 275.346 , \
                      297.512 , 319.0693, 339.8151, 359.5464, 378.0603, 395.154 , \
                      410.6246, 424.2691, 435.8847, 445.2685, 452.2176, 456.5291, \
                      459.0001 ])
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
    x_cable     = SoilMoist['dates'].values
    y_cable     = SoilMoist['Depth'].values
    value_cable = SoilMoist.iloc[:,2].values

    '''
    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    '''
    # add the 12 depths to 0
    X_cable     = np.arange(date_start_cable,date_end_cable,1) # 2013-1-1 to 2019-6-30
    Y_cable     = [25,50,75,100,125,150,200,250,300,350,400,450]
    grid_X_cable, grid_Y_cable = np.meshgrid(X_cable,Y_cable)

    # interpolate
    grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='nearest') #'cubic')#'linear')#'nearest')
    print(grid_cable.shape)

    return X_cable, grid_cable,subset;

if __name__ == "__main__":

    rings = ["amb"] #["R1","R2","R3","R4","R5","R6","amb","ele"]

    for ring in rings:
        fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
        layer =  "6"
        fcable1 ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/default-met_only_or-off/EucFACE_%s_out.nc" % (ring)
        X_cable1, grid_cable1,subset1 = main(fobs, fcable1, ring, layer)

        layer =  "31uni"
        fcable2 ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/ctl_met_LAI_vrt_SM_swilt-watr_31uni_HDM_or-off-litter_Hvrd/EucFACE_%s_out.nc" % ( ring)
        X_cable2, grid_cable2,subset2 = main(fobs, fcable2, ring, layer)

# ____________________ Plot obs _______________________
    fig = plt.figure(figsize=[10,7],constrained_layout=True)
    fig.subplots_adjust(hspace=0.2)
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

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    '''
    ax1.plot(X_cable1, grid_cable1[0,:]-grid_cable1[0,1546],  c="orange", lw=1.0, ls="-", label="CTL")
    ax2.plot(X_cable1, grid_cable1[3,:]-grid_cable1[3,1546],  c="orange", lw=1.0, ls="-", label="CTL")
    ax3.plot(X_cable1, grid_cable1[8,:]-grid_cable1[8,1546],  c="orange", lw=1.0, ls="-", label="CTL")

    ax1.plot(X_cable2, grid_cable2[0,:]-grid_cable2[0,1546],  c="green", lw=1.0, ls="-", label="NEW")
    ax2.plot(X_cable2, grid_cable2[3,:]-grid_cable2[3,1546],  c="green", lw=1.0, ls="-", label="NEW")
    ax3.plot(X_cable2, grid_cable2[8,:]-grid_cable2[8,1546],  c="green", lw=1.0, ls="-", label="NEW")

    ax1.scatter(subset1[(25)].index.values, subset1[(25)].values-subset1[(25)][subset1[(25)].index.values == 1547].values, marker='.', label="OBS")
    ax2.scatter(subset1[(100)].index.values, subset1[(100)]-subset1[(100)][subset1[(100)].index.values == 1547].values, marker='.', label="OBS")
    ax3.scatter(subset1[(300)].index.values, subset1[(300)]-subset1[(300)][subset1[(300)].index.values == 1547].values, marker='.', label="OBS")
    '''

    ax1.plot(X_cable1, grid_cable1[0,:],  c="orange", lw=1.0, ls="-", label="CTL")
    ax2.plot(X_cable1, grid_cable1[3,:],  c="orange", lw=1.0, ls="-", label="CTL")
    ax3.plot(X_cable1, grid_cable1[8,:],  c="orange", lw=1.0, ls="-", label="CTL")

    ax1.plot(X_cable2, grid_cable2[0,:],  c="green", lw=1.0, ls="-", label="NEW")
    ax2.plot(X_cable2, grid_cable2[3,:],  c="green", lw=1.0, ls="-", label="NEW")
    ax3.plot(X_cable2, grid_cable2[8,:],  c="green", lw=1.0, ls="-", label="NEW")

    ax1.scatter(subset1[(25)].index.values, subset1[(25)].values, marker='.', label="OBS")
    ax2.scatter(subset1[(100)].index.values, subset1[(100)].values, marker='.', label="OBS")
    ax3.scatter(subset1[(300)].index.values, subset1[(300)].values, marker='.', label="OBS")

    cleaner_dates = ["2017-4","2017-6","2017-8","2017-10","2017-12","2018-2",\
                     "2018-4","2018-6","2018-8","2018-10"]
    xtickslocs    = [1552,1613,1674,1735,1796,1858,\
                     1917,1978,2039,2100]

    dd = [25,50,75,100,125,150,200,250,300,350,400,450]
    cor1_neo = np.zeros(12)
    cor2_neo = np.zeros(12)

    for i,d in enumerate(dd):
        tmp1 = grid_cable1[i,np.isin(X_cable1,subset1[(d)].index)]
        tmp2 = subset1[(d)][np.isin(subset1[(d)].index,X_cable1)]
        mask = tmp2 > 0.0
        tmp1 = tmp1[mask]
        tmp2 = tmp2[mask]
        cor1_neo[i]= stats.pearsonr(tmp1,tmp2)[0]
        tmp1 = grid_cable2[i,np.isin(X_cable2,subset2[(d)].index)]
        tmp2 = subset2[(d)][np.isin(subset2[(d)].index,X_cable2)]
        mask = tmp2 > 0.0
        tmp1 = tmp1[mask]
        tmp2 = tmp2[mask]
        cor2_neo[i]= stats.pearsonr(tmp1,tmp2)[0]

    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_title("25cm")
    #ax1.set_title("25cm, r_CTL=% 5.3f, r_NEW=% 5.3f" %(cor1_neo[0],cor2_neo[0]))
    #ax1.axis('tight')
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_title("100cm")
    #ax2.set_title("100cm, r_CTL=% 5.3f, r_NEW=% 5.3f" %(cor1_neo[3],cor2_neo[3]))
    #ax2.axis('tight')
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_title("300cm")
    #ax3.set_title("300cm, r_CTL=% 5.3f, r_NEW=% 5.3f" %(cor1_neo[8],cor2_neo[8]))
    #ax3.axis('tight')

    ax1.set_xlim([1548,2100]) # 1625
    ax1.set_ylim([0.0,0.3])
    ax2.set_xlim([1548,2100])
    ax2.set_ylim([0.0,0.3])
    ax3.set_xlim([1548,2100])
    ax3.set_ylim([0.0,0.3])

    ax1.legend()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    #plt.suptitle('Volumetric Water Content (m3/m3)')
    fig.savefig("EucFACE_drydown_swc_%s.png" % (ring), bbox_inches='tight', pad_inches=0.1)
