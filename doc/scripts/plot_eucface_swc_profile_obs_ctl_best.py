#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-10-5"
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

def plot_profile(fctl, fbest, ring, contour):

    fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    neo = pd.read_csv(fobs, usecols = ['Ring','Depth','Date','VWC'])
    # usecols : read specific columns from CSV

    # translate datetime
    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)
    #  unit='D', origin=pd.Timestamp('2012-01-01')

    datemark   = neo['Date'].unique()
    datemark   = np.sort(datemark)
    print(datemark)

    # turn datetime64[ns] into timedelta64[ns] since 2011-12-31, e.g. 2012-1-1 as 1 days
    neo['Date'] = neo['Date'] - pd.datetime(2011,12,31)

    # extract days as integers from a timedelta64[ns] object
    neo['Date'] = neo['Date'].dt.days

    # sort by 'Date','Depth'
    neo = neo.sort_values(by=['Date','Depth'])

    print(neo['Depth'].unique())

    # divide neo into groups
    if ring == 'amb':
        subset = neo[neo['Ring'].isin(['R2','R3','R6'])]
    elif ring == 'ele':
        subset = neo[neo['Ring'].isin(['R1','R4','R5'])]
    else:
        subset = neo[neo['Ring'].isin([ring])]

    # calculate the mean of every group ( and unstack #.unstack(level=0)
    subset = subset.groupby(by=["Depth","Date"]).mean()
    print(subset)
    # remove 'VWC'
    subset = subset.xs('VWC', axis=1, drop_level=True)

    # Interpolate
    if contour:
        x     = np.concatenate((subset[(25)].index.values,               \
                                subset.index.get_level_values(1).values, \
                                subset[(450)].index.values ))
        y     = np.concatenate(([0]*len(subset[(25)]),                  \
                                subset.index.get_level_values(0).values, \
                                [460]*len(subset[(25)])    ))
        value =  np.concatenate((subset[(25)].values, subset.values, subset[(450)].values))
    else :
        x     = subset.index.get_level_values(1).values
        y     = subset.index.get_level_values(0).values
        value = subset.values

    print(subset[(25)].index.values)
    X     = subset[(25)].index.values[20:]
    Y     = np.arange(0,465,5)

    grid_X, grid_Y = np.meshgrid(X,Y)
    print(grid_X.shape)
    # interpolate
    if contour:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    else:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='nearest')
    print(grid_data.shape)

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

    cmap = plt.cm.viridis_r

    if contour:
        levels = np.arange(0.,52.,2.)
        img = ax1.contourf(grid_data, cmap=cmap, origin="upper", levels=levels)
        Y_labels = np.flipud(Y)
    else:
        img = ax1.imshow(grid_data, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels = Y

    cbar = fig.colorbar(img, orientation="vertical", shrink=.6, pad=0.02)  #"horizontal" bbox_inches='tight', pad=0.1,
    cbar.set_label('VWC Obs (%)')#('Volumetric soil water content (%)')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # every second tick
    ax1.set_yticks(np.arange(len(Y))[::20])
    ax1.set_yticklabels(Y_labels[::20])
    plt.setp(ax1.get_xticklabels(), visible=False)

    for i in range(len(datemark)):
        print(i, datemark[i]) # xtickslocs[i]

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [0,19,37,52,66,74,86]

    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_ylabel("Depth (cm)")
    ax1.axis('tight')

# _________________________ CTL ___________________________
    ctl  = nc.Dataset(fctl, 'r')

    Time = nc.num2date(ctl.variables['time'][:],ctl.variables['time'].units)
    SoilMoist = pd.DataFrame(ctl.variables['SoilMoist'][:,:,0,0], columns=[1.1, 5.1, 15.7, 43.85, 118.55, 316.4])

    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.stack() # turn multi-columns into one-column
    SoilMoist = SoilMoist.reset_index() # remove index 'dates'
    SoilMoist = SoilMoist.rename(index=str, columns={"level_1": "Depth"})
    SoilMoist = SoilMoist.sort_values(by=['Depth','dates'])

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    print(dates)
    if contour:
        x_ctl     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
        y_ctl     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
        value_ctl = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                       SoilMoist.iloc[:,2].values,         \
                                       SoilMoist.iloc[-(ntimes):,2].values ))
    else:
        x_ctl     = SoilMoist['dates'].values
        y_ctl     = SoilMoist['Depth'].values
        value_ctl = SoilMoist.iloc[:,2].values

    value_ctl = value_ctl*100.
    X_ctl     = X #np.arange(date_start_ctl,date_end_ctl,1) # 2013-1-1 to 2016-12-31
    Y_ctl     = np.arange(0,465,5)
    grid_X_ctl, grid_Y_ctl = np.meshgrid(X_ctl,Y_ctl)

    # interpolate
    if contour:
        grid_ctl = griddata((x_ctl, y_ctl) , value_ctl, (grid_X_ctl, grid_Y_ctl),\
                 method='cubic')
    else:
        grid_ctl = griddata((x_ctl, y_ctl) , value_ctl, (grid_X_ctl, grid_Y_ctl),\
                 method='nearest')

    ax2 = fig.add_subplot(312)#, sharey = ax1)#(nrows=2, ncols=2, index=2, sharey=ax1)

    if contour:
        img2 = ax2.contourf(grid_ctl, cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels2 = np.flipud(Y)
    else:
        img2 = ax2.imshow(grid_ctl, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels2 = Y

    cbar2 = fig.colorbar(img2, orientation="vertical", shrink=.6, pad=0.02) #  bbox_inches='tight', pad=0.1,
    cbar2.set_label('VWC CABLE CTL (%)')#('Volumetric soil water content (%)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()

    # every second tick
    ax2.set_yticks(np.arange(len(Y_ctl))[::20])
    ax2.set_yticklabels(Y_labels2[::20])
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("Depth (cm)")
    ax2.axis('tight')

    SoilMoist = None

# _________________________ BEST ___________________________
    best  = nc.Dataset(fbest, 'r')

    Time = nc.num2date(best.variables['time'][:],best.variables['time'].units)
    SoilMoist = pd.DataFrame(best.variables['SoilMoist'][:,:,0,0], columns = \
                   [7.5,   22.5 , 37.5 , 52.5 , 67.5 , 82.5 , 97.5 , \
                    112.5, 127.5, 142.5, 157.5, 172.5, 187.5, 202.5, \
                    217.5, 232.5, 247.5, 262.5, 277.5, 292.5, 307.5, \
                    322.5, 337.5, 352.5, 367.5, 382.5, 397.5, 412.5, \
                    427.5, 442.5, 457.5 ])

    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.stack() # turn multi-columns into one-column
    SoilMoist = SoilMoist.reset_index() # remove index 'dates'
    SoilMoist = SoilMoist.rename(index=str, columns={"level_1": "Depth"})
    SoilMoist = SoilMoist.sort_values(by=['Depth','dates'])

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    print(dates)
    if contour:
        x_best     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
        y_best     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
        value_best = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                       SoilMoist.iloc[:,2].values,         \
                                       SoilMoist.iloc[-(ntimes):,2].values ))
    else:
        x_best     = SoilMoist['dates'].values
        y_best     = SoilMoist['Depth'].values
        value_best = SoilMoist.iloc[:,2].values

    value_best = value_best*100.
    X_best     = X
    Y_best     = np.arange(0,465,5)
    grid_X_best, grid_Y_best = np.meshgrid(X_best,Y_best)

    # interpolate
    if contour:
        grid_best = griddata((x_best, y_best) , value_best, (grid_X_best, grid_Y_best),\
                 method='cubic')
    else:
        grid_best = griddata((x_best, y_best) , value_best, (grid_X_best, grid_Y_best),\
                 method='nearest')

    ax3 = fig.add_subplot(312)#, sharey = ax1)#(nrows=2, ncols=2, index=2, sharey=ax1)

    if contour:
        img3 = ax3.contourf(grid_best, cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax3.imshow(grid_best, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels3 = Y

    cbar3 = fig.colorbar(img3, orientation="vertical", shrink=.6, pad=0.02) #  bbox_inches='tight', pad=0.1,
    cbar3.set_label('VWC CABLE BEST (%)')#('Volumetric soil water content (%)')
    tick_locator3 = ticker.MaxNLocator(nbins=5)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    # every second tick
    ax3.set_yticks(np.arange(len(Y_ctl))[::20])
    ax3.set_yticklabels(Y_labels3[::20])
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

    if contour == True:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_contour_obs-ctl-best.png" , bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_obs-ctl-best.png" , bbox_inches='tight', pad_inches=0.1)
