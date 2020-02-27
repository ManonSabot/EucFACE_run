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

def plot_profile(fcable, case_name, ring, contour, layer):

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
    # 'VWC' : key on which to get cross section
    # axis=1 : get cross section of column
    # drop_level=True : returns cross section without the multilevel index

    #neo_mean = np.transpose(neo_mean)

# ___________________ From Pandas to Numpy __________________________
#    date_start = pd.datetime(2013,1,1) - pd.datetime(2011,12,31)
#    date_end   = pd.datetime(2017,1,1) - pd.datetime(2011,12,31)
#    date_start = pd.datetime(2012,4,30) - pd.datetime(2011,12,31)
#    date_end   = pd.datetime(2019,5,11) - pd.datetime(2011,12,31)
#    date_start = date_start.days
#    date_end   = date_end.days

    # Interpolate
    if contour:
        x     = np.concatenate((subset[(25)].index.values,               \
                                subset.index.get_level_values(1).values, \
                                subset[(450)].index.values ))            # time
        y     = np.concatenate(([0]*len(subset[(25)]),                  \
                                subset.index.get_level_values(0).values, \
                                [460]*len(subset[(25)])    ))
        value =  np.concatenate((subset[(25)].values, subset.values, subset[(450)].values))
    else :
        x     = subset.index.get_level_values(1).values #, \
                #                np.concatenate((subset[(25)].index.values,               \
                #                subset[(450)].index.values ))              # time
        y     = subset.index.get_level_values(0).values#, \
                #                np.concatenate(([0]*len(subset[(25)]),                   \
                #                [460]*len(subset[(25)])    ))
        value = subset.values
                #                np.concatenate((subset[(25)].values,
                #                , subset[(450)].values))
    # get_level_values(1) : Return an Index of values for requested level.
    # add Depth = 0 and Depth = 460

    print(subset[(25)].index.values)
    # add the 12 depths to 0
    X     = subset[(25)].index.values[20:] #np.arange(date_start,date_end,1) # 2012-4-30 to 2019-5-11
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

    ax1 = fig.add_subplot(311) #(nrows=2, ncols=2, index=1)

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

    #ax1.set_xticks(np.arange(len(X)))
    #cleaner_dates = X
    #ax1.set_xticklabels(cleaner_dates)

    #datemark = np.arange(np.datetime64('2013-01-01','D'), np.datetime64('2017-01-01','D'))
    #xtickslocs = ax1.get_xticks()

    for i in range(len(datemark)):
        print(i, datemark[i]) # xtickslocs[i]

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
                    #["2012","2013","2014","2015","2016","2017","2018","2019"]
                    #["2012-04","2013-01","2014-01","2015-01","2016-01",\
                    # "2017-03","2018-01","2019-01",]
    xtickslocs = [0,19,37,52,66,74,86]
                 #[1,20,39,57,72,86,94,106]

    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_ylabel("Depth (cm)")
    ax1.axis('tight')

#    plt.show()

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')

    Time = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    if layer == "6":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=[1.1, 5.1, 15.7, 43.85, 118.55, 316.4])
    elif layer == "31uni":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns = \
                   [7.5,   22.5 , 37.5 , 52.5 , 67.5 , 82.5 , 97.5 , \
                    112.5, 127.5, 142.5, 157.5, 172.5, 187.5, 202.5, \
                    217.5, 232.5, 247.5, 262.5, 277.5, 292.5, 307.5, \
                    322.5, 337.5, 352.5, 367.5, 382.5, 397.5, 412.5, \
                    427.5, 442.5, 457.5 ])
    elif layer == "31exp":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns = \
                    [ 1.021985, 2.131912, 2.417723, 2.967358, 3.868759, 5.209868,\
                    7.078627, 9.562978, 12.75086, 16.73022, 21.58899, 27.41512,\
                    34.29655, 42.32122, 51.57708, 62.15205, 74.1341 , 87.61115,\
                    102.6711, 119.402 , 137.8918, 158.2283, 180.4995, 204.7933,\
                    231.1978, 259.8008, 290.6903, 323.9542, 359.6805, 397.9571,\
                    438.8719 ])
    elif layer == "31para":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns = \
                   [ 1.000014,  3.47101, 7.782496, 14.73158, 24.11537, 35.73098, \
                     49.37551, 64.84607, 81.93976, 100.4537, 120.185 , 140.9308, \
                     162.4881, 184.6541, 207.2259, 230.    , 252.7742, 275.346 , \
                     297.512 , 319.0693, 339.8151, 359.5464, 378.0603, 395.154 , \
                     410.6246, 424.2691, 435.8847, 445.2685, 452.2176, 456.5291, \
                     459.0001 ])
    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.stack() # turn multi-columns into one-column
    SoilMoist = SoilMoist.reset_index() # remove index 'dates'
    SoilMoist = SoilMoist.rename(index=str, columns={"level_1": "Depth"})
    SoilMoist = SoilMoist.sort_values(by=['Depth','dates'])
    # rename columns level_1 to Depth
    #SoilMoist = SoilMoist.set_index('Depth')

    # Interpolate
#    date_start_cable = pd.datetime(2013,1,1) - pd.datetime(2011,12,31)
#    date_end_cable   = pd.datetime(2017,1,1) - pd.datetime(2011,12,31)
#    date_start_cable = date_start_cable.days
#    date_end_cable   = date_end_cable.days

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    print(dates)
    if contour:
        x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
        y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
        value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                       SoilMoist.iloc[:,2].values,         \
                                       SoilMoist.iloc[-(ntimes):,2].values ))
    else:
        x_cable     = SoilMoist['dates'].values
                      #np.concatenate(( dates,
                      #,dates)) # Time
        y_cable     = SoilMoist['Depth'].values
                      #np.concatenate(([0]*ntimes,
                      #,[460]*ntimes))# Depth
        value_cable = SoilMoist.iloc[:,2].values#,         \
                      #np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                      #SoilMoist.iloc[-(ntimes):,2].values ))
    value_cable = value_cable*100.
    # add the 12 depths to 0
    X_cable     = X #np.arange(date_start_cable,date_end_cable,1) # 2013-1-1 to 2016-12-31
    Y_cable     = np.arange(0,465,5)
    grid_X_cable, grid_Y_cable = np.meshgrid(X_cable,Y_cable)

    # interpolate
    if contour:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='cubic')
                 #'cubic')#'linear')#
    else:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='nearest')
                 #'cubic')#'linear')#'nearest')

    ax2 = fig.add_subplot(312)#, sharey = ax1)#(nrows=2, ncols=2, index=2, sharey=ax1)

    if contour:
        img2 = ax2.contourf(grid_cable, cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels2 = np.flipud(Y)
    else:
        img2 = ax2.imshow(grid_cable, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels2 = Y

    cbar2 = fig.colorbar(img2, orientation="vertical", shrink=.6, pad=0.02) #  bbox_inches='tight', pad=0.1,
    cbar2.set_label('VWC CABLE (%)')#('Volumetric soil water content (%)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()

    # every second tick
    ax2.set_yticks(np.arange(len(Y_cable))[::20])
    ax2.set_yticklabels(Y_labels2[::20])
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("Depth (cm)")
    ax2.axis('tight')

# ________________ plot difference _____________________
    ax3 = fig.add_subplot(313)
    difference = grid_cable -grid_data

    cmap = plt.cm.BrBG

    if contour:
        levels = np.arange(-30.,30.,2.)
        img3 = ax3.contourf(difference, cmap=cmap, origin="upper", levels=levels)
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax3.imshow(difference, cmap=cmap, vmin=-30, vmax=30, origin="upper", interpolation='nearest')
        #'spline16')#'nearest')
        Y_labels3 = Y


    cbar3 = fig.colorbar(img3, orientation="vertical", shrink=.6, pad=0.02) # bbox_inches='tight', pad=0.1,
    cbar3.set_label('CABLE - Obs (%)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    # every second tick
    ax3.set_yticks(np.arange(len(Y_cable))[::20])
    ax3.set_yticklabels(Y_labels3[::20])

    ax3.set_xticks(np.arange(len(X_cable)))
    cleaner_dates3 = X_cable
    ax3.set_xticklabels(cleaner_dates3)


    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')
    if contour == True:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_contour_%s_%s.png" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_%s_%s.png" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)
