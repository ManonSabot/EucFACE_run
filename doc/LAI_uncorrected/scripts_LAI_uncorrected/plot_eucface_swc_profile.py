#!/usr/bin/env python

"""
draw soil moisture profile

include functions:

    plot_profile
    plot_profile_3
    plot_profile_tdr_ET
    plot_dry_down

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
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import griddata
from plot_eucface_get_var import *
import matplotlib.font_manager
from matplotlib import rc

def plot_profile(fcable, case_name, ring, contour, layer):

    subset = read_obs_swc_neo(ring)

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
    SoilMoist = read_cable_SM(fcable, layer)

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
        fig.savefig("../plots/EucFACE_SW_obsved_dates_contour_%s_%s.png" % (os.path.basename(fcable).split("/")[-2], ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_%s_%s.png" % (os.path.basename(fcable).split("/")[-2], ring), bbox_inches='tight', pad_inches=0.1)

def plot_profile_3(fctl, fbest, ring, contour):

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
    cbar2.set_label('VWC CABLE Ctl (%)')#('Volumetric soil water content (%)')
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

    ax3 = fig.add_subplot(313)#, sharey = ax1)#(nrows=2, ncols=2, index=2, sharey=ax1)

    if contour:
        img3 = ax3.contourf(grid_best, cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax3.imshow(grid_best, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels3 = Y

    cbar3 = fig.colorbar(img3, orientation="vertical", shrink=.6, pad=0.02) #  bbox_inches='tight', pad=0.1,
    cbar3.set_label('VWC CABLE Best (%)')#('Volumetric soil water content (%)')
    tick_locator3 = ticker.MaxNLocator(nbins=5)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    # every second tick
    ax3.set_yticks(np.arange(len(Y_ctl))[::20])
    ax3.set_yticklabels(Y_labels3[::20])
    plt.setp(ax3.get_xticklabels(), visible=True)

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

    if contour == True:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_contour_obs-ctl-best.png" , bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_obs-ctl-best.png" , bbox_inches='tight', pad_inches=0.1)

def plot_profile_tdr_ET(fcable, ring, contour, layer):

    """
    plot simulation status and fluxes
    """

    # ========================= ET FLUX  ============================
    # ===== Obs   =====
    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)

    # ===== CABLE =====
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

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

    # ========================= SM IN TOP 25cm ==========================
    SoilMoist_25cm = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    if layer == "6":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                      + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                      + cable.variables['SoilMoist'][:,3,0,0]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.10 )/0.25

    SoilMoist_25cm['dates'] = Time
    SoilMoist_25cm = SoilMoist_25cm.set_index('dates')
    SoilMoist_25cm = SoilMoist_25cm.resample("D").agg('mean')
    SoilMoist_25cm.index = SoilMoist_25cm.index - pd.datetime(2011,12,31)
    SoilMoist_25cm.index = SoilMoist_25cm.index.days
    SoilMoist_25cm = SoilMoist_25cm.sort_values(by=['dates'])

    # Soil hydraulic param
    swilt = np.zeros(len(TVeg))
    sfc = np.zeros(len(TVeg))
    ssat = np.zeros(len(TVeg))

    if layer == "6":
        swilt[:] = ( cable.variables['swilt'][0]*0.022 + cable.variables['swilt'][1]*0.058 \
                   + cable.variables['swilt'][2]*0.154 + cable.variables['swilt'][3]*(0.25-0.022-0.058-0.154) )/0.25
        sfc[:] = ( cable.variables['sfc'][0]*0.022   + cable.variables['sfc'][1]*0.058 \
                   + cable.variables['sfc'][2]*0.154 + cable.variables['sfc'][3]*(0.25-0.022-0.058-0.154) )/0.25
        ssat[:] = ( cable.variables['ssat'][0]*0.022 + cable.variables['ssat'][1]*0.058 \
                   + cable.variables['ssat'][2]*0.154+ cable.variables['ssat'][3]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        swilt[:] =(cable.variables['swilt'][0]*0.15 + cable.variables['swilt'][1]*0.10 )/0.25
        sfc[:] =(cable.variables['sfc'][0]*0.15 + cable.variables['sfc'][1]*0.10 )/0.25
        ssat[:] =(cable.variables['ssat'][0]*0.15 + cable.variables['ssat'][1]*0.10 )/0.25

    # ========================= SM PROFILE ==========================
    ### 1. Read data
    # ==== SM Obs ====
    neo = read_obs_swc_neo(ring)
    tdr = read_obs_swc_tdr(ring)

    # ===== CABLE =====
    SoilMoist = read_cable_SM_one_clmn(fcable, layer)

    ### 2. interpolate SM
    # === Obs SoilMoist ===
    x     = np.concatenate((neo[(25)].index.values,               \
                            neo.index.get_level_values(1).values, \
                            neo[(450)].index.values ))
    y     = np.concatenate(([0]*len(neo[(25)]),                  \
                            neo.index.get_level_values(0).values, \
                            [460]*len(neo[(25)])    ))
    value =  np.concatenate((neo[(25)].values, neo.values, neo[(450)].values))

    X     = neo[(25)].index.values[20:]
    Y     = np.arange(0,461,1)

    grid_X, grid_Y = np.meshgrid(X,Y)

    # === CABLE SoilMoist ===
    if contour:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    else:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='linear') # 'linear' 'nearest'
    print(type(grid_data))

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    print(dates)

    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    value_cable = value_cable*100.

    # add the 12 depths to 0
    grid_X_cable, grid_Y_cable = np.meshgrid(X,Y)

    if contour:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='cubic')
    else:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='linear')
    difference = grid_cable -grid_data


    # ======================= PLOTTING  ==========================

    if fcable.split("/")[-2] == "met_LAI_6":
        fig = plt.figure(figsize=[9,17.5])
    else:
        fig = plt.figure(figsize=[9,14])

    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.05)
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #fm = matplotlib.font_manager.json_load(os.path.expanduser("~/.cache/matplotlib/fontlist-v310.json"))
    #fm.findfont('sans-serif', rebuild_if_missing=False)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    if fcable.split("/")[-2] == "met_LAI_6":
        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512)
        ax5 = fig.add_subplot(513)
        ax3 = fig.add_subplot(514)
        ax4 = fig.add_subplot(515)
    else:
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

    x = TVeg.index

    # set x-axis values
    cleaner_dates1 = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs1    = [367,732,1097,1462,1828,2193,2558]
    yticks         = [360,260,160,60]
    yticklabels    = ["100","200","300","400"]
    # set color
    cmap = plt.cm.viridis_r

    ax1.plot(x, TVeg['TVeg'].rolling(window=3).mean(),     c="green", lw=1.0, ls="-", label="Trans") #.rolling(window=7).mean()
    ax1.plot(x, ESoil['ESoil'].rolling(window=3).mean(),    c="orange", lw=1.0, ls="-", label="ESoil") #.rolling(window=7).mean()
    ax1.scatter(subs_Trans.index, subs_Trans['obs'].rolling(window=30).mean(), marker='o', c='',edgecolors='blue', s = 2., label="Trans Obs")
    ax1.scatter(subs_Esoil.index, subs_Esoil['obs'].rolling(window=30).mean(), marker='o', c='',edgecolors='red', s = 2., label="ESoil Obs")
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # this order of the setting can affect plot x & y axis
    plt.setp(ax1.get_xticklabels(), visible=True)
    ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    ax1.set_ylabel("T, Es (mm d$^{-1}$)")
    ax1.axis('tight')
    ax1.set_ylim(0.,4.5)
    ax1.set_xlim(367,1097)
    ax1.legend(loc='best', frameon=False)

    #ax1.update_ticks()

    ax2.plot(tdr.index, tdr.values,    c="orange", lw=1.0, ls="-", label="Obs")
    ax2.plot(x, SoilMoist_25cm.values, c="green", lw=1.0, ls="-", label="CABLE")
    ax2.plot(x, swilt,                 c="black", lw=1.0, ls="-", label="swilt")
    ax2.plot(x, sfc,                   c="black", lw=1.0, ls="-.", label="sfc")
    ax2.plot(x, ssat,                  c="black", lw=1.0, ls=":", label="ssat")
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    ax2.set_ylabel("VWC in 25cm (m$^{3}$ m$^{-3}$)")
    ax2.axis('tight') # it should be in front of ylim  and xlim
    ax2.set_ylim(0,0.5)
    ax2.set_xlim(367,2922)
    ax2.legend(loc='best', frameon=False)
    #ax2.update_ticks()

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [0,19,37,52,66,74,86]

    if fcable.split("/")[-2] == "met_LAI_6":

        if contour:
            levels = np.arange(0.,52.,2.)
            img = ax5.contourf(grid_data, cmap=cmap, origin="upper", levels=levels)
            Y_labels = np.flipud(Y)
        else:
            img = ax5.imshow(grid_data, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
            Y_labels = Y

        cbar = fig.colorbar(img, ax = ax5, orientation="vertical", pad=0.02, shrink=.6) #"horizontal"
        cbar.set_label('VWC Obs (%)')#('Volumetric soil water content (%)')
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

        ax5.text(0.02, 0.95, '(c)', transform=ax5.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        # every second tick
        ax5.set_yticks(yticks)
        ax5.set_yticklabels(yticklabels)
        plt.setp(ax5.get_xticklabels(), visible=False)

        ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax5.set_ylabel("Depth (cm)")
        ax5.axis('tight')

    if contour:
        levels = np.arange(0.,52.,2.)
        img2 = ax3.contourf(grid_cable, cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels2 = np.flipud(Y)
    else:
        img2 = ax3.imshow(grid_cable, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels2 = Y

    cbar2 = fig.colorbar(img2, ax = ax3,  orientation="vertical", pad=0.02, shrink=.6)
    cbar2.set_label('VWC CABLE (%)')#('Volumetric soil water content (%)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()
    if fcable.split("/")[-2] == "met_LAI_6":
        ax3.text(0.02, 0.95, '(d)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    else:
        ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # every second tick

    ax3.set_yticks(yticks)
    ax3.set_yticklabels(yticklabels)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

    cmap = plt.cm.BrBG

    if contour:
        levels = np.arange(-30.,30.,2.)
        img3 = ax4.contourf(difference, cmap=cmap, origin="upper", levels=levels)
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax4.imshow(difference, cmap=cmap, vmin=-30, vmax=30, origin="upper", interpolation='nearest')
        Y_labels3 = Y

    cbar3 = fig.colorbar(img3, ax = ax4, orientation="vertical", pad=0.02, shrink=.6)
    cbar3.set_label('CABLE - Obs (%)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    if fcable.split("/")[-2] == "met_LAI_6":
        ax4.text(0.02, 0.95, '(e)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    else:
        ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # every second tick
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(yticklabels)

    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("Depth (cm)")
    ax4.axis('tight')

    #plt.show()
    #fig.align_labels()

    if contour == True:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_ET_contour_%s_%s.png" % (fcable.split("/")[-2], ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_ET_%s_%s.png" % (fcable.split("/")[-2], ring), bbox_inches='tight', pad_inches=0.1)

def plot_profile_tdr_ET_error(fpath, case_name, ring, contour, layer):

    """
    plot simulation status and fluxes
    """

    # ========================= ET FLUX  ============================

    # ===== CABLE =====
    cable = nc.Dataset("%s/EucFACE_amb_out.nc" % fpath, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

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

    T  = np.zeros([3,len(TVeg)])
    Es = np.zeros([3,len(ESoil)])

    T[0,:]  =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'TVeg')['cable'].values
    Es[0,:] =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'ESoil')['cable'].values
    T[1,:]  =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'TVeg')['cable'].values
    Es[1,:] =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'ESoil')['cable'].values
    T[2,:]  =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'TVeg')['cable'].values
    Es[2,:] =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'ESoil')['cable'].values

    TVeg['min']  = T.min(axis=0)
    TVeg['max']  = T.max(axis=0)
    ESoil['min'] = Es.min(axis=0)
    ESoil['max'] = Es.max(axis=0)

    # ===== Obs   =====
    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)

    Es_R2 = read_obs_esoil('R2')['obs']
    Es_R3 = read_obs_esoil('R3')['obs']
    Es_R6 = read_obs_esoil('R6')['obs']

    T_R2 = read_obs_trans('R2')['obs']
    T_R3 = read_obs_trans('R3')['obs']
    T_R6 = read_obs_trans('R6')['obs']

    T_error = np.zeros([3,len(TVeg)])
    Es_error = np.zeros([3,len(TVeg)])

    for date in TVeg.index:
        if np.any(Es_R2.index == date):
            Es_error[0,date-367] = Es_R2[Es_R2.index == date].values
        else:
            Es_error[0,date-367] = float('NaN')
        if np.any(Es_R3.index == date):
            Es_error[1,date-367] = Es_R3[Es_R3.index == date].values
        else:
            Es_error[1,date-367] = float('NaN')
        if np.any(Es_R6.index == date):
            Es_error[2,date-367] = Es_R6[Es_R6.index == date].values
        else:
            Es_error[2,date-367] = float('NaN')

        if np.any(T_R2.index == date):
            T_error[0,date-367] = T_R2[T_R2.index == date].values
        else:
            T_error[0,date-367] = float('NaN')
        if np.any(T_R3.index == date):
            T_error[1,date-367] = T_R3[T_R3.index == date].values
        else:
            T_error[1,date-367] = float('NaN')
        if np.any(T_R6.index == date):
            T_error[2,date-367] = T_R6[T_R6.index == date].values
        else:
            T_error[2,date-367] = float('NaN')

    ESoil['obs_min'] = Es_error.min(axis=0)
    ESoil['obs_max'] = Es_error.max(axis=0)
    TVeg['obs_min']  = T_error.min(axis=0)
    TVeg['obs_max']  = T_error.max(axis=0)

    # ========================= SM IN TOP 25cm ==========================
    SoilMoist_25cm = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    if layer == "6":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                      + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                      + cable.variables['SoilMoist'][:,3,0,0]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.10 )/0.25

    SoilMoist_25cm['dates'] = Time
    SoilMoist_25cm = SoilMoist_25cm.set_index('dates')
    SoilMoist_25cm = SoilMoist_25cm.resample("D").agg('mean')
    SoilMoist_25cm.index = SoilMoist_25cm.index - pd.datetime(2011,12,31)
    SoilMoist_25cm.index = SoilMoist_25cm.index.days
    SoilMoist_25cm = SoilMoist_25cm.sort_values(by=['dates'])

    # Soil hydraulic param
    swilt = np.zeros(len(TVeg))
    sfc = np.zeros(len(TVeg))
    ssat = np.zeros(len(TVeg))

    if layer == "6":
        swilt[:] = ( cable.variables['swilt'][0]*0.022 + cable.variables['swilt'][1]*0.058 \
                   + cable.variables['swilt'][2]*0.154 + cable.variables['swilt'][3]*(0.25-0.022-0.058-0.154) )/0.25
        sfc[:] = ( cable.variables['sfc'][0]*0.022   + cable.variables['sfc'][1]*0.058 \
                   + cable.variables['sfc'][2]*0.154 + cable.variables['sfc'][3]*(0.25-0.022-0.058-0.154) )/0.25
        ssat[:] = ( cable.variables['ssat'][0]*0.022 + cable.variables['ssat'][1]*0.058 \
                   + cable.variables['ssat'][2]*0.154+ cable.variables['ssat'][3]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        swilt[:] =(cable.variables['swilt'][0]*0.15 + cable.variables['swilt'][1]*0.10 )/0.25
        sfc[:] =(cable.variables['sfc'][0]*0.15 + cable.variables['sfc'][1]*0.10 )/0.25
        ssat[:] =(cable.variables['ssat'][0]*0.15 + cable.variables['ssat'][1]*0.10 )/0.25

    # ========================= SM PROFILE ==========================
    ### 1. Read data
    # ==== SM Obs ====
    neo = read_obs_swc_neo(ring)
    tdr = read_obs_swc_tdr(ring)

    # ===== CABLE =====
    SoilMoist = read_cable_SM_one_clmn("%s/EucFACE_amb_out.nc" % fpath, layer)

    ### 2. interpolate SM
    # === Obs SoilMoist ===
    x     = np.concatenate((neo[(25)].index.values,               \
                            neo.index.get_level_values(1).values, \
                            neo[(450)].index.values ))
    y     = np.concatenate(([0]*len(neo[(25)]),                  \
                            neo.index.get_level_values(0).values, \
                            [460]*len(neo[(25)])    ))
    value =  np.concatenate((neo[(25)].values, neo.values, neo[(450)].values))

    X     = neo[(25)].index.values[20:]
    Y     = np.arange(0,461,1)

    grid_X, grid_Y = np.meshgrid(X,Y)

    # === CABLE SoilMoist ===
    if contour:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    else:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='linear') # 'linear' 'nearest'
    print(type(grid_data))

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    print(dates)

    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    value_cable = value_cable*100.

    # add the 12 depths to 0
    grid_X_cable, grid_Y_cable = np.meshgrid(X,Y)

    if contour:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='cubic')
    else:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='linear')
    difference = grid_cable -grid_data


    # ======================= PLOTTING  ==========================

    if case_name == "met_LAI_6":
        fig = plt.figure(figsize=[9,17.5])
    else:
        fig = plt.figure(figsize=[9,14])

    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.05)
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #fm = matplotlib.font_manager.json_load(os.path.expanduser("~/.cache/matplotlib/fontlist-v310.json"))
    #fm.findfont('sans-serif', rebuild_if_missing=False)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    if case_name == "met_LAI_6":
        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512)
        ax5 = fig.add_subplot(513)
        ax3 = fig.add_subplot(514)
        ax4 = fig.add_subplot(515)
    else:
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

    x = TVeg.index

    # set x-axis values
    cleaner_dates1 = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs1    = [367,732,1097,1462,1828,2193,2558]
    # set color
    cmap = plt.cm.viridis_r

    ax1.fill_between(x, TVeg['min'].rolling(window=3).mean(),
        TVeg['max'].rolling(window=3).mean(), color="green", alpha=0.2)
    ax1.fill_between(x, ESoil['min'].rolling(window=3).mean(),
        ESoil['max'].rolling(window=3).mean(), color="orange", alpha=0.2)

    ax1.plot(x, TVeg['TVeg'].rolling(window=3).mean(),
        c="green", lw=1.0, ls="-", label="$T_{CABLE}$") #.rolling(window=7).mean()
    ax1.plot(x, ESoil['ESoil'].rolling(window=3).mean(),
        c="orange", lw=1.0, ls="-", label="$Es_{CABLE}$") #.rolling(window=7).mean()

    ax1.fill_between(x, TVeg['obs_min'].rolling(window=3).mean(),
        TVeg['obs_max'].rolling(window=3).mean(), color="blue", alpha=0.2)
    ax1.fill_between(x, ESoil['obs_min'].rolling(window=3).mean(),
        ESoil['obs_max'].rolling(window=3).mean(), color="red", alpha=0.2)
    ax1.scatter(subs_Trans.index, subs_Trans['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='blue', s = 2., label="$T_{Obs}$")
    ax1.scatter(subs_Esoil.index, subs_Esoil['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='red', s = 2., label="$Es_{Obs}$")
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # this order of the setting can affect plot x & y axis
    plt.setp(ax1.get_xticklabels(), visible=True)
    ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    ax1.set_ylabel("$T, Es$(mm d$^{-1}$)")
    ax1.axis('tight')
    ax1.set_ylim(0.,4.5)
    ax1.set_xlim(367,1097)
    ax1.legend(loc='best', frameon=False)

    #ax1.update_ticks()

    ax2.plot(tdr.index, tdr.values,    c="orange", lw=1.0, ls="-", label="$θ_{Obs}$")
    ax2.plot(x, SoilMoist_25cm.values, c="green", lw=1.0, ls="-", label="$θ_{CABLE}$")
    ax2.plot(x, swilt,                 c="black", lw=1.0, ls="-", label="$θ_{w}$")
    ax2.plot(x, sfc,                   c="black", lw=1.0, ls="-.", label="$θ_{fc}$")
    ax2.plot(x, ssat,                  c="black", lw=1.0, ls=":", label="$θ_{sat}$")
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    ax2.set_ylabel("$θ$ in 0.25m (m$^{3}$ m$^{-3}$)")
    ax2.axis('tight') # it should be in front of ylim  and xlim
    ax2.set_ylim(0,0.5)
    ax2.set_xlim(367,2922)
    ax2.legend(loc='best', frameon=False)
    #ax2.update_ticks()

    cleaner_dates  = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs     = [1,19,37,52,66,74,86]
    yticks         = [360,260,160,60]
    yticklabels    = ["100","200","300","400"]

    if case_name == "met_LAI_6":

        if contour:
            levels = np.arange(0.,52.,2.)
            img = ax5.contourf(grid_data, cmap=cmap, origin="upper", levels=levels)
            Y_labels = np.flipud(Y)
        else:
            img = ax5.imshow(grid_data, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
            Y_labels = Y

        cbar = fig.colorbar(img, ax = ax5, orientation="vertical", pad=0.02, shrink=.6) #"horizontal"
        cbar.set_label('$θ_{Obs}$ (%)')#('Volumetric soil water content (%)')
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

        ax5.text(0.02, 0.95, '(c)', transform=ax5.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        # every second tick
        ax5.set_yticks(yticks)
        ax5.set_yticklabels(yticklabels)
        plt.setp(ax5.get_xticklabels(), visible=False)

        ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax5.set_ylabel("Depth (cm)")
        ax5.axis('tight')

    if contour:
        levels = np.arange(0.,52.,2.)
        img2 = ax3.contourf(grid_cable, cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels2 = np.flipud(Y)
    else:
        img2 = ax3.imshow(grid_cable, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels2 = Y

    cbar2 = fig.colorbar(img2, ax = ax3,  orientation="vertical", pad=0.02, shrink=.6)
    cbar2.set_label('$θ_{CABLE}$ (%)')#('Volumetric soil water content (%)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()
    if case_name == "met_LAI_6":
        ax3.text(0.02, 0.95, '(d)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    else:
        ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # every second tick
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(yticklabels)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

    cmap = plt.cm.BrBG

    if contour:
        levels = np.arange(-30.,30.,2.)
        img3 = ax4.contourf(difference, cmap=cmap, origin="upper", levels=levels)
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax4.imshow(difference, cmap=cmap, vmin=-30, vmax=30, origin="upper", interpolation='nearest')
        Y_labels3 = Y

    cbar3 = fig.colorbar(img3, ax = ax4, orientation="vertical", pad=0.02, shrink=.6)
    cbar3.set_label('$θ_{CABLE}$ - $θ_{Obs}$ (%)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    if case_name == "met_LAI_6":
        ax4.text(0.02, 0.95, '(e)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    else:
        ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # every second tick
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(yticklabels)

    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("Depth (cm)")
    ax4.axis('tight')

    #plt.show()
    #fig.align_labels()

    if contour == True:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_ET_contour_error_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_ET_error_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)

def plot_profile_tdr_ET_error_rain(fpath, case_name, ring, contour, layer):

    """
    plot simulation status and fluxes
    """

    # ========================= ET FLUX  ============================

    # ===== CABLE =====
    cable = nc.Dataset("%s/EucFACE_amb_out.nc" % fpath, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

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

    T  = np.zeros([3,len(TVeg)])
    Es = np.zeros([3,len(ESoil)])

    T[0,:]  =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'TVeg')['cable'].values
    Es[0,:] =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'ESoil')['cable'].values
    T[1,:]  =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'TVeg')['cable'].values
    Es[1,:] =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'ESoil')['cable'].values
    T[2,:]  =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'TVeg')['cable'].values
    Es[2,:] =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'ESoil')['cable'].values

    TVeg['min']  = T.min(axis=0)
    TVeg['max']  = T.max(axis=0)
    ESoil['min'] = Es.min(axis=0)
    ESoil['max'] = Es.max(axis=0)

    # ===== Obs   =====
    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)

    Es_R2 = read_obs_esoil('R2')['obs']
    Es_R3 = read_obs_esoil('R3')['obs']
    Es_R6 = read_obs_esoil('R6')['obs']

    T_R2 = read_obs_trans('R2')['obs']
    T_R3 = read_obs_trans('R3')['obs']
    T_R6 = read_obs_trans('R6')['obs']

    T_error = np.zeros([3,len(TVeg)])
    Es_error = np.zeros([3,len(TVeg)])

    for date in TVeg.index:
        if np.any(Es_R2.index == date):
            Es_error[0,date-367] = Es_R2[Es_R2.index == date].values
        else:
            Es_error[0,date-367] = float('NaN')
        if np.any(Es_R3.index == date):
            Es_error[1,date-367] = Es_R3[Es_R3.index == date].values
        else:
            Es_error[1,date-367] = float('NaN')
        if np.any(Es_R6.index == date):
            Es_error[2,date-367] = Es_R6[Es_R6.index == date].values
        else:
            Es_error[2,date-367] = float('NaN')

        if np.any(T_R2.index == date):
            T_error[0,date-367] = T_R2[T_R2.index == date].values
        else:
            T_error[0,date-367] = float('NaN')
        if np.any(T_R3.index == date):
            T_error[1,date-367] = T_R3[T_R3.index == date].values
        else:
            T_error[1,date-367] = float('NaN')
        if np.any(T_R6.index == date):
            T_error[2,date-367] = T_R6[T_R6.index == date].values
        else:
            T_error[2,date-367] = float('NaN')

    ESoil['obs_min'] = Es_error.min(axis=0)
    ESoil['obs_max'] = Es_error.max(axis=0)
    TVeg['obs_min']  = T_error.min(axis=0)
    TVeg['obs_max']  = T_error.max(axis=0)

    # ========================= SM IN TOP 25cm ==========================
    SoilMoist_25cm = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    if layer == "6":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                      + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                      + cable.variables['SoilMoist'][:,3,0,0]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.10 )/0.25

    SoilMoist_25cm['dates'] = Time
    SoilMoist_25cm = SoilMoist_25cm.set_index('dates')
    SoilMoist_25cm = SoilMoist_25cm.resample("D").agg('mean')
    SoilMoist_25cm.index = SoilMoist_25cm.index - pd.datetime(2011,12,31)
    SoilMoist_25cm.index = SoilMoist_25cm.index.days
    SoilMoist_25cm = SoilMoist_25cm.sort_values(by=['dates'])

    # Soil hydraulic param
    swilt = np.zeros(len(TVeg))
    sfc = np.zeros(len(TVeg))
    ssat = np.zeros(len(TVeg))

    if layer == "6":
        swilt[:] = ( cable.variables['swilt'][0]*0.022 + cable.variables['swilt'][1]*0.058 \
                   + cable.variables['swilt'][2]*0.154 + cable.variables['swilt'][3]*(0.25-0.022-0.058-0.154) )/0.25
        sfc[:] = ( cable.variables['sfc'][0]*0.022   + cable.variables['sfc'][1]*0.058 \
                   + cable.variables['sfc'][2]*0.154 + cable.variables['sfc'][3]*(0.25-0.022-0.058-0.154) )/0.25
        ssat[:] = ( cable.variables['ssat'][0]*0.022 + cable.variables['ssat'][1]*0.058 \
                   + cable.variables['ssat'][2]*0.154+ cable.variables['ssat'][3]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        swilt[:] =(cable.variables['swilt'][0]*0.15 + cable.variables['swilt'][1]*0.10 )/0.25
        sfc[:] =(cable.variables['sfc'][0]*0.15 + cable.variables['sfc'][1]*0.10 )/0.25
        ssat[:] =(cable.variables['ssat'][0]*0.15 + cable.variables['ssat'][1]*0.10 )/0.25

    # ========================= SM PROFILE ==========================
    ### 1. Read data
    # ==== SM Obs ====
    neo = read_obs_swc_neo(ring)
    tdr = read_obs_swc_tdr(ring)

    # ===== CABLE =====
    SoilMoist = read_cable_SM_one_clmn("%s/EucFACE_amb_out.nc" % fpath, layer)

    ### 2. interpolate SM
    # === Obs SoilMoist ===
    x     = np.concatenate((neo[(25)].index.values,               \
                            neo.index.get_level_values(1).values, \
                            neo[(450)].index.values ))
    y     = np.concatenate(([0]*len(neo[(25)]),                  \
                            neo.index.get_level_values(0).values, \
                            [460]*len(neo[(25)])    ))
    value =  np.concatenate((neo[(25)].values, neo.values, neo[(450)].values))

    X     = neo[(25)].index.values[20:]
    Y     = np.arange(0,461,1)

    grid_X, grid_Y = np.meshgrid(X,Y)

    # === CABLE SoilMoist ===
    if contour:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    else:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='linear') # 'linear' 'nearest'
    #print(type(grid_data))

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    #print(dates)

    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    value_cable = value_cable*100.

    # add the 12 depths to 0
    grid_X_cable, grid_Y_cable = np.meshgrid(X,Y)

    if contour:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='cubic')
    else:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='linear')
    difference = grid_cable -grid_data


    # ======================= PLOTTING  ==========================

    if case_name == "met_LAI_6":
        fig = plt.figure(figsize=[9,17.5])
    else:
        fig = plt.figure(figsize=[9,14])

    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.05)
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #fm = matplotlib.font_manager.json_load(os.path.expanduser("~/.cache/matplotlib/fontlist-v310.json"))
    #fm.findfont('sans-serif', rebuild_if_missing=False)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    if case_name == "met_LAI_6":
        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512)
        ax5 = fig.add_subplot(513)
        ax3 = fig.add_subplot(514)
        ax4 = fig.add_subplot(515)
    else:
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

    x = TVeg.index

    # set x-axis values
    cleaner_dates1 = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs1    = [367,732,1097,1462,1828,2193,2558]
    # set color
    cmap = plt.cm.viridis_r

    ax1.fill_between(x, TVeg['min'].rolling(window=3).mean(),
        TVeg['max'].rolling(window=3).mean(), color="green", alpha=0.2)
    ax1.fill_between(x, ESoil['min'].rolling(window=3).mean(),
        ESoil['max'].rolling(window=3).mean(), color="orange", alpha=0.2)

    ax1.plot(x, TVeg['TVeg'].rolling(window=3).mean(),
        c="green", lw=1.0, ls="-", label="$E_{tr}$ (CABLE)") #.rolling(window=7).mean()
    ax1.plot(x, ESoil['ESoil'].rolling(window=3).mean(),
        c="orange", lw=1.0, ls="-", label="$E_{s}$ (CABLE)") #.rolling(window=7).mean()

    ax1.fill_between(x, TVeg['obs_min'].rolling(window=3).mean(),
        TVeg['obs_max'].rolling(window=3).mean(), color="blue", alpha=0.2)
    ax1.fill_between(x, ESoil['obs_min'].rolling(window=3).mean(),
        ESoil['obs_max'].rolling(window=3).mean(), color="red", alpha=0.2)
    ax1.scatter(subs_Trans.index, subs_Trans['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='blue', s = 2., label="$E_{tr}$ (Obs)")
    ax1.scatter(subs_Esoil.index, subs_Esoil['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='red', s = 2., label="$E_{s}$ (Obs)")
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    if case_name == "met_LAI_6":

        # this order of the setting can affect plot x & y axis
        plt.setp(ax1.get_xticklabels(), visible=True)
        ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
        ax1.set_ylabel("$E_{tr}$, $E_{s}$ (mm d$^{-1}$)")
        ax1.axis('tight')
        ax1.set_ylim(0.,6.)
        ax1.set_xlim(367,1097)
        ax1.legend(loc='upper right', ncol=2, labelspacing=0.2, columnspacing=0.2, frameon=False)
        #ax1.update_ticks()

        ax6  = ax1.twinx()
        ax6.set_ylabel('$P$ (mm d$^{-1}$)')
        ax6.bar(x, -Rainf['Rainf'],  1., color='gray', alpha = 0.5, label='Rainfall') # 'royalblue'
        ax6.set_ylim(-220.,0)
        ax6.set_xlim(367,1097)
        y_ticks      = [-200,-150,-100,-50,0.]
        y_ticklabels = ['200','150','100','50','0']
        ax6.set_yticks(y_ticks)
        ax6.set_yticklabels(y_ticklabels)
        ax6.get_xaxis().set_visible(False)
    else:
        # this order of the setting can affect plot x & y axis
        plt.setp(ax1.get_xticklabels(), visible=True)
        ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
        ax1.set_ylabel("$E_{tr}$, $E_{s}$ (mm d$^{-1}$)")
        ax1.axis('tight')
        ax1.set_ylim(0.,4.)
        ax1.set_xlim(367,1097)
        ax1.legend(loc='upper right', ncol=2,labelspacing=0.2, columnspacing=0.2, frameon=False)
        #ax1.update_ticks()


    ax2.plot(tdr.index, tdr.values,    c="orange", lw=1.0, ls="-", label="$θ$ (Obs)")
    ax2.plot(x, SoilMoist_25cm.values, c="green", lw=1.0, ls="-", label="$θ$ (CABLE)")
    ax2.plot(x, swilt,                 c="black", lw=1.0, ls="-", label="$θ_{w}$")
    ax2.plot(x, sfc,                   c="black", lw=1.0, ls="-.", label="$θ_{fc}$")
    ax2.plot(x, ssat,                  c="black", lw=1.0, ls=":", label="$θ_{sat}$")
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    ax2.set_ylabel("$θ$ in 0.25m (m$^{3}$ m$^{-3}$)")
    ax2.axis('tight') # it should be in front of ylim  and xlim
    ax2.set_ylim(0,0.5)
    ax2.set_xlim(367,2922)
    ax2.legend(loc='upper right', ncol=2, labelspacing=0.2, columnspacing=0.2, frameon=False)
    #ax2.update_ticks()

    cleaner_dates  = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs     = [1,19,37,52,66,74,86]
    yticks         = [360,260,160,60]
    yticklabels    = ["100","200","300","400"]

    if case_name == "met_LAI_6":

        if contour:
            levels = np.arange(0.,0.52,0.02)
            img = ax5.contourf(grid_data/100., cmap=cmap, origin="upper", levels=levels)
            Y_labels = np.flipud(Y)
        else:
            img = ax5.imshow(grid_data/100., cmap=cmap, vmin=0, vmax=0.52, origin="upper", interpolation='nearest')
            Y_labels = Y

        cbar = fig.colorbar(img, ax = ax5, orientation="vertical", pad=0.02, shrink=.6) #"horizontal"
        cbar.set_label('$θ$ Obs (m$^{3}$ m$^{-3}$)')#('Volumetric soil water content (%)')
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

        ax5.text(0.02, 0.95, '(c)', transform=ax5.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        # every second tick
        ax5.set_yticks(yticks)
        ax5.set_yticklabels(yticklabels)
        plt.setp(ax5.get_xticklabels(), visible=False)

        ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax5.set_ylabel("Depth (cm)")
        ax5.axis('tight')

    if contour:
        levels = np.arange(0.,0.52,0.02)
        img2 = ax3.contourf(grid_cable/100., cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels2 = np.flipud(Y)
    else:
        img2 = ax3.imshow(grid_cable/100., cmap=cmap, vmin=0., vmax=0.52, origin="upper", interpolation='nearest')
        Y_labels2 = Y

    cbar2 = fig.colorbar(img2, ax = ax3,  orientation="vertical", pad=0.02, shrink=.6)
    cbar2.set_label('$θ$ CABLE (m$^{3}$ m$^{-3}$)')#('Volumetric soil water content (%)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()
    if case_name == "met_LAI_6":
        ax3.text(0.02, 0.95, '(d)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    else:
        ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # every second tick
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(yticklabels)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

    cmap = plt.cm.BrBG

    if contour:
        levels = np.arange(-0.30,0.30,0.02)
        img3 = ax4.contourf(difference/100., cmap=cmap, origin="upper", levels=levels)
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax4.imshow(difference/100., cmap=cmap, vmin=-0.30, vmax=0.30, origin="upper", interpolation='nearest')
        Y_labels3 = Y

    cbar3 = fig.colorbar(img3, ax = ax4, orientation="vertical", pad=0.02, shrink=.6)
    cbar3.set_label('$θ$(CABLE) - $θ$(Obs) (m$^{3}$ m$^{-3}$)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    if case_name == "met_LAI_6" or case_name == "met_LAI_non-site-param_6":
        ax4.text(0.02, 0.95, '(e)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    else:
        ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # every second tick
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(yticklabels)

    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("Depth (cm)")
    ax4.axis('tight')

    #plt.show()
    #fig.align_labels()

    if contour == True:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_ET_contour_error_rain_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_ET_error_rain_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)

def plot_profile_ET_error_rain(fpath, case_name, ring, contour, layer):

    """
    plot simulation status and fluxes
    """

    # ========================= ET FLUX  ============================

    # ===== CABLE =====
    cable = nc.Dataset("%s/EucFACE_amb_out.nc" % fpath, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

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

    T  = np.zeros([3,len(TVeg)])
    Es = np.zeros([3,len(ESoil)])

    T[0,:]  =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'TVeg')['cable'].values
    Es[0,:] =  read_cable_var("%s/EucFACE_R2_out.nc" % fpath, 'ESoil')['cable'].values
    T[1,:]  =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'TVeg')['cable'].values
    Es[1,:] =  read_cable_var("%s/EucFACE_R3_out.nc" % fpath, 'ESoil')['cable'].values
    T[2,:]  =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'TVeg')['cable'].values
    Es[2,:] =  read_cable_var("%s/EucFACE_R6_out.nc" % fpath, 'ESoil')['cable'].values

    TVeg['min']  = T.min(axis=0)
    TVeg['max']  = T.max(axis=0)
    ESoil['min'] = Es.min(axis=0)
    ESoil['max'] = Es.max(axis=0)

    # ===== Obs   =====
    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)

    Es_R2 = read_obs_esoil('R2')['obs']
    Es_R3 = read_obs_esoil('R3')['obs']
    Es_R6 = read_obs_esoil('R6')['obs']

    T_R2 = read_obs_trans('R2')['obs']
    T_R3 = read_obs_trans('R3')['obs']
    T_R6 = read_obs_trans('R6')['obs']

    T_error = np.zeros([3,len(TVeg)])
    Es_error = np.zeros([3,len(TVeg)])

    for date in TVeg.index:
        if np.any(Es_R2.index == date):
            Es_error[0,date-367] = Es_R2[Es_R2.index == date].values
        else:
            Es_error[0,date-367] = float('NaN')
        if np.any(Es_R3.index == date):
            Es_error[1,date-367] = Es_R3[Es_R3.index == date].values
        else:
            Es_error[1,date-367] = float('NaN')
        if np.any(Es_R6.index == date):
            Es_error[2,date-367] = Es_R6[Es_R6.index == date].values
        else:
            Es_error[2,date-367] = float('NaN')

        if np.any(T_R2.index == date):
            T_error[0,date-367] = T_R2[T_R2.index == date].values
        else:
            T_error[0,date-367] = float('NaN')
        if np.any(T_R3.index == date):
            T_error[1,date-367] = T_R3[T_R3.index == date].values
        else:
            T_error[1,date-367] = float('NaN')
        if np.any(T_R6.index == date):
            T_error[2,date-367] = T_R6[T_R6.index == date].values
        else:
            T_error[2,date-367] = float('NaN')

    ESoil['obs_min'] = Es_error.min(axis=0)
    ESoil['obs_max'] = Es_error.max(axis=0)
    TVeg['obs_min']  = T_error.min(axis=0)
    TVeg['obs_max']  = T_error.max(axis=0)

    # ========================= SM IN TOP 25cm ==========================
    SoilMoist_25cm = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    if layer == "6":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                      + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                      + cable.variables['SoilMoist'][:,3,0,0]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        SoilMoist_25cm['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.10 )/0.25

    SoilMoist_25cm['dates'] = Time
    SoilMoist_25cm = SoilMoist_25cm.set_index('dates')
    SoilMoist_25cm = SoilMoist_25cm.resample("D").agg('mean')
    SoilMoist_25cm.index = SoilMoist_25cm.index - pd.datetime(2011,12,31)
    SoilMoist_25cm.index = SoilMoist_25cm.index.days
    SoilMoist_25cm = SoilMoist_25cm.sort_values(by=['dates'])

    # Soil hydraulic param
    swilt = np.zeros(len(TVeg))
    sfc = np.zeros(len(TVeg))
    ssat = np.zeros(len(TVeg))

    if layer == "6":
        swilt[:] = ( cable.variables['swilt'][0]*0.022 + cable.variables['swilt'][1]*0.058 \
                   + cable.variables['swilt'][2]*0.154 + cable.variables['swilt'][3]*(0.25-0.022-0.058-0.154) )/0.25
        sfc[:] = ( cable.variables['sfc'][0]*0.022   + cable.variables['sfc'][1]*0.058 \
                   + cable.variables['sfc'][2]*0.154 + cable.variables['sfc'][3]*(0.25-0.022-0.058-0.154) )/0.25
        ssat[:] = ( cable.variables['ssat'][0]*0.022 + cable.variables['ssat'][1]*0.058 \
                   + cable.variables['ssat'][2]*0.154+ cable.variables['ssat'][3]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        swilt[:] =(cable.variables['swilt'][0]*0.15 + cable.variables['swilt'][1]*0.10 )/0.25
        sfc[:] =(cable.variables['sfc'][0]*0.15 + cable.variables['sfc'][1]*0.10 )/0.25
        ssat[:] =(cable.variables['ssat'][0]*0.15 + cable.variables['ssat'][1]*0.10 )/0.25

    # ========================= SM PROFILE ==========================
    ### 1. Read data
    # ==== SM Obs ====
    neo = read_obs_swc_neo(ring)
    tdr = read_obs_swc_tdr(ring)

    # ===== CABLE =====
    SoilMoist = read_cable_SM_one_clmn("%s/EucFACE_amb_out.nc" % fpath, layer)

    ### 2. interpolate SM
    # === Obs SoilMoist ===
    x     = np.concatenate((neo[(25)].index.values,               \
                            neo.index.get_level_values(1).values, \
                            neo[(450)].index.values ))
    y     = np.concatenate(([0]*len(neo[(25)]),                  \
                            neo.index.get_level_values(0).values, \
                            [460]*len(neo[(25)])    ))
    value =  np.concatenate((neo[(25)].values, neo.values, neo[(450)].values))

    X     = neo[(25)].index.values[20:]
    Y     = np.arange(0,461,1)

    grid_X, grid_Y = np.meshgrid(X,Y)

    # === CABLE SoilMoist ===
    if contour:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    else:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='linear') # 'linear' 'nearest'
    #print(type(grid_data))

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    #print(dates)

    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    value_cable = value_cable*100.

    # add the 12 depths to 0
    grid_X_cable, grid_Y_cable = np.meshgrid(X,Y)

    if contour:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='cubic')
    else:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='linear')
    difference = grid_cable -grid_data


    # ======================= PLOTTING  ==========================

    # if case_name == "met_LAI_6":
    #     fig = plt.figure(figsize=[9,17.5])
    # else:
    #     fig = plt.figure(figsize=[9,14])
    fig = plt.figure(figsize=[9,14])

    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.05)
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #fm = matplotlib.font_manager.json_load(os.path.expanduser("~/.cache/matplotlib/fontlist-v310.json"))
    #fm.findfont('sans-serif', rebuild_if_missing=False)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # if case_name == "met_LAI_6":
    #     ax1 = fig.add_subplot(511)
    #     ax2 = fig.add_subplot(512)
    #     ax5 = fig.add_subplot(513)
    #     ax3 = fig.add_subplot(514)
    #     ax4 = fig.add_subplot(515)
    # else:
    #     ax1 = fig.add_subplot(411)
    #     ax2 = fig.add_subplot(412)
    #     ax3 = fig.add_subplot(413)
    #     ax4 = fig.add_subplot(414)

    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    x = TVeg.index

    # set x-axis values
    cleaner_dates1 = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs1    = [367,732,1097,1462,1828,2193,2558]
    # set color
    cmap = plt.cm.viridis_r

    ax1.fill_between(x, ESoil['min'].rolling(window=3).mean(),
        ESoil['max'].rolling(window=3).mean(), color="orange", alpha=0.2)
    ax1.plot(x, ESoil['ESoil'].rolling(window=3).mean(),
        c="orange", lw=1.0, ls="-", label="$E_{s}$ (CABLE)") #.rolling(window=7).mean()
    ax1.fill_between(x, ESoil['obs_min'].rolling(window=3).mean(),
        ESoil['obs_max'].rolling(window=3).mean(), color="red", alpha=0.2)
    ax1.scatter(subs_Esoil.index, subs_Esoil['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='red', s = 2., label="$E_{s}$ (Obs)")
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    ax2.fill_between(x, TVeg['min'].rolling(window=3).mean(),
        TVeg['max'].rolling(window=3).mean(), color="green", alpha=0.2)
    ax2.plot(x, TVeg['TVeg'].rolling(window=3).mean(),
        c="green", lw=1.0, ls="-", label="$E_{tr}$ (CABLE)") #.rolling(window=7).mean()
    ax2.fill_between(x, TVeg['obs_min'].rolling(window=3).mean(),
        TVeg['obs_max'].rolling(window=3).mean(), color="blue", alpha=0.2)
    ax2.scatter(subs_Trans.index, subs_Trans['obs'].rolling(window=3).mean(),
        marker='o', c='',edgecolors='blue', s = 2., label="$E_{tr}$ (Obs)")
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # if case_name == "met_LAI_6":
    #
    #     # this order of the setting can affect plot x & y axis
    #     plt.setp(ax1.get_xticklabels(), visible=True)
    #     ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    #     ax1.set_ylabel("$E_{s}$ (mm d$^{-1}$)")
    #     ax1.axis('tight')
    #     ax1.set_ylim(0.,6.)
    #     ax1.set_xlim(367,1097)
    #     ax1.legend(loc='upper right', ncol=2, labelspacing=0.2, columnspacing=0.2, frameon=False)
    #     #ax1.update_ticks()
    #
    #     ax6  = ax1.twinx()
    #     ax6.set_ylabel('$P$ (mm d$^{-1}$)')
    #     ax6.bar(x, -Rainf['Rainf'],  1., color='gray', alpha = 0.5, label='Rainfall') # 'royalblue'
    #     ax6.set_ylim(-220.,0)
    #     ax6.set_xlim(367,1097)
    #     y_ticks      = [-200,-150,-100,-50,0.]
    #     y_ticklabels = ['200','150','100','50','0']
    #     ax6.set_yticks(y_ticks)
    #     ax6.set_yticklabels(y_ticklabels)
    #     ax6.get_xaxis().set_visible(False)
    #
    # else:
    #     # this order of the setting can affect plot x & y axis
    #     plt.setp(ax1.get_xticklabels(), visible=True)
    #     ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    #     ax1.set_ylabel("$E_{s}$ (mm d$^{-1}$)")
    #     ax1.axis('tight')
    #     ax1.set_ylim(0.,2.)
    #     ax1.set_xlim(367,1097)
    #     ax1.legend(loc='upper right', ncol=2,labelspacing=0.2, columnspacing=0.2, frameon=False)
    #     #ax1.update_ticks()

    # this order of the setting can affect plot x & y axis
    plt.setp(ax1.get_xticklabels(), visible=True)
    ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    ax1.set_ylabel("$E_{s}$ (mm d$^{-1}$)")
    ax1.axis('tight')
    ax1.set_ylim(0.,2.)
    ax1.set_xlim(367,1097)
    ax1.legend(loc='upper right', ncol=2,labelspacing=0.2, columnspacing=0.2, frameon=False)
    #ax1.update_ticks()


    # this order of the setting can affect plot x & y axis
    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    ax2.set_ylabel("$E_{tr}$ (mm d$^{-1}$)")
    ax2.axis('tight')
    ax2.set_ylim(0.,3.5)
    ax2.set_xlim(367,1097)
    ax2.legend(loc='upper right', ncol=2,labelspacing=0.2, columnspacing=0.2, frameon=False)
    #ax2.update_ticks()

    # ax2.plot(tdr.index, tdr.values,    c="orange", lw=1.0, ls="-", label="$θ$ (Obs)")
    # ax2.plot(x, SoilMoist_25cm.values, c="green", lw=1.0, ls="-", label="$θ$ (CABLE)")
    # ax2.plot(x, swilt,                 c="black", lw=1.0, ls="-", label="$θ_{w}$")
    # ax2.plot(x, sfc,                   c="black", lw=1.0, ls="-.", label="$θ_{fc}$")
    # ax2.plot(x, ssat,                  c="black", lw=1.0, ls=":", label="$θ_{sat}$")
    # ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    #
    # plt.setp(ax2.get_xticklabels(), visible=True)
    # ax2.set(xticks=xtickslocs1, xticklabels=cleaner_dates1) ####
    # ax2.set_ylabel("$θ$ in 0.25m (m$^{3}$ m$^{-3}$)")
    # ax2.axis('tight') # it should be in front of ylim  and xlim
    # ax2.set_ylim(0,0.5)
    # ax2.set_xlim(367,2922)
    # ax2.legend(loc='upper right', ncol=2, labelspacing=0.2, columnspacing=0.2, frameon=False)
    # #ax2.update_ticks()

    cleaner_dates  = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs     = [1,19,37,52,66,74,86]
    yticks         = [360,260,160,60]
    yticklabels    = ["100","200","300","400"]

    # if case_name == "met_LAI_6":
    #
    #     if contour:
    #         levels = np.arange(0.,0.52,0.02)
    #         img = ax5.contourf(grid_data/100., cmap=cmap, origin="upper", levels=levels)
    #         Y_labels = np.flipud(Y)
    #     else:
    #         img = ax5.imshow(grid_data/100., cmap=cmap, vmin=0, vmax=0.52, origin="upper", interpolation='nearest')
    #         Y_labels = Y
    #
    #     cbar = fig.colorbar(img, ax = ax5, orientation="vertical", pad=0.02, shrink=.6) #"horizontal"
    #     cbar.set_label('Obs (m$^{3}$ m$^{-3}$)')#('Volumetric soil water content (%)')
    #     tick_locator = ticker.MaxNLocator(nbins=5)
    #     cbar.locator = tick_locator
    #     cbar.update_ticks()
    #
    #     ax5.text(0.02, 0.95, '(c)', transform=ax5.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    #
    #     # every second tick
    #     ax5.set_yticks(yticks)
    #     ax5.set_yticklabels(yticklabels)
    #     plt.setp(ax5.get_xticklabels(), visible=False)
    #
    #     ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    #     ax5.set_ylabel("Depth (cm)")
    #     ax5.axis('tight')

    if contour:
        levels = np.arange(0.,0.52,0.02)
        img2 = ax3.contourf(grid_cable/100., cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels2 = np.flipud(Y)
    else:
        img2 = ax3.imshow(grid_cable/100., cmap=cmap, vmin=0., vmax=0.52, origin="upper", interpolation='nearest')
        Y_labels2 = Y

    cbar2 = fig.colorbar(img2, ax = ax3,  orientation="vertical", pad=0.02, shrink=.6)
    cbar2.set_label('$θ$ CABLE (m$^{3}$ m$^{-3}$)')#('Volumetric soil water content (%)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()
    # if case_name == "met_LAI_6":
    #     ax3.text(0.02, 0.95, '(d)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # else:
    #     ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # every second tick
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(yticklabels)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

    cmap = plt.cm.BrBG

    if contour:
        levels = np.arange(-0.30,0.30,0.02)
        img3 = ax4.contourf(difference/100., cmap=cmap, origin="upper", levels=levels)
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax4.imshow(difference/100., cmap=cmap, vmin=-0.30, vmax=0.30, origin="upper", interpolation='nearest')
        Y_labels3 = Y

    cbar3 = fig.colorbar(img3, ax = ax4, orientation="vertical", pad=0.02, shrink=.6)
    cbar3.set_label('$θ$(CABLE) - $θ$(Obs) (m$^{3}$ m$^{-3}$)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    # if case_name == "met_LAI_6" or case_name == "met_LAI_non-site-param_6":
    #     ax4.text(0.02, 0.95, '(e)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # else:
    #     ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # every second tick
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(yticklabels)

    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("Depth (cm)")
    ax4.axis('tight')

    #plt.show()
    #fig.align_labels()

    if contour == True:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_ET_contour_error_rain_no-tdr_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("../plots/EucFACE_SW_obsved_dates_ET_error_rain_no-tdr_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)

def plot_dry_down(fcable, case_name, ring, layer):

    subset = read_obs_swc_neo(ring)

# ___________________ From Pandas to Numpy __________________________
#    date_start = pd.datetime(2013,1,1) - pd.datetime(2011,12,31)
#    date_end   = pd.datetime(2017,1,1) - pd.datetime(2011,12,31)
#    date_start = pd.datetime(2012,4,30) - pd.datetime(2011,12,31)
#    date_end   = pd.datetime(2019,5,11) - pd.datetime(2011,12,31)
#    date_start = date_start.days
#    date_end   = date_end.days

    # Interpolate
    x     = subset.index.get_level_values(1).values
    y     = subset.index.get_level_values(0).values
    value = subset.values

    print(subset[(25)].index.values)
    # add the 12 depths to 0
    X     = subset[(25)].index.values[20:]
    Y     = np.arange(0,465,5)

    grid_X, grid_Y = np.meshgrid(X,Y)
    print(grid_X.shape)
    # interpolate
    grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='nearest')
    print(grid_data.shape)

    top_obs = np.mean(grid_data[0:7,:],axis=0)/100.#*5.
    mid_obs = np.mean(grid_data[7:31,:],axis=0)/100.#*5.
    bot_obs = np.mean(grid_data[31:,:],axis=0)/100.#*5.
    print(top_obs)
    print(mid_obs)
    print(bot_obs)

# _________________________ CABLE ___________________________
    SoilMoist = read_cable_SM(fcable, layer)

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    x_cable     = SoilMoist['dates'].values
    y_cable     = SoilMoist['Depth'].values
    value_cable = SoilMoist.iloc[:,2].values

    # add the 12 depths to 0
    X_cable     = X #np.arange(date_start_cable,date_end_cable,1) # 2013-1-1 to 2016-12-31
    Y_cable     = np.arange(0,465,5)
    grid_X_cable, grid_Y_cable = np.meshgrid(X_cable,Y_cable)

    grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='nearest')

    top_cable = np.mean(grid_cable[0:7,:],axis=0)
    mid_cable = np.mean(grid_cable[7:31,:],axis=0)
    bot_cable = np.mean(grid_cable[31:,:],axis=0)
    print(top_cable)
    print(mid_cable)
    print(bot_cable)

# _________________________ Plotting ___________________________
    fig = plt.figure(figsize=[12,12])
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

    ax1.plot(top_obs,     c="orange", lw=1.0, ls="-", label="Obs")
    ax1.plot(top_cable,   c="green", lw=1.0, ls="-", label="CABLE")
    ax2.plot(mid_obs,     c="orange", lw=1.0, ls="-", label="Obs")
    ax2.plot(mid_cable,   c="green", lw=1.0, ls="-", label="CABLE")
    ax3.plot(bot_obs,     c="orange", lw=1.0, ls="-", label="Obs")
    ax3.plot(bot_cable,   c="green", lw=1.0, ls="-", label="CABLE")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [0,19,37,52,66,74,86]

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.set_ylabel("0 - 30cm")
    ax1.axis('tight')
    ax1.set_ylim(0,0.5)
    ax1.legend()

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax2.set_ylabel("30cm - 1.5m")
    ax2.axis('tight')
    ax2.set_ylim(0,0.5)

    plt.setp(ax3.get_xticklabels(), visible=True)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax3.set_ylabel("1.5 - 4.65m")
    ax3.axis('tight')
    ax3.set_ylim(0,0.5)

    fig.savefig("EucFACE_SW_top-mid-bot_%s_%s.png" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)
