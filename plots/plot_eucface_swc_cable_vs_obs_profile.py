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
from plot_eucface_get_var import *

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

    #print(subset[(25)].index.values)
    # add the 12 depths to 0
    X     = subset[(25)].index.values[20:] #np.arange(date_start,date_end,1) # 2012-4-30 to 2019-5-11
    Y     = np.arange(0,465,5)

    grid_X, grid_Y = np.meshgrid(X,Y)
    #print(grid_X.shape)
    # interpolate
    if contour:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    else:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='nearest')
    print(type(grid_data))

# ____________________ Plot obs _______________________
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

    ax1 = fig.add_subplot(311) #(nrows=2, ncols=2, index=1)

    cmap = plt.cm.viridis_r

    if contour:
        levels = np.arange(0.,52.,2.)
        img = ax1.contourf(grid_data, cmap=cmap, origin="upper", levels=levels)
        Y_labels = np.flipud(Y)
    else:
        img = ax1.imshow(grid_data, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels = Y

    cbar = fig.colorbar(img, orientation="vertical", pad=0.02, shrink=.6) #"horizontal"
    cbar.set_label('VWC Obs (%)')#('Volumetric soil water content (%)')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # every second tick
    ax1.set_yticks(np.arange(len(Y))[::10])
    ax1.set_yticklabels(Y_labels[::10])
    plt.setp(ax1.get_xticklabels(), visible=False)

    #ax1.set_xticks(np.arange(len(X)))
    #cleaner_dates = X
    #ax1.set_xticklabels(cleaner_dates)

    #datemark = np.arange(np.datetime64('2013-01-01','D'), np.datetime64('2017-01-01','D'))
    #xtickslocs = ax1.get_xticks()

    #for i in range(len(datemark)):
    #    print(i, datemark[i]) # xtickslocs[i]

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

    cbar2 = fig.colorbar(img2, orientation="vertical", pad=0.02, shrink=.6)
    cbar2.set_label('VWC CABLE (%)')#('Volumetric soil water content (%)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()

    # every second tick
    ax2.set_yticks(np.arange(len(Y_cable))[::10])
    ax2.set_yticklabels(Y_labels2[::10])
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


    cbar3 = fig.colorbar(img3, orientation="vertical", pad=0.02, shrink=.6)
    cbar3.set_label('CABLE - Obs (%)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    # every second tick
    ax3.set_yticks(np.arange(len(Y_cable))[::10])
    ax3.set_yticklabels(Y_labels3[::10])

    ax3.set_xticks(np.arange(len(X_cable)))
    cleaner_dates3 = X_cable
    ax3.set_xticklabels(cleaner_dates3)


    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')
    if contour == True:
        fig.savefig("EucFACE_SW_obsved_dates_contour_%s_%s.png" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("EucFACE_SW_obsved_dates_%s_%s.png" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)


def plot_profile_tdr_ET(fcable, case_name, ring, contour, layer):
    """
    plot simulation status and fluxes
    """

# ____________________ Plot setting _______________________
    fig = plt.figure(figsize=[14,12])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
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

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [0,19,37,52,66,74,86]


# ____________________ Read data _______________________
    subset = read_obs_swc_tdr(ring)



    subset = read_obs_swc_neo(ring)
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
        y     = subset.index.get_level_values(0).values#, \
        value = subset.values

    X     = subset[(25)].index.values[20:] #np.arange(date_start,date_end,1) # 2012-4-30 to 2019-5-11
    Y     = np.arange(0,465,5)

    grid_X, grid_Y = np.meshgrid(X,Y)

    if contour:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    else:
        grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='nearest')
    print(type(grid_data))



    cmap = plt.cm.viridis_r


    '''
    ax1 = fig.add_subplot(412) #(nrows=2, ncols=2, index=1)

    if contour:
        levels = np.arange(0.,52.,2.)
        img = ax1.contourf(grid_data, cmap=cmap, origin="upper", levels=levels)
        Y_labels = np.flipud(Y)
    else:
        img = ax1.imshow(grid_data, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels = Y

    cbar = fig.colorbar(img, orientation="vertical", pad=0.02, shrink=.6) #"horizontal"
    cbar.set_label('VWC Obs (%)')#('Volumetric soil water content (%)')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # every second tick
    ax1.set_yticks(np.arange(len(Y))[::10])
    ax1.set_yticklabels(Y_labels[::10])
    plt.setp(ax1.get_xticklabels(), visible=False)


    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_ylabel("Depth (cm)")
    ax1.axis('tight')
    '''
# _________________________ CABLE ___________________________
    SoilMoist = read_cable_SM(fcable, layer)

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
        y_cable     = SoilMoist['Depth'].values
        value_cable = SoilMoist.iloc[:,2].values
    value_cable = value_cable*100.
    # add the 12 depths to 0
    X_cable     = X #np.arange(date_start_cable,date_end_cable,1) # 2013-1-1 to 2016-12-31
    Y_cable     = np.arange(0,465,5)
    grid_X_cable, grid_Y_cable = np.meshgrid(X_cable,Y_cable)

    # interpolate
    if contour:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='cubic')
    else:
        grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='nearest')

    ax2 = fig.add_subplot(312)#, sharey = ax1)#(nrows=2, ncols=2, index=2, sharey=ax1)

    if contour:
        img2 = ax2.contourf(grid_cable, cmap=cmap, origin="upper", levels=levels,interpolation='nearest')
        Y_labels2 = np.flipud(Y)
    else:
        img2 = ax2.imshow(grid_cable, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
        Y_labels2 = Y

    cbar2 = fig.colorbar(img2, orientation="vertical", pad=0.02, shrink=.6)
    cbar2.set_label('VWC CABLE (%)')#('Volumetric soil water content (%)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()

    # every second tick
    ax2.set_yticks(np.arange(len(Y_cable))[::10])
    ax2.set_yticklabels(Y_labels2[::10])
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("Depth (cm)")
    ax2.axis('tight')

# ________________ plot difference _____________________
    ax3 = fig.add_subplot(313)#,figsize=(12,4))
    difference = grid_cable -grid_data

    cmap = plt.cm.BrBG

    if contour:
        levels = np.arange(-30.,30.,2.)
        img3 = ax3.contourf(difference, cmap=cmap, origin="upper", levels=levels)
        Y_labels3 = np.flipud(Y)
    else:
        img3 = ax3.imshow(difference, cmap=cmap, vmin=-30, vmax=30, origin="upper", interpolation='nearest')
        Y_labels3 = Y

    cbar3 = fig.colorbar(img3, orientation="vertical", pad=0.02, shrink=.6)
    cbar3.set_label('CABLE - Obs (%)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    # every second tick
    ax3.set_yticks(np.arange(len(Y_cable))[::10])
    ax3.set_yticklabels(Y_labels3[::10])

    ax3.set_xticks(np.arange(len(X_cable)))
    cleaner_dates3 = X_cable
    ax3.set_xticklabels(cleaner_dates3)


    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

# ________________________ ET _________________________

    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)
    subset     = read_obs_swc_tdr(ring)

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

    ax4 = fig.add_subplot(311)#,figsize=(15,15)
    width = 1.
    x = TVeg.index

    ax4.plot(x, TVeg['TVeg'].rolling(window=5).mean(),     c="green", lw=1.0, ls="-", label="Trans") #.rolling(window=7).mean()
    ax4.plot(x, ESoil['ESoil'].rolling(window=5).mean(),    c="orange", lw=1.0, ls="-", label="ESoil") #.rolling(window=7).mean()
    ax4.scatter(subs_Trans.index, subs_Trans['obs'], marker='o', c='',edgecolors='blue', s = 2., label="Trans Obs") # subs['EfloorPred']
    ax4.scatter(subs_Esoil.index, subs_Esoil['obs'], marker='o', c='',edgecolors='red', s = 2., label="ESoil Obs") # subs['EfloorPred']

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax4.set_ylabel(" Trans, Esoil ($mm$ $d^{-1}$)")
    ax4.axis('tight')
    ax4.set_ylim(0.,4.5)
    ax4.set_xlim(367,1097)
    ax4.legend()

    fig.align_labels()

    if contour == True:
        fig.savefig("EucFACE_SW_obsved_dates_ET_contour_%s_%s.png" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig("EucFACE_SW_obsved_dates_ET_%s_%s.png" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)

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



'''
if __name__ == "__main__":

    contour = False
    #  True for contour
    #  False for raster
    layer =  "31uni"
'''
