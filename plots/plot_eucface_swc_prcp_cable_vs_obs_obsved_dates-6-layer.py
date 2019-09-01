#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-8-26"
__changefrom__ = 'plot_eucface_swc_cable_vs_obs_obsved_dates-6-layer.py'

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

# ______________________________ SWC obs _______________________________
    neo = pd.read_csv(fobs, usecols = ['Ring','Depth','Date','VWC'])
    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)

    datemark   = neo['Date'].unique()
    datemark   = np.sort(datemark)
    print(datemark)
    neo['Date'] = neo['Date'] - pd.datetime(2011,12,31)
    neo['Date'] = neo['Date'].dt.days
    neo = neo.sort_values(by=['Date','Depth'])

    print(neo['Depth'].unique())

    subset = neo[neo['Ring'].isin(['R2','R3','R6'])]
                 #ele : isin(['R1','R4','R5'])]

    subset = subset.groupby(by=["Depth","Date"]).mean()
    subset = subset.xs('VWC', axis=1, drop_level=True)

# ___________________ From Pandas to Numpy __________________________
    date_start = pd.datetime(2013,1,1) - pd.datetime(2011,12,31)
    date_end   = pd.datetime(2017,1,1) - pd.datetime(2011,12,31)
    date_start = date_start.days
    date_end   = date_end.days

    x     = np.concatenate((subset[(25)].index.values,               \
                            subset.index.get_level_values(1).values, \
                            subset[(450)].index.values ))              # time
    y     = np.concatenate(([0]*len(subset[(25)]),                   \
                            subset.index.get_level_values(0).values, \
                            [460]*len(subset[(25)])    ))
    value = np.concatenate((subset[(25)].values, subset.values, subset[(450)].values))

    print(subset[(25)].index.values)
    X     = subset[(25)].index.values #np.arange(date_start,date_end,1) # 2012-4-30 to 2019-5-11
    Y     = np.arange(0,465,5)

    grid_X, grid_Y = np.meshgrid(X,Y)
    print(grid_X.shape)
    # interpolate
    grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    #'cubic')#'linear')#'nearest')
    print(grid_data.shape)

# _____________________________ CABLE-Rainfall ___________________________
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.

    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    Rainf.index = Rainf.index.days
    Rain = np.zeros(len(X))
    print("aaaaaaaaaa")
    #print(Rainf.loc[X[-1]].values)
    for i in np.arange(29,len(Rainf)+1):
        Rainf['Rainf'][i] = Rainf['Rainf'][i-29:i+1].sum()
    #print(Rainf)

    for i in np.arange(0,len(X)):
        if X[i] >=367:
            print(X[i])
            print(Rainf[Rainf.index == X[i]].index)
            Rain[i] = Rainf[Rainf.index == X[i]].values
            print(Rain[i])
    print(Rain)

# ____________________________ CABLE-SoilMoist ___________________________
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=[1.1, 5.1, 15.7, 43.85, 118.55, 316.4])
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
    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    value_cable = value_cable*100.
    # add the 12 depths to 0
    X_cable     = X #np.arange(date_start_cable,date_end_cable,1) # 2013-1-1 to 2016-12-31
    Y_cable     = np.arange(0,465,5)
    grid_X_cable, grid_Y_cable = np.meshgrid(X_cable,Y_cable)

    # interpolate
    grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='cubic') #'cubic')#'linear')#'nearest')

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

    cmap = plt.cm.viridis_r

    ax1   = fig.add_subplot(411)
    x     = np.arange(len(Rain))
    width = 1
    rects = ax1.bar(x, Rain, width, color='royalblue', label='Prcp')

    plt.setp(ax1.get_xticklabels(), visible=False)

    #ax1.set_xticks(np.arange(len(X)))
    #cleaner_dates = X
    #ax1.set_xticklabels(cleaner_dates)

    #datemark = np.arange(np.datetime64('2013-01-01','D'), np.datetime64('2017-01-01','D'))
    #xtickslocs = ax1.get_xticks()

    #for i in range(len(datemark)):
    #    print(i, datemark[i]) # xtickslocs[i]

    cleaner_dates = ["2012","2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [1,20,39,57,72,86,94,106]
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_ylabel("3 days Precipitation (mm)")
    ax1.axis('tight')

    ax2   = fig.add_subplot(412)
    #######
    #plt.imshow(amb_mean, cmap=cmap, vmin=0, vmax=40, origin="upper", interpolation='nearest')
    #plt.show()
    ######
    img = ax2.imshow(grid_data, cmap=cmap, vmin=0, vmax=40, origin="upper", interpolation='nearest')
    #'spline16')#'nearest')

    #levels = np.arange(0.,52.,2.)
    #img = ax2.contourf(grid_data, cmap=cmap, origin="upper", levels=levels) # vmin=0, vmax=40,
    cbar = fig.colorbar(img, orientation="horizontal", pad=0.1, shrink=.6) #"horizontal"
    cbar.set_label('VWC Obs (%)')#('Volumetric soil water content (%)')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # every second tick
    ax2.set_yticks(np.arange(len(Y))[::10])
    Y_labels = Y #np.flipud(Y) #Y #np.flipud(Y)
    ax2.set_yticklabels(Y_labels[::10])
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("Depth (cm)")
    ax2.axis('tight')


    ax3 = fig.add_subplot(413)#, sharey = ax1)#(nrows=2, ncols=2, index=2, sharey=ax1)

    img3 = ax3.imshow(grid_cable, cmap=cmap, vmin=0, vmax=40, origin="upper", interpolation='nearest')
    #'spline16')#'nearest')

    #img3 = ax3.contourf(grid_cable, cmap=cmap, origin="upper", levels=levels) #vmin=0, vmax=40,
    cbar3 = fig.colorbar(img3, orientation="horizontal", pad=0.1, shrink=.6) #"vertical"
    cbar3.set_label('VWC CABLE (%)')#('Volumetric soil water content (%)')
    tick_locator3 = ticker.MaxNLocator(nbins=5)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    # every second tick
    ax3.set_yticks(np.arange(len(Y_cable))[::10])
    Y_labels3 = Y #np.flipud(Y) #Y #np.flipud(Y_cable)
    ax3.set_yticklabels(Y_labels3[::10])
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

# ________________ plot difference _____________________
    ax4 = fig.add_subplot(414)
    difference = grid_cable -grid_data

    cmap = plt.cm.BrBG

    img4 = ax4.imshow(difference, cmap=cmap, vmin=-30, vmax=30, origin="upper", interpolation='nearest')
    #'spline16')#'nearest')
    #levels = np.arange(-30.,30.,2.)
    #img4 = ax4.contourf(difference, cmap=cmap, origin="upper", levels=levels)
    cbar4 = fig.colorbar(img4, orientation="horizontal", pad=0.1, shrink=.6)
    cbar4.set_label('CABLE - Obs (%)')
    tick_locator4 = ticker.MaxNLocator(nbins=6)
    cbar4.locator = tick_locator4
    cbar4.update_ticks()

    # every second tick
    ax4.set_yticks(np.arange(len(Y_cable))[::10])
    Y_labels4 = Y #np.flipud(Y_cable) #Y #np.flipud(Y_cable)
    ax4.set_yticklabels(Y_labels4[::10])

    ax4.set_xticks(np.arange(len(X_cable)))
    cleaner_dates4 = X_cable
    ax4.set_xticklabels(cleaner_dates4)

    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_ylabel("Depth (cm)")
    ax4.axis('tight')

    fig.savefig("EucFACE_Prcp-SW_contour_obsrvd-date_GW_Or_Hvrd_Nzdpt_6l_amb_%s.png" % (case_name), bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    case_name = "hyds"
    fobs = "/short/w35/mm3972/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    fcable ="/g/data/w35/mm3972/cable/EucFACE/EucFACE_run/outputs/hyds_test/6layer_hyds_test/no_zdepth/%s/EucFACE_amb_out.nc" % (case_name)

    main(fobs, fcable, case_name)
