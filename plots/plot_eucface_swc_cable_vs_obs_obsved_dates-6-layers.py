#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-7-30"
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
    subset_amb = neo[neo['Ring'].isin(['R2','R3','R6'])]
    subset_ele = neo[neo['Ring'].isin(['R1','R4','R5'])]
    subset_R1  = neo[neo['Ring'].isin(['R1'])]
    subset_R2  = neo[neo['Ring'].isin(['R2'])]
    subset_R3  = neo[neo['Ring'].isin(['R3'])]
    subset_R4  = neo[neo['Ring'].isin(['R4'])]
    subset_R5  = neo[neo['Ring'].isin(['R5'])]
    subset_R6  = neo[neo['Ring'].isin(['R6'])]

    # calculate the mean of every group ( and unstack #.unstack(level=0)
    neo_mean = neo.groupby(by=["Depth","Date"]).mean()
    amb_mean = subset_amb.groupby(by=["Depth","Date"]).mean()
    ele_mean = subset_ele.groupby(by=["Depth","Date"]).mean()
    R1_mean  = subset_R1.groupby(by=["Depth","Date"]).mean()
    R2_mean  = subset_R2.groupby(by=["Depth","Date"]).mean()
    R3_mean  = subset_R3.groupby(by=["Depth","Date"]).mean()
    R4_mean  = subset_R4.groupby(by=["Depth","Date"]).mean()
    R5_mean  = subset_R5.groupby(by=["Depth","Date"]).mean()
    R6_mean  = subset_R6.groupby(by=["Depth","Date"]).mean()

    # remove 'VWC'
    neo_mean = neo_mean.xs('VWC', axis=1, drop_level=True)
    amb_mean = amb_mean.xs('VWC', axis=1, drop_level=True)
    ele_mean = ele_mean.xs('VWC', axis=1, drop_level=True)
    R1_mean  = R1_mean.xs('VWC', axis=1, drop_level=True)
    R2_mean  = R2_mean.xs('VWC', axis=1, drop_level=True)
    R3_mean  = R3_mean.xs('VWC', axis=1, drop_level=True)
    R4_mean  = R4_mean.xs('VWC', axis=1, drop_level=True)
    R5_mean  = R5_mean.xs('VWC', axis=1, drop_level=True)
    R6_mean  = R6_mean.xs('VWC', axis=1, drop_level=True)
    # 'VWC' : key on which to get cross section
    # axis=1 : get cross section of column
    # drop_level=True : returns cross section without the multilevel index

    #neo_mean = np.transpose(neo_mean)

    vars = amb_mean

# ___________________ From Pandas to Numpy __________________________
    date_start = pd.datetime(2013,1,1) - pd.datetime(2011,12,31)
    date_end   = pd.datetime(2017,1,1) - pd.datetime(2011,12,31)
#    date_start = pd.datetime(2012,4,30) - pd.datetime(2011,12,31)
#    date_end   = pd.datetime(2019,5,11) - pd.datetime(2011,12,31)
    date_start = date_start.days
    date_end   = date_end.days

    # Interpolate
    x     = np.concatenate((vars[(25)].index.values,               \
                            vars.index.get_level_values(1).values, \
                            vars[(450)].index.values ))              # time
    y     = np.concatenate(([0]*len(vars[(25)]),                   \
                            vars.index.get_level_values(0).values, \
                            [460]*len(vars[(25)])    ))
    value = np.concatenate((vars[(25)].values, vars.values, vars[(450)].values))
    # get_level_values(1) : Return an Index of values for requested level.
    # add Depth = 0 and Depth = 460

    print(vars[(25)].index.values)
    # add the 12 depths to 0
    X     = vars[(25)].index.values #np.arange(date_start,date_end,1) # 2012-4-30 to 2019-5-11
    Y     = np.arange(0,465,5)

    grid_X, grid_Y = np.meshgrid(X,Y)
    print(grid_X.shape)
    # interpolate
    grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='cubic')
    #'cubic')#'linear')#'nearest')
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
    #######
    #plt.imshow(amb_mean, cmap=cmap, vmin=0, vmax=40, origin="upper", interpolation='nearest')
    #plt.show()
    ######
    # img = ax1.imshow(grid_data, cmap=cmap, vmin=0, vmax=40, origin="upper", interpolation='nearest')
    #'spline16')#'nearest')

    levels = np.arange(0.,52.,2.)
    img = ax1.contourf(grid_data, cmap=cmap, origin="upper", levels=levels) # vmin=0, vmax=40,
    cbar = fig.colorbar(img, orientation="vertical", pad=0.1, shrink=.6) #"horizontal"
    cbar.set_label('VWC Obs (%)')#('Volumetric soil water content (%)')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # every second tick
    ax1.set_yticks(np.arange(len(Y))[::10])
    Y_labels = np.flipud(Y) #Y #np.flipud(Y)
    ax1.set_yticklabels(Y_labels[::10])
    plt.setp(ax1.get_xticklabels(), visible=False)

    #ax1.set_xticks(np.arange(len(X)))
    #cleaner_dates = X
    #ax1.set_xticklabels(cleaner_dates)

    #datemark = np.arange(np.datetime64('2013-01-01','D'), np.datetime64('2017-01-01','D'))
    #xtickslocs = ax1.get_xticks()

    for i in range(len(datemark)):
        print(i, datemark[i]) # xtickslocs[i]

    #cleaner_dates = ["2014","2015","2016",]
    #xtickslocs    = [365,730,1095]

    cleaner_dates = ["2012","2013","2014","2015","2016","2017","2018","2019"]
                    #["2012-04","2013-01","2014-01","2015-01","2016-01",\
                    # "2017-03","2018-01","2019-01",]
    xtickslocs = [1,20,39,57,72,86,94,106]

    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.set_ylabel("Depth (cm)")
    ax1.axis('tight')

    plt.show()

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')

    Time = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=[1.1, 5.1, 15.7, 43.85, 118.55, 316.4])
    #
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
    date_start_cable = pd.datetime(2013,1,1) - pd.datetime(2011,12,31)
    date_end_cable   = pd.datetime(2017,1,1) - pd.datetime(2011,12,31)
    date_start_cable = date_start_cable.days
    date_end_cable   = date_end_cable.days

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
                 method='cubic')
                 #'cubic')#'linear')#'nearest')

    ax2 = fig.add_subplot(312)#, sharey = ax1)#(nrows=2, ncols=2, index=2, sharey=ax1)

    #img2 = ax2.imshow(grid_cable, cmap=cmap, vmin=0, vmax=40, origin="upper", interpolation='nearest')
    #'spline16')#'nearest')

    img2 = ax2.contourf(grid_cable, cmap=cmap, origin="upper", levels=levels) #vmin=0, vmax=40,
    cbar2 = fig.colorbar(img2, orientation="vertical", pad=0.1, shrink=.6)
    cbar2.set_label('VWC CABLE (%)')#('Volumetric soil water content (%)')
    tick_locator2 = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator2
    cbar2.update_ticks()

    # every second tick
    ax2.set_yticks(np.arange(len(Y_cable))[::10])
    Y_labels2 = np.flipud(Y) #Y #np.flipud(Y_cable)
    ax2.set_yticklabels(Y_labels2[::10])
    plt.setp(ax2.get_xticklabels(), visible=False)

    #ax2.set_xticks(np.arange(len(X_cable)))
    #cleaner_dates2 = X_cable
    #ax2.set_xticklabels(cleaner_dates2)

    #datemark2 = np.arange(np.datetime64('2013-01-01','D'), np.datetime64('2017-01-01','D'))

    #xtickslocs2 = ax2.get_xticks()
    #for i in range(len(datemark2)):
    #    print(xtickslocs2[i], datemark2[i])

    #cleaner_dates2 = ["2014","2015","2016",]
                  # ["2013-01","2014-01","2015-01","2016-01",]
    #xtickslocs2 = [365,730,1095]

    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("Depth (cm)")
    ax2.axis('tight')

# ________________ plot difference _____________________
    ax3 = fig.add_subplot(313)
    difference = grid_cable -grid_data

    cmap = plt.cm.BrBG

    #img3 = ax3.imshow(difference, cmap=cmap, vmin=-30, vmax=30, origin="upper", interpolation='nearest')
    #'spline16')#'nearest')
    levels = np.arange(-30.,30.,2.)
    img3 = ax3.contourf(difference, cmap=cmap, origin="upper", levels=levels)
    cbar3 = fig.colorbar(img3, orientation="vertical", pad=0.1, shrink=.6)
    cbar3.set_label('CABLE - Obs (%)')
    tick_locator3 = ticker.MaxNLocator(nbins=6)
    cbar3.locator = tick_locator3
    cbar3.update_ticks()

    # every second tick
    ax3.set_yticks(np.arange(len(Y_cable))[::10])
    Y_labels3 = np.flipud(Y_cable) #Y #np.flipud(Y_cable)
    ax3.set_yticklabels(Y_labels3[::10])

    ax3.set_xticks(np.arange(len(X_cable)))
    cleaner_dates3 = X_cable
    ax3.set_xticklabels(cleaner_dates3)

    #cleaner_dates3 = ["2014","2015","2016","2017","2018","2019"]
                  # ["2013-01","2014-01","2015-01","2016-01",]
    #xtickslocs3 = [365,730,1095,1461,1826,2191]

    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_ylabel("Depth (cm)")
    ax3.axis('tight')

    fig.savefig("EucFACE_SW_amb_obsved_dates_contour_6layers_%s_gw_on_or_on.png" % (case_name), bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    case = ["hyds0.01","hyds0.1","hyds","hyds10","hyds100"]
    for case_name in case:
        fobs = "/short/w35/mm3972/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
        fcable ="/g/data/w35/mm3972/cable/EucFACE/EucFACE_run/outputs/hyds_test/6layer_hyds_test/%s/EucFACE_amb_out.nc" % (case_name)

        main(fobs, fcable, case_name)
