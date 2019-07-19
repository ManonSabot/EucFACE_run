#!/usr/bin/env python

"""
Plot EucFACE soil moisture

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (24.10.2018)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from osgeo import gdal
from gdalconst import *
from scipy import interpolate

#import iris
def main(fobs, fcable):

    neo = pd.read_csv(fobs, usecols = ['Ring','Depth','Date','VWC'])
    # usecols : read specific columns from CSV

    # divide neo into groups
    subset_amb = neo[neo['Ring'].isin(['R2','R3','R6'])]
    subset_ele = neo[neo['Ring'].isin(['R1','R4','R5'])]
    subset_R1  = neo[neo['Ring'].isin(['R1'])]
    subset_R2  = neo[neo['Ring'].isin(['R2'])]
    subset_R3  = neo[neo['Ring'].isin(['R3'])]
    subset_R4  = neo[neo['Ring'].isin(['R4'])]
    subset_R5  = neo[neo['Ring'].isin(['R5'])]
    subset_R6  = neo[neo['Ring'].isin(['R6'])]

    # calculate the mean of every group and unstack
    neo_mean = neo.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    amb_mean = subset_amb.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    ele_mean = subset_ele.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R1_mean  = subset_R1.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R2_mean  = subset_R2.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R3_mean  = subset_R3.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R4_mean  = subset_R4.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R5_mean  = subset_R5.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R6_mean  = subset_R6.groupby(by=["Depth","Date"]).mean().unstack(level=0)

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

    # Converting the index as date
    neo_mean.index = pd.to_datetime(neo_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    amb_mean.index = pd.to_datetime(amb_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    ele_mean.index = pd.to_datetime(ele_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R1_mean.index  = pd.to_datetime(R1_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R2_mean.index  = pd.to_datetime(R2_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R3_mean.index  = pd.to_datetime(R3_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R4_mean.index  = pd.to_datetime(R4_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R5_mean.index  = pd.to_datetime(R5_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R6_mean.index  = pd.to_datetime(R6_mean.index,format="%d/%m/%y",infer_datetime_format=False)

    # Sort by date
    neo_mean = neo_mean.sort_values(by=['Date'])
    amb_mean = amb_mean.sort_values(by=['Date'])
    ele_mean = ele_mean.sort_values(by=['Date'])
    R1_mean = R1_mean.sort_values(by=['Date'])
    R2_mean = R2_mean.sort_values(by=['Date'])
    R3_mean = R3_mean.sort_values(by=['Date'])
    R4_mean = R4_mean.sort_values(by=['Date'])
    R5_mean = R5_mean.sort_values(by=['Date'])
    R6_mean = R6_mean.sort_values(by=['Date'])

    depths = np.unique(neo.Depth)
    ndepths = len(depths)
    dates_org = np.unique(amb_mean.index).astype('datetime64[D]') translate to data
    ntimes_org = len(dates_org)

    neo_mean = np.transpose(neo_mean)
    amb_mean = np.transpose(amb_mean)
    ele_mean = np.transpose(ele_mean)
    R1_mean = np.transpose(R1_mean)
    R2_mean = np.transpose(R2_mean)
    R3_mean = np.transpose(R3_mean)
    R4_mean = np.transpose(R4_mean)
    R5_mean = np.transpose(R5_mean)
    R6_mean = np.transpose(R6_mean)

    # fill the time gap
    dates_new = np.arange('2012-04-30', '2019-05-10', dtype='datetime64[D]')
    ntimes_new = len(dates_new)
    datanew = np.zeros((ndepths, ntimes_new), dtype=np.float64)

    dataxx, datayy = np.meshgrid(dates_org, depths)

    f1 = interpolate.interp2d(dates_org, depths,amb_mean, kind='cubic')
    datanew = f1(dates_new, depths)

    #sample_points = [('time',dates),('depth',depths)]
    #data_interpolate = amb_mean.interpolate(sample_points, iris.analysis.Linear())
    #print(data_interpolate)

    #data = pd.DataFrame(np.zeros((ntimes,12), dtype = float), index = dates, \
    #       columns = ['25', '50', '75', '100', '125', '150', '200', '250',   \
    #       '300', '350' ,'400' , '450'])
    #data_interpolate = amb_mean.regrid(data, iris.analysis.Linear())
    #data.fill(np.nan)
    #for i in dates1:
    #    data.index[i] = neo_mean.index[i]
    #    grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')

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

    #plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

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

    ax = fig.add_subplot(211)

    cmap = plt.cm.viridis_r

    #img = ax.imshow(data, cmap=cmap, origin="upper", interpolation='nearest')

    print(amb_mean)
    #print(neo_mean)
    img = ax.contourf(datanew, cmap=cmap, origin="upper", levels=8)
    cbar = fig.colorbar(img, orientation="horizontal", pad=0.1,
                        shrink=.6)
    cbar.set_label('Volumetric soil water content (%)')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    #print(depths)

    # every second tick
    ax.set_yticks(np.arange(len(depths))[::2])
    depth_labels = np.flipud(depths)
    ax.set_yticklabels(depth_labels[::2])

    # years = dates.year
    #ax.set_xticks(np.arange(len(dates))[::14])
    #cleaner_dates = [str(i)[0:7] for i in dates]
    #ax.set_xticklabels(cleaner_dates[::14])

    ax.set_xticks(np.arange(len(dates_new)))
    cleaner_dates = [str(i)[0:7] for i in dates_new]
    ax.set_xticklabels(cleaner_dates)

    xtickslocs = ax.get_xticks()
    for i in range(len(cleaner_dates)):
        print(xtickslocs[i], cleaner_dates[i])

    cleaner_dates = ["2012-04","2013-01","2014-01","2015-01","2016-01",\
                     "2017-03","2018-01","2019-01",]
    xtickslocs = [1,20,39,57,72,86,94,106]
    xtickslocs_raw = ax.get_xticks()
    xtickslocs_raw = np.arange(len(dates_new))

    #ax.set_xticks(xtickslocs_raw, xtickslocs)
    #ax.set_xticklabels(cleaner_dates)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax.set_ylabel("Depth (cm)")
    ax.axis('tight')

    '''
    # CABLE
    f = nc.Dataset(fcable, 'r')
    time = f.variables['time'][:]
    soil = [1.1, 5.1, 15.7, 43.85, 118.55, 316.4]
    SoilMoist = f.variables['SoilMoist'][:,:,0]
    print(SoilMoist)

    ax1 = fig.add_subplot(212)

    img1 = ax1.contourf(SoilMoist, cmap=cmap, origin="upper", levels=8)
    cbar1 = fig.colorbar(img1, orientation="horizontal", pad=0.1,
                            shrink=.6)
    cbar1.set_label('Volumetric soil water content')
    #tick_locator = ticker.MaxNLocator(nbins=5)
    cbar1.locator = tick_locator
    cbar1.update_ticks()

    ax1.set_yticks(np.arange(len(soil)))
    soil_labels = np.flipud(soil)
    ax1.set_yticklabels(soil_labels)

        # years = dates.year
        #ax.set_xticks(np.arange(len(dates))[::14])
        #cleaner_dates = [str(i)[0:7] for i in dates]
        #ax.set_xticklabels(cleaner_dates[::14])

    ax1.set_xticks(np.arange(len(time)))
    cleaner_dates1 = [str(i)[0:7] for i in time]
    ax1.set_xticklabels(cleaner_dates1)

    xtickslocs1 = ax1.get_xticks()
    for i in range(len(cleaner_dates1)):
        print(xtickslocs1[i], cleaner_dates1[i])

    cleaner_dates1 = ["2012-04","2013-01","2014-01","2015-01","2016-01",\
                     "2017-03","2018-01","2019-01",]
    xtickslocs1 = [1,20,39,57,72,86,94,106]
    xtickslocs_raw1 = ax1.get_xticks()
    xtickslocs_raw1 = np.arange(len(time))

        #ax.set_xticks(xtickslocs_raw, xtickslocs)
        #ax.set_xticklabels(cleaner_dates)
    ax1.set(xticks=xtickslocs1, xticklabels=cleaner_dates1)
    ax1.set_ylabel("Depth (cm)")
    ax1.axis('tight')
    #plt.show()

    #plt.plot(df.index, df.VWC)
    #plt.show()
    '''
    fig.savefig("EucFACE_SW_amb.pdf", bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    fobs = "/short/w35/mm3972/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    fcable = "/short/w35/mm3972/cable/runs/EucFACE/EucFACE_jim_dushan/outputs/EucFACE_amb_out.nc"
    main(fobs, fcable)
