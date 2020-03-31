#!/usr/bin/env python

__author__ = "MU Mengyuan"

import os
import sys
import glob
import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime as dt
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import savgol_filter

def interpolate_raw_lai(lai_fname, ring, method):

    """
    raw LAI data is from 2012-10-26 to 2019-12-29. It needs to be interpolate to
    daily data
    """

    df_lai = pd.read_csv(lai_fname, usecols = ['Ring','LAI','days_201311']) # raw data
    df_lai = df_lai.sort_values(by=['days_201311'])

    # the LAI divide into different columns
    lai = pd.DataFrame(df_lai[df_lai['Ring'].values == 'R1']['LAI'].values, columns=['R1'])
    lai['R2'] = df_lai[df_lai['Ring'].values == 'R2']['LAI'].values
    lai['R3'] = df_lai[df_lai['Ring'].values == 'R3']['LAI'].values
    lai['R4'] = df_lai[df_lai['Ring'].values == 'R4']['LAI'].values
    lai['R5'] = df_lai[df_lai['Ring'].values == 'R5']['LAI'].values
    lai['R6'] = df_lai[df_lai['Ring'].values == 'R6']['LAI'].values
    lai['Date'] = df_lai[df_lai['Ring'].values == 'R6']['days_201311'].values

    # for interpolation, add a row of 2019-12-31
    insertRow = pd.DataFrame([[1.4672, 1.5551, 1.3979, 1.3515, 1.7840, 1.5353, 2555]],columns = ['R1','R2','R3','R4','R5','R6',"Date"])
    lai = lai.append(insertRow,ignore_index=True)

    # make LAI_daily date array
    daily =np.arange(0,2556)

    # interpolate to daily LAI
    if ring == "amb":
        func1 = interp1d(lai['Date'].values, lai['R2'].values, kind = "cubic")#cubic
        func2 = interp1d(lai['Date'].values, lai['R3'].values, kind = "cubic")
        func3 = interp1d(lai['Date'].values, lai['R6'].values, kind = "cubic")
        LAI_daily = (func1(daily)+func2(daily)+func3(daily))/3.
    elif ring == "ele":
        func1 = interp1d(lai['Date'].values, lai['R1'].values, kind = "cubic")
        func2 = interp1d(lai['Date'].values, lai['R4'].values, kind = "cubic")
        func3 = interp1d(lai['Date'].values, lai['R5'].values, kind = "cubic")
        LAI_daily = (func1(daily)+func2(daily)+func3(daily))/3.
    else:
        func = interp1d(lai['Date'].values, lai[ring].values, kind = "cubic")
        LAI_daily = func(daily)
    print(LAI_daily)
    #plt.plot(LAI_daily)
    #plt.show()

    # smooth
    if method == "savgol_filter" :
        # using Savitzky Golay Filter to smooth LAI
        LAI_daily_smooth = savgol_filter(LAI_daily, 91,3) # window size 11, polynomial order 3

    elif method == "average":
        # using 9 points smoothing
        LAI_daily_smooth = np.zeros(len(LAI_daily))
        smooth_length    = 61
        half_length      = 30

        for i in np.arange(len(LAI_daily)):
            if i < half_length:
                LAI_daily_smooth[i] = np.mean(LAI_daily[0:(i+1+half_length)])
            elif i > (2555-half_length):
                LAI_daily_smooth[i] = np.mean(LAI_daily[(i-half_length):])
            else:
                print(i)
                LAI_daily_smooth[i] = np.mean(LAI_daily[(i-half_length):(i+1+half_length)])
                #,LAI_daily

    # linearly interpolate to half hour resolution
    seconds = 2556.*24.*60.*60.
    day_second       = np.arange(0.,seconds,60*60*24.)
    half_hour_second = np.arange(0.,seconds,1800.)

    LAI_half_hour = np.interp(half_hour_second, day_second, LAI_daily_smooth)

    fig = plt.figure(figsize=[12,8])
    ax = fig.add_subplot(111)
    ax.plot(LAI_half_hour,c='red')
    #return LAI_half_hour

    fig.savefig("EucFACE_LAI" , bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":

    met_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/met_July2019/eucMet_gap_filled.csv"
    lai_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/met_2013-2019/eucLAI1319.csv"
    swc_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    tdr_fname = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    stx_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/soil_texture/FACE_P0018_RA_SOILTEXT_L2_20120501.csv"
    method = "savgol_filter" #'average' #"savgol_filter"
    interpolate_raw_lai(lai_fname, 'R2',method)
