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
from scipy.interpolate import spline

def interpolate_raw_lai(lai_fname, ring):

    """
    raw LAI data is from 2012-10-26 to 2019-12-29. It needs to be interpolate to
    daily data
    """

    df_lai      = pd.read_csv(lai_fname, usecols = ['Ring','LAI','days_201311']) # raw data
    #df_lai_date = pd.read_csv(lai_fname, usecols = ['Date'])       # raw data
    #df_lai['Date'] = pd.to_datetime(df_lai_date['Date'],format="%d/%m/%Y",infer_datetime_format=False)
    df_lai = df_lai.sort_values(by=['days_201311'])

    # the LAI divide into different columns
    lai = pd.DataFrame(df_lai[df_lai['Ring'].values == 'R1']['LAI'].values, columns=['R1'])
    lai['R2'] = df_lai[df_lai['Ring'].values == 'R2']['LAI'].values
    lai['R3'] = df_lai[df_lai['Ring'].values == 'R3']['LAI'].values
    lai['R4'] = df_lai[df_lai['Ring'].values == 'R4']['LAI'].values
    lai['R5'] = df_lai[df_lai['Ring'].values == 'R5']['LAI'].values
    lai['R6'] = df_lai[df_lai['Ring'].values == 'R6']['LAI'].values
    lai['Date'] = df_lai[df_lai['Ring'].values == 'R6']['days_201311'].values
    #lai       = lai.set_index('Date')
    #lai       = lai.resample("D")

    #lai.index = lai.index.astype('datetime64[D]') - pd.datetime(2013,1,1)
    #lai.index = lai.index.days

    #lai['Date'] = day_1[].dt.total_seconds()

    print(lai)
    # make LAI_daily date array
    daily =np.arange(0,2556)
    #print(LAI_daily)
    #day_2     = LAI_daily['Date']-np.datetime64('2013-01-01','D')
    #day_2     = day_2.total_seconds()
    #print(day_1)
    #print(np.datetime64('2013-01-01','D'))
    #print(LAI_daily)
    R1 = np.interp(daily, lai['Date'].values, lai['R1'].values)
    R2 = np.interp(daily, lai['Date'].values, lai['R2'].values)
    R3 = np.interp(daily, lai['Date'].values, lai['R3'].values)
    R4 = np.interp(daily, lai['Date'].values, lai['R4'].values)
    R5 = np.interp(daily, lai['Date'].values, lai['R5'].values)
    R6 = np.interp(daily, lai['Date'].values, lai['R6'].values)



    if ring == "amb":
        LAI_daily = (np.interp(daily, lai['Date'].values, lai['R2'].values)
                   + np.interp(daily, lai['Date'].values, lai['R3'].values)
                   + np.interp(daily, lai['Date'].values, lai['R6'].values))/3.
    elif ring == "ele":
        LAI_daily = (np.interp(daily, lai['Date'].values, lai['R1'].values)
                   + np.interp(daily, lai['Date'].values, lai['R3'].values)
                   + np.interp(daily, lai['Date'].values, lai['R4'].values))/3.
    else:
        LAI_daily = np.interp(daily, lai['Date'].values, lai[ring].values)

    #date = LAI_daily['Date'] - np.datetime64('2013-01-01T00:00:00')

    #LAI_daily['Date']  = np.zeros(len(LAI_daily))
    #for i in np.arange(0,len(LAI_daily),1):
    #    LAI_daily['Date'][i] = date.iloc[i].total_seconds()
    seconds = 2556.*24.*60.*60.
    half_hour = np.arange(0.,seconds,60*60*24.)
    grid_x = np.arange(0.,seconds,1800.)

    LAI_half_hour = np.interp(grid_x, half_hour, LAI_daily)
    y_smooth = spline(half_hour,LAI_daily, grid_x)
    plt.plot(y_smooth)
    plt.show()
    #return LAI_half_hour

if __name__ == "__main__":

    met_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/met_July2019/eucMet_gap_filled.csv"
    lai_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/met_2013-2019/eucLAI1319.csv"
    swc_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    tdr_fname = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    stx_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/soil_texture/FACE_P0018_RA_SOILTEXT_L2_20120501.csv"

    interpolate_raw_lai(lai_fname, 'R2')
