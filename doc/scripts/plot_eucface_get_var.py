#!/usr/bin/env python

"""
Get variable

"""

__author__ = "MU Mengyuan"
__version__ = "2020-03-04"

import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import netCDF4 as nc

def read_cable_var(fcable, var_name):

    """
    read a var from CABLE output
    """

    print("carry on read_cable_var")
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    if var_name in ["TVeg", "ESoil", "Rainf"]:
        var = pd.DataFrame(cable.variables[var_name][:,0,0]*1800., columns=['cable'])
    else:
        var = pd.DataFrame(cable.variables[var_name][:,0,0], columns=['cable'])
    var['Date'] = Time
    var = var.set_index('Date')
    if var_name in ["TVeg", "ESoil", "Rainf"]:
        var = var.resample("D").agg('sum')
    elif var_name in ["Fwsoil"]:
        var = var.resample("D").agg('mean')
    var.index = var.index - pd.datetime(2011,12,31)
    var.index = var.index.days
    var = var.sort_values(by=['Date'])

    return var

def read_cable_SM_one_clmn(fcable, layer):
    """
    Note: the SM here is for plotting profile, it has been turned into one column
          by SoilMoist = SoilMoist.stack()
    """
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
    print("----------------------------")
    print(SoilMoist)

    return SoilMoist

def read_cable_SM(fcable, layer):
    """
    Note: the SM here is for plotting profile, it has been turned into one column
          by SoilMoist = SoilMoist.stack()
    """
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

    return SoilMoist

def read_obs_esoil(ring):

    fobs_Esoil = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_underET.csv"

    est_esoil = pd.read_csv(fobs_Esoil, usecols = ['Ring','Date','wuTP'])
    est_esoil['Date'] = pd.to_datetime(est_esoil['Date'],format="%d/%m/%Y",infer_datetime_format=False)
    est_esoil['Date'] = est_esoil['Date'] - pd.datetime(2011,12,31)
    est_esoil['Date'] = est_esoil['Date'].dt.days
    est_esoil = est_esoil.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
       subs = est_esoil[(est_esoil['Ring'].isin(['R2','R3','R6'])) & (est_esoil.Date > 366)]
    elif ring == 'ele':
       subs = est_esoil[(est_esoil['Ring'].isin(['R1','R4','R5'])) & (est_esoil.Date > 366)]
    else:
       subs = est_esoil[(est_esoil['Ring'].isin([ring]))  & (est_esoil.Date > 366)]

    subs = subs.groupby(by=["Date"]).mean()
    subs['wuTP']   = subs['wuTP'].clip(lower=0.)
    subs['wuTP']   = subs['wuTP'].replace(0., float('nan'))
    subs = subs.rename({'wuTP' : 'obs'}, axis='columns')

    return subs

def read_obs_trans(ring):

    fobs_Trans = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_sapflow.csv"

    est_trans = pd.read_csv(fobs_Trans, usecols = ['Ring','Date','volRing'])
    est_trans['Date'] = pd.to_datetime(est_trans['Date'],format="%d/%m/%Y",infer_datetime_format=False)
    est_trans['Date'] = est_trans['Date'] - pd.datetime(2011,12,31)
    est_trans['Date'] = est_trans['Date'].dt.days
    est_trans = est_trans.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
       subs = est_trans[(est_trans['Ring'].isin(['R2','R3','R6'])) & (est_trans.Date > 366)]
    elif ring == 'ele':
       subs = est_trans[(est_trans['Ring'].isin(['R1','R4','R5'])) & (est_trans.Date > 366)]
    else:
       subs = est_trans[(est_trans['Ring'].isin([ring]))  & (est_trans.Date > 366)]

    subs = subs.groupby(by=["Date"]).mean()
    subs['volRing']   = subs['volRing'].clip(lower=0.)
    subs['volRing']   = subs['volRing'].replace(0., float('nan'))
    subs = subs.rename({'volRing' : 'obs'}, axis='columns')

    return subs

def read_obs_swc_tdr(ring):

    fobs   = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    tdr = pd.read_csv(fobs, usecols = ['Ring','Date','swc.tdr'])
    tdr['Date'] = pd.to_datetime(tdr['Date'],format="%Y-%m-%d",infer_datetime_format=False)
    tdr['Date'] = tdr['Date'] - pd.datetime(2011,12,31)
    tdr['Date'] = tdr['Date'].dt.days
    tdr = tdr.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
        subset = tdr[(tdr['Ring'].isin(['R2','R3','R6'])) & (tdr.Date > 366)]
    elif ring == 'ele':
        subset = tdr[(tdr['Ring'].isin(['R1','R4','R5'])) & (tdr.Date > 366)]
    else:
        subset = tdr[(tdr['Ring'].isin([ring]))  & (tdr.Date > 366)]

    subset = subset.groupby(by=["Date"]).mean()/100.
    subset = subset.rename({'swc.tdr' : 'obs'}, axis='columns')
    return subset


def read_obs_swc_neo(ring):

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

    # remove 'VWC'
    subset = subset.xs('VWC', axis=1, drop_level=True)

    # 'VWC' : key on which to get cross section
    # axis=1 : get cross section of column
    # drop_level=True : returns cross section without the multilevel index

    #neo_mean = np.transpose(neo_mean)

    return subset
