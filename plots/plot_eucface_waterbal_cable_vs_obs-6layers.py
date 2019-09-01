#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc

layers = 6 # 13 #31

if layers == 6:
    pyth = "6layer_hyds_test"
    zse  = [0.022, 0.058, 0.154, 0.409, 1.085, 2.872]
elif layers == 13:
    pyth = "13layer_hyds_test"
    zse  = [0.02, 0.05,0.06,0.13,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.75,1.50]
elif layers == 31:
    pyth = "31layer_hyds_test"
    zse  = [0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
            0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
            0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
            0.15 ]
case = ["hyds100"]
#case = ["hyds100"]
for case_name in case:
    file_path = "/g/data/w35/mm3972/cable/EucFACE/EucFACE_run/outputs/hyds_test/%s/%s/" % (pyth, case_name)
    famb = "EucFACE_amb_out.nc"
    fele = "EucFACE_ele_out.nc"

    amb = nc.Dataset(os.path.join(file_path,famb), 'r')
    ele = nc.Dataset(os.path.join(file_path,fele), 'r')

    step_2_sec = 30.*60.

    df_amb              = pd.DataFrame(amb.variables['Rainf'][:,0], columns=['Rainf']) # 'Rainfall+snowfall'
    df_amb['Evap']      = amb.variables['Evap'][:,0]   # 'Total evaporation'
    df_amb['TVeg']      = amb.variables['TVeg'][:,0]   # 'Vegetation transpiration'
    df_amb['ESoil']     = amb.variables['ESoil'][:,0]  # 'evaporation from soil'
    df_amb['ECanop']    = amb.variables['ECanop'][:,0] # 'Wet canopy evaporation'
    df_amb['Qs']        = amb.variables['Qs'][:,0]     # 'Surface runoff'
    df_amb['Qsb']       = amb.variables['Qsb'][:,0]    # 'Subsurface runoff'
    df_amb['Qrecharge'] = amb.variables['Qrecharge'][:,0]
    df_amb['dates']     = nc.num2date(amb.variables['time'][:],amb.variables['time'].units)
    df_amb              = df_amb.set_index('dates')

    df_ele              = pd.DataFrame(ele.variables['Rainf'][:,0], columns=['Rainf']) # 'Rainfall+snowfall'
    df_ele['Evap']      = ele.variables['Evap'][:,0]   # 'Total evaporation'
    df_ele['TVeg']      = ele.variables['TVeg'][:,0]   # 'Vegetation transpiration'
    df_ele['ESoil']     = ele.variables['ESoil'][:,0]  # 'evaporation from soil'
    df_ele['ECanop']    = ele.variables['ECanop'][:,0] # 'Wet canopy evaporation'
    df_ele['Qs']        = ele.variables['Qs'][:,0]     # 'Surface runoff'
    df_ele['Qsb']       = ele.variables['Qsb'][:,0]    # 'Subsurface runoff'
    df_ele['Qrecharge'] = ele.variables['Qrecharge'][:,0]
    df_ele['dates']     = nc.num2date(ele.variables['time'][:],ele.variables['time'].units)
    df_ele              = df_ele.set_index('dates')

    df_amb              = df_amb*step_2_sec
    df_amb              = df_amb.resample("M").agg('sum')
    df_amb              = df_amb.drop(df_amb.index[len(df_amb)-1])
    df_amb.index        = df_amb.index.strftime('%Y-%m-%d')
    #turn DatetimeIndex into the formatted strings specified by date_format

    df_ele              = df_ele*step_2_sec
    df_ele              = df_ele.resample("M").agg('sum')
    df_ele              = df_ele.drop(df_ele.index[len(df_ele)-1])
    df_ele.index        = df_ele.index.strftime('%Y-%m-%d')
    #turn DatetimeIndex into the formatted strings specified by date_format

    df_amb['Season']    = np.zeros(len(df_amb))
    df_amb['Year']      = np.zeros(len(df_amb))
    df_ele['Season']    = np.zeros(len(df_ele))
    df_ele['Year']      = np.zeros(len(df_ele))
    for i in np.arange(0,len(df_amb),1):
        df_amb['Year'][i] = df_amb.index[i][0:4]
        df_ele['Year'][i] = df_ele.index[i][0:4]
        if df_amb.index[i][5:7] in ['01','02','12']:
            df_amb['Season'][i] = 1
            df_ele['Season'][i] = 1
        elif df_amb.index[i][5:7] in ['03','04','05']:
            df_amb['Season'][i] = 2
            df_ele['Season'][i] = 2
        elif df_amb.index[i][5:7] in ['06','07','08']:
            df_amb['Season'][i] = 3
            df_ele['Season'][i] = 3
        elif df_amb.index[i][5:7] in ['09','10','11']:
            df_amb['Season'][i] = 4
            df_ele['Season'][i] = 4

    df_amb['Year'][0:-1] = df_amb['Year'][1:]
    df_ele['Year'][0:-1] = df_ele['Year'][1:]


    df_amb = df_amb.groupby(by=['Year','Season']).sum()
    df_ele = df_ele.groupby(by=['Year','Season']).sum()


    df_amb['soil_storage_chg'] = np.zeros(len(df_amb))
    df_ele['soil_storage_chg'] = np.zeros(len(df_ele))


    # Soil Moisture
    df_SM_amb              = pd.DataFrame(amb.variables['SoilMoist'][:,0,0], columns=['SoilMoist'])
    df_SM_ele              = pd.DataFrame(ele.variables['SoilMoist'][:,0,0], columns=['SoilMoist'])
    df_SM_amb['SoilMoist'] = 0.0
    df_SM_ele['SoilMoist'] = 0.0

    for i in np.arange(0,layers,1):
        df_SM_amb = df_SM_amb + amb.variables['SoilMoist'][:,i,0]*zse[i]*1000.
        df_SM_ele = df_SM_ele + ele.variables['SoilMoist'][:,i,0]*zse[i]*1000.

    df_SM_amb['dates']    = nc.num2date(amb.variables['time'][:],amb.variables['time'].units)
    df_SM_amb             = df_SM_amb.set_index('dates')
    df_SM_index_amb       = df_SM_amb.index.strftime('%Y-%m-%d %H:%M')

    df_SM_ele['dates']    = nc.num2date(ele.variables['time'][:],ele.variables['time'].units)
    df_SM_ele             = df_SM_ele.set_index('dates')
    df_SM_index_ele       = df_SM_ele.index.strftime('%Y-%m-%d %H:%M')

    # monthly soil water content and monthly changes
    df_SM_mth_laststep_amb         = df_SM_amb.resample("M").agg('mean')
    j = 0
    for i in np.arange(0,len(df_SM_amb),1):
        if df_SM_amb.index.is_month_end[i] and df_SM_index_amb[i][11:16] == '23:30':
            print(df_SM_amb.index[i])
            print(df_SM_index_amb[i])
            df_SM_mth_laststep_amb.iloc[j] = df_SM_amb.iloc[i]
            j       += 1

    df_SM_mth_laststep_ele         = df_SM_ele.resample("M").agg('mean')
    j = 0
    for i in np.arange(0,len(df_SM_ele),1):
        if df_SM_ele.index.is_month_end[i] and df_SM_index_ele[i][11:16] == '23:30':
            df_SM_mth_laststep_ele.iloc[j] = df_SM_ele.iloc[i]
            j       += 1

    # soil water storage changes
    for i in np.arange(0,25,1):
        a = i+1
        b = 4+i*3
        c = 1+i*3
        print(a)
        print(b)
        df_amb['soil_storage_chg'][a] = df_SM_mth_laststep_amb.iloc[b] - df_SM_mth_laststep_amb.iloc[c]
        df_ele['soil_storage_chg'][a] = df_SM_mth_laststep_ele.iloc[b] - df_SM_mth_laststep_ele.iloc[c]


    # output
    df_amb.to_csv("EucFACE_amb_%slayers_%s_gw_on_or_on.csv" %(layers, case_name))
    df_ele.to_csv("EucFACE_ele_%slayers_%s_gw_on_or_on.csv" %(layers, case_name))
