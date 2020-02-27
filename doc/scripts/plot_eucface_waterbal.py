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

def calc_waterbal(fcbl_def, fcbl_best, layer_def, layer_best):

    if layer_def == "6":
        zse_def = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
    elif layer_def == "31uni":
        zse_def = [ 0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                     0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                     0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                     0.15 ]
    elif layer_def == "31exp":
        zse_def = [ 0.020440, 0.001759, 0.003957, 0.007035, 0.010993, 0.015829,\
                      0.021546, 0.028141, 0.035616, 0.043971, 0.053205, 0.063318,\
                      0.074311, 0.086183, 0.098934, 0.112565, 0.127076, 0.142465,\
                      0.158735, 0.175883, 0.193911, 0.212819, 0.232606, 0.253272,\
                      0.274818, 0.297243, 0.320547, 0.344731, 0.369794, 0.395737,\
                      0.422559 ]
    elif layer_def == "31para":
        zse_def = [ 0.020000, 0.029420, 0.056810, 0.082172, 0.105504, 0.126808,\
                      0.146083, 0.163328, 0.178545, 0.191733, 0.202892, 0.212023,\
                      0.219124, 0.224196, 0.227240, 0.228244, 0.227240, 0.224196,\
                      0.219124, 0.212023, 0.202892, 0.191733, 0.178545, 0.163328,\
                      0.146083, 0.126808, 0.105504, 0.082172, 0.056810, 0.029420,\
                      0.020000 ]

    if layer_best == "6":
        zse_best = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
    elif layer_best == "31uni":
        zse_best = [ 0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                     0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                     0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                     0.15 ]
    elif layer_best == "31exp":
        zse_best = [ 0.020440, 0.001759, 0.003957, 0.007035, 0.010993, 0.015829,\
                      0.021546, 0.028141, 0.035616, 0.043971, 0.053205, 0.063318,\
                      0.074311, 0.086183, 0.098934, 0.112565, 0.127076, 0.142465,\
                      0.158735, 0.175883, 0.193911, 0.212819, 0.232606, 0.253272,\
                      0.274818, 0.297243, 0.320547, 0.344731, 0.369794, 0.395737,\
                      0.422559 ]
    elif layer_best == "31para":
        zse_best = [ 0.020000, 0.029420, 0.056810, 0.082172, 0.105504, 0.126808,\
                      0.146083, 0.163328, 0.178545, 0.191733, 0.202892, 0.212023,\
                      0.219124, 0.224196, 0.227240, 0.228244, 0.227240, 0.224196,\
                      0.219124, 0.212023, 0.202892, 0.191733, 0.178545, 0.163328,\
                      0.146083, 0.126808, 0.105504, 0.082172, 0.056810, 0.029420,\
                      0.020000 ]

    cable_def  = nc.Dataset(fcbl_def, 'r')
    cable_best = nc.Dataset(fcbl_best, 'r')

    step_2_sec = 30.*60.

    df_def              = pd.DataFrame(cable_def.variables['Rainf'][:,0], columns=['Rainf']) # 'Rainfall+snowfall'
    df_def['Evap']      = cable_def.variables['Evap'][:,0]   # 'Total evaporation'
    df_def['TVeg']      = cable_def.variables['TVeg'][:,0]   # 'Vegetation transpiration'
    df_def['ESoil']     = cable_def.variables['ESoil'][:,0]  # 'evaporation from soil'
    df_def['ECanop']    = cable_def.variables['ECanop'][:,0] # 'Wet canopy evaporation'
    df_def['Qs']        = cable_def.variables['Qs'][:,0]     # 'Surface runoff'
    df_def['Qsb']       = cable_def.variables['Qsb'][:,0]    # 'Subsurface runoff'
    df_def['Qrecharge'] = cable_def.variables['Qrecharge'][:,0]
    df_def['dates']     = nc.num2date(cable_def.variables['time'][:],cable_def.variables['time'].units)
    df_def              = df_def.set_index('dates')

    df_best              = pd.DataFrame(cable_best.variables['Rainf'][:,0], columns=['Rainf']) # 'Rainfall+snowfall'
    df_best['Evap']      = cable_best.variables['Evap'][:,0]   # 'Total evaporation'
    df_best['TVeg']      = cable_best.variables['TVeg'][:,0]   # 'Vegetation transpiration'
    df_best['ESoil']     = cable_best.variables['ESoil'][:,0]  # 'evaporation from soil'
    df_best['ECanop']    = cable_best.variables['ECanop'][:,0] # 'Wet canopy evaporation'
    df_best['Qs']        = cable_best.variables['Qs'][:,0]     # 'Surface runoff'
    df_best['Qsb']       = cable_best.variables['Qsb'][:,0]    # 'Subsurface runoff'
    df_best['Qrecharge'] = cable_best.variables['Qrecharge'][:,0]
    df_best['dates']     = nc.num2date(cable_best.variables['time'][:],cable_best.variables['time'].units)
    df_best              = df_best.set_index('dates')

    df_def              = df_def*step_2_sec
    df_def              = df_def.resample("M").agg('sum')
    df_def              = df_def.drop(df_def.index[len(df_def)-1])
    df_def.index        = df_def.index.strftime('%Y-%m-%d')
    #turn DatetimeIndex into the formatted strings specified by date_format

    df_best              = df_best*step_2_sec
    df_best              = df_best.resample("M").agg('sum')
    df_best              = df_best.drop(df_best.index[len(df_best)-1])
    df_best.index        = df_best.index.strftime('%Y-%m-%d')
    #turn DatetimeIndex into the formatted strings specified by date_format

    df_def['Season']    = np.zeros(len(df_def))
    df_def['Year']      = np.zeros(len(df_def))
    df_best['Season']    = np.zeros(len(df_best))
    df_best['Year']      = np.zeros(len(df_best))
    for i in np.arange(0,len(df_def),1):
        df_def['Year'][i] = df_def.index[i][0:4]
        df_best['Year'][i] = df_best.index[i][0:4]
        if df_def.index[i][5:7] in ['01','02','12']:
            df_def['Season'][i] = 1
            df_best['Season'][i] = 1
        elif df_def.index[i][5:7] in ['03','04','05']:
            df_def['Season'][i] = 2
            df_best['Season'][i] = 2
        elif df_def.index[i][5:7] in ['06','07','08']:
            df_def['Season'][i] = 3
            df_best['Season'][i] = 3
        elif df_def.index[i][5:7] in ['09','10','11']:
            df_def['Season'][i] = 4
            df_best['Season'][i] = 4

    df_def['Year'][0:-1] = df_def['Year'][1:]
    df_best['Year'][0:-1] = df_best['Year'][1:]

    df_def = df_def.groupby(by=['Year','Season']).sum()
    df_best = df_best.groupby(by=['Year','Season']).sum()

    df_def['soil_storage_chg']  = np.zeros(len(df_def))
    df_best['soil_storage_chg'] = np.zeros(len(df_best))

    # Soil Moisture
    df_SM_def               = pd.DataFrame(cable_def.variables['SoilMoist'][:,0,0], columns=['SoilMoist'])
    df_SM_best              = pd.DataFrame(cable_best.variables['SoilMoist'][:,0,0], columns=['SoilMoist'])
    df_SM_def['SoilMoist']  = 0.0
    df_SM_best['SoilMoist'] = 0.0

    for i in np.arange(len(zse_def)):
        df_SM_def  = df_SM_def + cable_def.variables['SoilMoist'][:,i,0]*zse_def[i]*1000.
    for i in np.arange(len(zse_best)):
        df_SM_best = df_SM_best + cable_best.variables['SoilMoist'][:,i,0]*zse_best[i]*1000.

    df_SM_def['dates']    = nc.num2date(cable_def.variables['time'][:],cable_def.variables['time'].units)
    df_SM_def             = df_SM_def.set_index('dates')
    df_SM_index_def       = df_SM_def.index.strftime('%Y-%m-%d %H:%M')

    df_SM_best['dates']    = nc.num2date(cable_best.variables['time'][:],cable_best.variables['time'].units)
    df_SM_best             = df_SM_best.set_index('dates')
    df_SM_index_best       = df_SM_best.index.strftime('%Y-%m-%d %H:%M')

    # monthly soil water content and monthly changes
    df_SM_mth_laststep_def = df_SM_def.resample("M").agg('mean')
    j = 0
    for i in np.arange(0,len(df_SM_def),1):
        if df_SM_def.index.is_month_end[i] and df_SM_index_def[i][11:16] == '23:30':
            print(df_SM_def.index[i])
            print(df_SM_index_def[i])
            df_SM_mth_laststep_def.iloc[j] = df_SM_def.iloc[i]
            j       += 1

    df_SM_mth_laststep_best         = df_SM_best.resample("M").agg('mean')
    j = 0
    for i in np.arange(0,len(df_SM_best),1):
        if df_SM_best.index.is_month_end[i] and df_SM_index_best[i][11:16] == '23:30':
            df_SM_mth_laststep_best.iloc[j] = df_SM_best.iloc[i]
            j       += 1

    # soil water storage changes
    for i in np.arange(0,25,1):
        a = i+1
        b = 4+i*3
        c = 1+i*3
        print(a)
        print(b)
        df_def['soil_storage_chg'][a] = df_SM_mth_laststep_def.iloc[b] - df_SM_mth_laststep_def.iloc[c]
        df_best['soil_storage_chg'][a] = df_SM_mth_laststep_best.iloc[b] - df_SM_mth_laststep_best.iloc[c]

    # output
    df_def.to_csv("EucFACE_%s.csv" %(fcbl_def.split("/")[-2]))
    df_best.to_csv("EucFACE_%s.csv" %(fcbl_best.split("/")[-2]))


def plot_waterbal(fwatbal_def,fwatbal_best):

    dfl = pd.read_csv(fwatbal_def, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])
    ctl = pd.read_csv(fwatbal_best, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])

    dfl['Qs'] = dfl['Qs']+dfl['Qsb']
    ctl['Qs'] = ctl['Qs']+ctl['Qsb']

    dfl = dfl.drop(['Year','Season','Qsb'], axis=1)
    ctl = ctl.drop(['Year','Season','Qsb'], axis=1)

    dfl = dfl.drop([0])
    ctl = ctl.drop([0])

    print(dfl)
    print(ctl)

    obs = [[155,153,99,34,20,0,0,-113],\
           [84,84,61,19,3,0,0,-45],\
           [250,120,75,24,21,0,0,114],\
           [151,159,106,36,16,0,0,-149],\
           [170,132,76,27,30,0,0,-26],\
           [150,80,50,13,18,0,0,25]]
           # Autum-2013
           # Winter-2013
           # Spring-2013
           # Summer-2014
           # Autum-2014
           # Winter-2014
    print(np.sum(dfl.iloc[0:4].values,axis=0))
    print(np.sum(obs[1:5],axis=0))

    #title = 'Water Balance'

    # _____________ Make plot _____________
    fig = plt.figure(figsize=(8,6))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax = fig.add_subplot(111)

    labels = ['Rain','Evap','TVeg','ESoil','ECanop','Runoff','Rechrg','ΔS']
    x = np.arange(len(labels))  # the label locations
    width = 0.6                # the width of the bars

    # using CABLE met rainfall replace G 2015's rainfall
    obs_data = np.sum(obs[0:4],axis=0)
    sim_data = np.sum(dfl.iloc[1:5].values,axis=0)
    obs_data[0] = sim_data[0]
    print(obs_data)
    print(sim_data)

    sim_data_1     = np.sum(ctl.iloc[1:5].values,axis=0)
    sim_data_1[-1] = sim_data_1[-1]*10.

    rects1 = ax.bar(x - 0.2, obs_data, width/3, color='blue', label='Obs')
    rects2 = ax.bar(x      , sim_data, width/3, color='orange', label='Ctl')
    rects3 = ax.bar(x + 0.2, sim_data_1, width/3, color='green', label='Best')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Water Balance Element $mm y^{-1}$')
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.savefig('../plots/water_balance_2013_obs-def-best', bbox_inches='tight',pad_inches=0.1)
