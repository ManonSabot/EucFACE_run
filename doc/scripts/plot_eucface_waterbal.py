#!/usr/bin/env python

"""
Calculate water cycle items and make water budget bar plot

Include functions :

    plot_waterbal
    calc_waterbal_year
    calc_waterbal

"""

__author__ = "MU Mengyuan"
__version__ = "2020-03-10"

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

def plot_waterbal(fwatbal_ctl, fwatbal_best_std, fwatbal_best_site):

    ctl       = pd.read_csv(fwatbal_ctl, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])
    best_std  = pd.read_csv(fwatbal_best_std, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])
    best_site = pd.read_csv(fwatbal_best_site, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])

    ctl['Qs']       = ctl['Qs'] + ctl['Qsb']
    best_std['Qs']  = best_std['Qs'] + best_std['Qsb']
    best_site['Qs'] = best_site['Qs']+ best_site['Qsb']

    ctl       = ctl.drop(['Year','Season','Qsb'], axis=1)
    best_std  = best_std.drop(['Year','Season','Qsb'], axis=1)
    best_site = best_site.drop(['Year','Season','Qsb'], axis=1)

    ctl = ctl.drop([0])
    best_std  = best_std.drop([0])
    best_site = best_site.drop([0])

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

    #title = 'Water Balance'

    # _____________ Make plot _____________
    fig = plt.figure(figsize=(8,6))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)

    labels = ['Rain','Evap','TVeg','ESoil','ECanop','Runoff','Rechrg','ΔS']
    x = np.arange(len(labels))  # the label locations
    width = 0.8                # the width of the bars

    # using CABLE met rainfall replace G 2015's rainfall
    obs_data = np.sum(obs[0:4],axis=0)
    sim_data = np.sum(ctl.iloc[1:5].values,axis=0)
    obs_data[0] = sim_data[0]
    print(obs_data)
    print(sim_data)

    sim_data_std     = np.sum(best_std.iloc[1:5].values,axis=0)
    sim_data_std[-1] = sim_data_std[-1]

    sim_data_site     = np.sum(best_site.iloc[1:5].values,axis=0)
    sim_data_site[-1] = sim_data_site[-1]

    rects1 = ax.bar(x - 0.3, obs_data,      width/4, color='blue', label='Obs')
    rects2 = ax.bar(x - 0.1, sim_data,      width/4, color='red', label='Ctl')
    rects3 = ax.bar(x + 0.1, sim_data_std,  width/4, color='orange', label='β-std')
    rects4 = ax.bar(x + 0.3, sim_data_site, width/4, color='green', label='β-exp')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Water Budget $mm y^{-1}$')
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.savefig('../plots/water_balance_2013_obs-ctl-std-exp', bbox_inches='tight',pad_inches=0.1)

def calc_waterbal_year(fcbl, layer):

    if layer == "6":
        zse = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
    elif layer == "31uni":
        zse = [ 0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15 ]

    cable  = nc.Dataset(fcbl, 'r')
    step_2_sec = 30.*60.

    df              = pd.DataFrame(cable.variables['Rainf'][:,0,0]*step_2_sec, columns=['Rainf']) # 'Rainfall+snowfall'
    df['Evap']      = cable.variables['Evap'][:,0,0]*step_2_sec   # 'Total evaporation'
    df['TVeg']      = cable.variables['TVeg'][:,0,0]*step_2_sec   # 'Vegetation transpiration'
    df['ESoil']     = cable.variables['ESoil'][:,0,0]*step_2_sec  # 'evaporation from soil'
    df['ECanop']    = cable.variables['ECanop'][:,0,0]*step_2_sec # 'Wet canopy evaporation'
    df['Qs']        = cable.variables['Qs'][:,0,0]*step_2_sec     # 'Surface runoff'
    df['Qsb']       = cable.variables['Qsb'][:,0,0]*step_2_sec    # 'Subsurface runoff'
    df['Qrecharge'] = cable.variables['Qrecharge'][:,0,0]*step_2_sec
    df['dates']     = nc.num2date(cable.variables['time'][:], cable.variables['time'].units)
    df              = df.set_index('dates')

    df              = df.resample("Y").agg('sum')
    print(df)
    #df              = df.drop(df.index[len(df)-1])
    #df.index        = df.index.strftime('%Y-%m-%d')
    #turn DatetimeIndex into the formatted strings specified by date_format

    df['soil_storage_chg']  = np.zeros(len(df))
    # Soil Moisture
    df_SM               = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])
    df_SM['SoilMoist']  = 0.0
    print(zse)
    for i in np.arange(len(zse)):
        df_SM['SoilMoist']  = df_SM['SoilMoist'] + cable.variables['SoilMoist'][:,i,0,0]*zse[i]*1000.


    df_SM['dates']     = nc.num2date(cable.variables['time'][:], cable.variables['time'].units)
    df_SM              = df_SM.set_index('dates')
    df_SM              = df_SM.resample("D").agg('mean')

    # monthly soil water content and monthly changes
    df_SM_year_start  = df_SM[df_SM.index.is_year_start]
    print(df_SM_year_start)

    df_SM_year_end  = df_SM[df_SM.index.is_year_end]
    print(df_SM_year_end)
    print(df_SM_year_end['SoilMoist'])
    df.soil_storage_chg[0:6] = df_SM_year_end.SoilMoist.values[0:6] - df_SM_year_start.SoilMoist[0:6]
    # output
    df.to_csv("EucFACE_year_%s.csv" %(fcbl.split("/")[-2]))

def calc_waterbal(fcbl, layer):

    if layer == "6":
        zse = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
    elif layer == "31uni":
        zse = [ 0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15 ]

    cable  = nc.Dataset(fcbl, 'r')

    step_2_sec = 30.*60.

    df_cable              = pd.DataFrame(cable.variables['Rainf'][:,0], columns=['Rainf']) # 'Rainfall+snowfall'
    df_cable['Evap']      = cable.variables['Evap'][:,0]   # 'Total evaporation'
    df_cable['TVeg']      = cable.variables['TVeg'][:,0]   # 'Vegetation transpiration'
    df_cable['ESoil']     = cable.variables['ESoil'][:,0]  # 'evaporation from soil'
    df_cable['ECanop']    = cable.variables['ECanop'][:,0] # 'Wet canopy evaporation'
    df_cable['Qs']        = cable.variables['Qs'][:,0]     # 'Surface runoff'
    df_cable['Qsb']       = cable.variables['Qsb'][:,0]    # 'Subsurface runoff'
    df_cable['Qrecharge'] = cable.variables['Qrecharge'][:,0]
    df_cable['dates']     = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    df_cable              = df_cable.set_index('dates')

    df_cable              = df_cable*step_2_sec
    df_cable              = df_cable.resample("M").agg('sum')
    df_cable              = df_cable.drop(df_cable.index[len(df_cable)-1])
    df_cable.index        = df_cable.index.strftime('%Y-%m-%d')
    #turn DatetimeIndex into the formatted strings specified by date_format

    df_cable['Season']    = np.zeros(len(df_cable))
    df_cable['Year']      = np.zeros(len(df_cable))
    for i in np.arange(0,len(df_cable),1):
        df_cable['Year'][i] = df_cable.index[i][0:4]
        if df_cable.index[i][5:7] in ['01','02','12']:
            df_cable['Season'][i] = 1
        elif df_cable.index[i][5:7] in ['03','04','05']:
            df_cable['Season'][i] = 2
        elif df_cable.index[i][5:7] in ['06','07','08']:
            df_cable['Season'][i] = 3
        elif df_cable.index[i][5:7] in ['09','10','11']:
            df_cable['Season'][i] = 4

    df_cable['Year'][0:-1] = df_cable['Year'][1:]

    df_cable = df_cable.groupby(by=['Year','Season']).sum()

    df_cable['soil_storage_chg'] = np.zeros(len(df_cable))

    # Soil Moisture
    df_SM_cable              = pd.DataFrame(cable.variables['SoilMoist'][:,0,0], columns=['SoilMoist'])
    df_SM_cable['SoilMoist'] = 0.0

    for i in np.arange(0,len(zse),1):
        df_SM_cable = df_SM_cable + cable.variables['SoilMoist'][:,i,0]*zse[i]*1000.

    df_SM_cable['dates']    = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    df_SM_cable             = df_SM_cable.set_index('dates')
    df_SM_index_cable       = df_SM_cable.index.strftime('%Y-%m-%d %H:%M')

    # monthly soil water content and monthly changes
    df_SM_mth_laststep_cable         = df_SM_cable.resample("M").agg('mean')
    j = 0
    for i in np.arange(0,len(df_SM_cable),1):
        if df_SM_cable.index.is_month_end[i] and df_SM_index_cable[i][11:16] == '23:30':
            print(df_SM_cable.index[i])
            print(df_SM_index_cable[i])
            df_SM_mth_laststep_cable.iloc[j] = df_SM_cable.iloc[i]
            j       += 1

    # soil water storage changes
    for i in np.arange(0,25,1):
        a = i+1
        b = 4+i*3
        c = 1+i*3
        print(a)
        print(b)
        df_cable['soil_storage_chg'][a] = df_SM_mth_laststep_cable.iloc[b] - df_SM_mth_laststep_cable.iloc[c]

    # output
    df_cable.to_csv("./csv/EucFACE_amb_%s.csv" %(fcbl.split("/")[-2]))
