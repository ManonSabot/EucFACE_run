#!/usr/bin/env python

"""
Calculate water cycle items and make water budget bar plot

Include functions :

    plot_waterbal
    plot_waterbal_no_total_Evap
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
from matplotlib import cm
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc

def plot_waterbal(fcables, case_labels):
    # ======= Obs ========
    obs = [[155,153,99,34,20,0,0,-113],
           [84,84,61,19,3,0,0,-45],
           [250,120,75,24,21,0,0,114],
           [151,159,106,36,16,0,0,-149],
           [170,132,76,27,30,0,0,-26],
           [150,80,50,13,18,0,0,25]]
           # Autum-2013
           # Winter-2013
           # Spring-2013
           # Summer-2014
           # Autum-2014
           # Winter-2014
    obs_data = np.sum(obs[0:4],axis=0)

    # ======= CABLE =======
    case_sum = len(fcables)
    cable_year = np.zeros([case_sum,8])

    for case_num in np.arange(case_sum):
        cable       = pd.read_csv(fcables[case_num], usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])
        cable['Qs'] = cable['Qs'] + cable['Qsb']
        cable       = cable.drop(['Year','Season','Qsb'], axis=1)
        cable       = cable.drop([0])
        cable_year[case_num,:] = np.sum(cable.iloc[1:5].values,axis=0)

    # using CABLE met rainfall replace G 2015's rainfall
    obs_data[0] = cable_year[0,0]

    # _____________ Make plot _____________
    fig = plt.figure(figsize=(12,9))
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

    colors = cm.tab20(np.linspace(0,1,case_sum))

    labels = ['Rain','Evap','TVeg','ESoil','ECanop','Runoff','Rechrg','ΔS']
    x = np.arange(len(labels))  # the label locations
    width = 1/(case_sum+3)                # the width of the bars

    offset = 1.5*width -0.5

    ax.bar( x + offset , obs_data, width, color='red', label='Obs')

    for case_num in np.arange(case_sum):
        ax.bar(x + offset + (case_num+1)*width, cable_year[case_num,:], width, color=colors[case_num], label=case_labels[case_num])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Water Budget (mm y$^{-1}$)')
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.savefig('../plots/water_balance_2013_obs-ctl-std-exp', bbox_inches='tight',pad_inches=0.1)

def plot_waterbal_no_total_Evap(fcables, case_labels):

    # ======= Obs ========
    obs = [[155,99,34,20,0,0,-113],
           [84,61,19,3,0,0,-45],
           [250,75,24,21,0,0,114],
           [151,106,36,16,0,0,-149],
           [170,76,27,30,0,0,-26],
           [150,50,13,18,0,0,25]]
           # Autumn-2013
           # Winter-2013
           # Spring-2013
           # Summer-2014
           # Autumn-2014
           # Winter-2014
    obs_data = np.sum(obs[0:4],axis=0)

    # ======= CABLE =======
    case_sum = len(fcables)
    cable_year = np.zeros([case_sum,7])

    for case_num in np.arange(case_sum):
        cable       = pd.read_csv(fcables[case_num], usecols = ['Year','Season','Rainf','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])
        cable['Qs'] = cable['Qs'] + cable['Qsb']
        cable       = cable.drop(['Year','Season','Qsb'], axis=1)
        cable       = cable.drop([0])
        cable_year[case_num,:] = np.sum(cable.iloc[1:5].values,axis=0)

    # using CABLE met rainfall replace G 2015's rainfall
    #obs_data[0] = cable_year[0,0]

    # ======================= Plot setting ============================
    fig = plt.figure(figsize=[7,5])
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.0)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    ax = fig.add_subplot(111)

    colors = cm.Set2(np.arange(0,len(case_labels)))

    labels = ['P','T','Es','Ec','R','D','ΔS']
    x = np.arange(len(labels))  # the label locations
    width = 1/(case_sum+3)                # the width of the bars

    offset = 1.5*width -0.5

    ax.bar( x + offset , obs_data, width, color='blue', label='Obs')

    for case_num in np.arange(case_sum):
        ax.bar(x + offset + (case_num+1)*width, cable_year[case_num,:], width,
               color=colors[case_num], label=case_labels[case_num])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mm y$^{-1}$')
    #ax.set_title(title)
    ax.set_ylim(-250, 650)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend( loc='best', frameon=False)

    fig.savefig('../plots/water_balance_Aut2013-Sum2014', bbox_inches='tight',pad_inches=0.1)

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

def calc_waterbal(fcable, layer):

    if layer == "6":
        zse = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
    elif layer == "31uni":
        zse = [ 0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15 ]

    cable  = nc.Dataset(fcable, 'r')

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
    df_cable.to_csv("./csv/EucFACE_amb_%s.csv" %(fcable.split("/")[-2]))
