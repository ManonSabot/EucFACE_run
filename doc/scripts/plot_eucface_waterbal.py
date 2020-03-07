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
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

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
    sim_data_std[-1] = sim_data_std[-1]*10.

    sim_data_site     = np.sum(best_site.iloc[1:5].values,axis=0)
    sim_data_site[-1] = sim_data_site[-1]*10.

    rects1 = ax.bar(x - 0.3, obs_data,      width/4, color='red', label='Obs')
    rects2 = ax.bar(x - 0.1, sim_data,      width/4, color='orange', label='Ctl')
    rects3 = ax.bar(x + 0.1, sim_data_std,  width/4, color='green', label='Best_β-std')
    rects4 = ax.bar(x + 0.3, sim_data_site, width/4, color='blue', label='Best_β-site')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Water Budget $mm y^{-1}$')
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.savefig('../plots/water_balance_2013_obs-ctl-best-site', bbox_inches='tight',pad_inches=0.1)
