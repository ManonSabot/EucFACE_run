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

    if layer_best == "6":
        zse_best = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
    elif layer_best == "31uni":
        zse_best = [ 0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                     0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                     0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                     0.15 ]

    cable_def  = nc.Dataset(fcbl_def, 'r')
    cable_best = nc.Dataset(fcbl_best, 'r')

    step_2_sec = 30.*60.

    df_def              = pd.DataFrame(cable_def.variables['Rainf'][:,0,0]*step_2_sec, columns=['Rainf']) # 'Rainfall+snowfall'
    df_def['Evap']      = cable_def.variables['Evap'][:,0,0]*step_2_sec   # 'Total evaporation'
    df_def['TVeg']      = cable_def.variables['TVeg'][:,0,0]*step_2_sec   # 'Vegetation transpiration'
    df_def['ESoil']     = cable_def.variables['ESoil'][:,0,0]*step_2_sec  # 'evaporation from soil'
    df_def['ECanop']    = cable_def.variables['ECanop'][:,0,0]*step_2_sec # 'Wet canopy evaporation'
    df_def['Qs']        = cable_def.variables['Qs'][:,0,0]*step_2_sec     # 'Surface runoff'
    df_def['Qsb']       = cable_def.variables['Qsb'][:,0,0]*step_2_sec    # 'Subsurface runoff'
    df_def['Qrecharge'] = cable_def.variables['Qrecharge'][:,0,0]*step_2_sec
    df_def['dates']     = nc.num2date(cable_def.variables['time'][:], cable_def.variables['time'].units)
    df_def              = df_def.set_index('dates')

    df_best              = pd.DataFrame(cable_best.variables['Rainf'][:,0,0]*step_2_sec, columns=['Rainf']) # 'Rainfall+snowfall'
    df_best['Evap']      = cable_best.variables['Evap'][:,0,0]*step_2_sec   # 'Total evaporation'
    df_best['TVeg']      = cable_best.variables['TVeg'][:,0,0]*step_2_sec   # 'Vegetation transpiration'
    df_best['ESoil']     = cable_best.variables['ESoil'][:,0,0]*step_2_sec  # 'evaporation from soil'
    df_best['ECanop']    = cable_best.variables['ECanop'][:,0,0]*step_2_sec # 'Wet canopy evaporation'
    df_best['Qs']        = cable_best.variables['Qs'][:,0,0]*step_2_sec     # 'Surface runoff'
    df_best['Qsb']       = cable_best.variables['Qsb'][:,0,0]*step_2_sec    # 'Subsurface runoff'
    df_best['Qrecharge'] = cable_best.variables['Qrecharge'][:,0,0]*step_2_sec
    df_best['dates']     = nc.num2date(cable_best.variables['time'][:],cable_best.variables['time'].units)
    df_best              = df_best.set_index('dates')

    df_def              = df_def.resample("Y").agg('sum')
    print(df_def)
    #df_def              = df_def.drop(df_def.index[len(df_def)-1])
    #df_def.index        = df_def.index.strftime('%Y-%m-%d')
    #turn DatetimeIndex into the formatted strings specified by date_format

    df_best              = df_best.resample("Y").agg('sum')
    print(df_best)
    #df_best              = df_best.drop(df_best.index[len(df_best)-1])
    #df_best.index        = df_best.index.strftime('%Y-%m-%d')
    #turn DatetimeIndex into the formatted strings specified by date_format

    df_def['soil_storage_chg']  = np.zeros(len(df_def))
    df_best['soil_storage_chg'] = np.zeros(len(df_best))

    # Soil Moisture
    df_SM_def               = pd.DataFrame(cable_def.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])
    df_SM_best              = pd.DataFrame(cable_best.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])
    df_SM_def['SoilMoist']  = 0.0
    df_SM_best['SoilMoist'] = 0.0
    print(zse_def)
    for i in np.arange(len(zse_def)):
        df_SM_def['SoilMoist']  = df_SM_def['SoilMoist'] + cable_def.variables['SoilMoist'][:,i,0,0]*zse_def[i]*1000.
    for i in np.arange(len(zse_best)):
        df_SM_best['SoilMoist'] = df_SM_best['SoilMoist'] + cable_best.variables['SoilMoist'][:,i,0,0]*zse_best[i]*1000.

    df_SM_def['dates']     = nc.num2date(cable_def.variables['time'][:], cable_def.variables['time'].units)
    df_SM_def              = df_SM_def.set_index('dates')

    df_SM_best['dates']    = nc.num2date(cable_def.variables['time'][:], cable_def.variables['time'].units)
    df_SM_best             = df_SM_best.set_index('dates')

    df_SM_def              = df_SM_def.resample("D").agg('mean')
    df_SM_best             = df_SM_best.resample("D").agg('mean')

    # monthly soil water content and monthly changes
    df_SM_def_year_start  = df_SM_def[df_SM_def.index.is_year_start]
    df_SM_best_year_start = df_SM_best[df_SM_best.index.is_year_start]
    print(df_SM_def_year_start)
    print(df_SM_best_year_start)

    df_SM_def_year_end  = df_SM_def[df_SM_def.index.is_year_end]
    df_SM_best_year_end = df_SM_best[df_SM_best.index.is_year_end]
    print(df_SM_def_year_end)
    print(df_SM_best_year_end)
    print(df_SM_def_year_end['SoilMoist'])
    df_def.soil_storage_chg[0:6] = df_SM_def_year_end.SoilMoist.values[0:6] - df_SM_def_year_start.SoilMoist[0:6]
    df_best.soil_storage_chg[0:6]= df_SM_best_year_end.SoilMoist.values[0:6] - df_SM_best_year_start.SoilMoist[0:6]
    print(df_def)
    print(df_best)
    # output
    df_def.to_csv("EucFACE_def_year_%s.csv" %(fcbl_def.split("/")[-2]))
    df_best.to_csv("EucFACE_best_year_%s.csv" %(fcbl_best.split("/")[-2]))


if __name__ == "__main__":

    ring  = "amb"

    contour = False

    case_1 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6"
    fcbl_1 ="%s/EucFACE_%s_out.nc" % (case_1, ring)

    case_2 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6_litter"
    fcbl_2 ="%s/EucFACE_%s_out.nc" % (case_2, ring)

    case_3 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_SM_6_litter"
    fcbl_3 ="%s/EucFACE_%s_out.nc" % (case_3, ring)

    case_4 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_31uni_litter"
    fcbl_4 ="%s/EucFACE_%s_out.nc" % (case_4, ring)

    case_5 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter"
    fcbl_5 ="%s/EucFACE_%s_out.nc" % (case_5, ring)

    case_6 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter"
    fcbl_6 ="%s/EucFACE_%s_out.nc" % (case_6, ring)

    case_7 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_hie-exp"
    fcbl_7 ="%s/EucFACE_%s_out.nc" % (case_7, ring)

    case_8 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_Hvrd"
    fcbl_8 ="%s/EucFACE_%s_out.nc" % (case_8, ring)

    calc_waterbal(fcbl_6, fcbl_7, "31uni", "31uni")






'''
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
    #rects1 = ax.bar(x - 0.2, obs_data, width/3, color='blue', label='Obs')
    #rects2 = ax.bar(x      , np.sum(dfl.iloc[1:5].values,axis=0), width/3, color='orange', label='Default')
    #rects3 = ax.bar(x + 0.2, np.sum(ctl.iloc[1:5].values,axis=0), width/3, color='green', label='Best')

    rects1 = ax.bar(x - 0.15, np.sum(dfl.iloc[4:8].values,axis=0), width/2, color='orange', label='Ctl')
    rects2 = ax.bar(x + 0.15, np.sum(ctl.iloc[4:8].values,axis=0), width/2, color='green', label='Best')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Water Balance Element $mm y^{-1}$')
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.savefig('../plots/water_balance_2014_def-best', bbox_inches='tight',pad_inches=0.1)
'''
