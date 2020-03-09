#!/usr/bin/env python

"""
Calculate metrics for every simulations

"""

__author__ = "MU Mengyuan"
__version__ = "2020-03-07"

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import griddata
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from plot_eucface_get_var import *

def calc_metrics(fcable, case_name, ring, layer):

    subs_Esoil = read_obs_esoil(ring)
    print(subs_Esoil)
    subs_Trans = read_obs_trans(ring)
    print(subs_Trans)
    subs_tdr   = read_obs_swc_tdr(ring)
    print(subs_tdr)
    subs_neo   = read_obs_neo_top_mid_bot(ring)
    print(subs_neo)
    subs_cable = read_ET_SM_top_mid_bot(fcable, ring, layer)
    print(subs_cable)

    # unify dates
    Esoil_obs   = subs_Esoil['obs'].loc[np.all([subs_Esoil.index.isin(subs_Trans.index),subs_Esoil.index.isin(subs_cable.index)],axis=0)]
    Trans_obs   = subs_Trans['obs'].loc[np.all([subs_Trans.index.isin(subs_Esoil.index),subs_Trans.index.isin(subs_cable.index)],axis=0)]
    Esoil_cable = subs_cable["ESoil"].loc[np.all([subs_cable.index.isin(subs_Esoil.index),subs_cable.index.isin(subs_Trans.index)],axis=0)]
    Trans_cable = subs_cable["TVeg"].loc[np.all([subs_cable.index.isin(subs_Esoil.index),subs_cable.index.isin(subs_Trans.index)],axis=0)]

    mask           = np.any([np.isnan(Esoil_obs), np.isnan(Trans_obs)],axis=0)
    Esoil_obs      = Esoil_obs[mask == False]
    Trans_obs      = Trans_obs[mask == False]
    Esoil_cable    = Esoil_cable[mask == False]
    Trans_cable    = Trans_cable[mask == False]

    SM_50cm_obs   = subs_tdr["obs"].loc[subs_tdr.index.isin(subs_cable.index)]
    SM_50cm_cable = subs_cable["SM_50cm"].loc[subs_cable.index.isin(subs_tdr.index)]

    SM_top_obs  = subs_neo["SM_top"].loc[subs_neo.index.isin(subs_cable.index)]
    SM_top_cable= subs_cable["SM_top"].loc[subs_cable.index.isin(subs_neo.index)]

    mask           = np.isnan(SM_top_obs)
    SM_top_obs     = SM_top_obs[mask == False]
    SM_top_cable   = SM_top_cable[mask == False]

    SM_mid_obs  = subs_neo["SM_mid"].loc[subs_neo.index.isin(subs_cable.index)]
    SM_mid_cable= subs_cable["SM_mid"].loc[subs_cable.index.isin(subs_neo.index)]

    mask           = np.isnan(SM_mid_obs)
    SM_mid_obs     = SM_mid_obs[mask == False]
    SM_mid_cable   = SM_mid_cable[mask == False]

    SM_bot_obs  = subs_neo["SM_bot"].loc[subs_neo.index.isin(subs_cable.index)]
    SM_bot_cable= subs_cable["SM_bot"].loc[subs_cable.index.isin(subs_neo.index)]

    mask          = np.isnan(SM_bot_obs)
    SM_bot_obs    = SM_bot_obs[mask == False]
    SM_bot_cable  = SM_bot_cable[mask == False]

    WA_all_obs    = subs_neo["WA_all"].loc[subs_neo.index.isin(subs_cable.index)]
    WA_all_cable  = subs_cable["WA_all"].loc[subs_cable.index.isin(subs_neo.index)]

    mask          = np.isnan(WA_all_obs)
    WA_all_obs    = WA_all_obs[mask == False]
    WA_all_cable  = WA_all_cable[mask == False]

    Esoil_r   = stats.pearsonr(Esoil_obs, Esoil_cable)[0]
    Esoil_MSE = mean_squared_error(Esoil_obs, Esoil_cable)

    Trans_r   = stats.pearsonr(Trans_obs, Trans_cable)[0]
    Trans_MSE = mean_squared_error(Trans_obs, Trans_cable)

    Esoil_Trans_r   = stats.pearsonr(Esoil_obs/Trans_obs,Esoil_cable/Trans_cable)[0]
    Esoil_Trans_MSE = mean_squared_error(Esoil_obs/Trans_obs,Esoil_cable/Trans_cable)
    SM_50cm_r       = stats.pearsonr(SM_50cm_obs.values, SM_50cm_cable.values)[0]
    SM_50cm_MSE     = mean_squared_error(SM_50cm_obs, SM_50cm_cable)
    SM_top_r        = stats.pearsonr(SM_top_obs, SM_top_cable)[0]
    SM_top_MSE      = mean_squared_error(SM_top_obs, SM_top_cable)
    SM_mid_r        = stats.pearsonr(SM_mid_obs, SM_mid_cable)[0]
    SM_mid_MSE      = mean_squared_error(SM_mid_obs, SM_mid_cable)
    SM_bot_r        = stats.pearsonr(SM_bot_obs, SM_bot_cable)[0]
    SM_bot_MSE      = mean_squared_error(SM_bot_obs, SM_bot_cable)
    WA_all_r        = stats.pearsonr(WA_all_obs, WA_all_cable)[0]
    WA_all_MSE      = mean_squared_error(WA_all_obs, WA_all_cable)

    return Esoil_r,   Trans_r,   Esoil_Trans_r,   SM_50cm_r,  SM_top_r,   SM_mid_r,   SM_bot_r,  WA_all_r, \
           Esoil_MSE, Trans_MSE, Esoil_Trans_MSE, SM_50cm_MSE,SM_top_MSE, SM_mid_MSE, SM_bot_MSE,WA_all_MSE;

def plotting(metrics,ring):

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

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    width = 1.
    im1 = ax1.imshow(metrics[:,0:8], interpolation='nearest')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(metrics[:,8:15], interpolation='nearest')
    fig.colorbar(im2, ax=ax2)


    fig.savefig("EucFACE_metrics_%s.png" % (ring), bbox_inches='tight', pad_inches=0.1)

def annual_value(fcable, case_name, ring, layer):

    """
    calculate annual water budget items, energy flux and soil status
    """

    if layer == "6":
        zse = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
    elif layer == "31uni":
        zse = [ 0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                0.15 ]

    cable = nc.Dataset(fcable, 'r')

    step_2_sec = 30.*60.

    df              = pd.DataFrame(cable.variables['Rainf'][:,0,0]*step_2_sec, columns=['Rainf']) # 'Rainfall+snowfall'
    df['Evap']      = cable.variables['Evap'][:,0,0]*step_2_sec   # 'Total evaporation'
    df['TVeg']      = cable.variables['TVeg'][:,0,0]*step_2_sec   # 'Vegetation transpiration'
    df['ESoil']     = cable.variables['ESoil'][:,0,0]*step_2_sec  # 'evaporation from soil'
    df['ECanop']    = cable.variables['ECanop'][:,0,0]*step_2_sec # 'Wet canopy evaporation'
    df['Qs']        = cable.variables['Qs'][:,0,0]*step_2_sec     # 'Surface runoff'
    df['Qsb']       = cable.variables['Qsb'][:,0,0]*step_2_sec    # 'Subsurface runoff'
    df['Qrecharge'] = cable.variables['Qrecharge'][:,0,0]*step_2_sec

    status              = pd.DataFrame(cable.variables['Qle'][:,0,0] , columns=['Qle'])   # 'Surface latent heat flux'
    status['Qh']        = cable.variables['Qh'][:,0,0]    # 'Surface sensible heat flux'
    status['Qg']        = cable.variables['Qg'][:,0,0]    # 'Surface ground heat flux'

    if layer == "6":

        status['SoilMoist_top']  = (  cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                    + cable.variables['SoilMoist'][:,1,0,0]*0.058\
                                    + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                    + cable.variables['SoilMoist'][:,3,0,0]*(0.3-0.022-0.058-0.154) )/0.3
        status['SoilMoist_mid']  = (  cable.variables['SoilMoist'][:,3,0,0]*0.343 \
                                    + cable.variables['SoilMoist'][:,4,0,0]*(1.2-0.343) )/1.2
        status['SoilMoist_bot']  = (  cable.variables['SoilMoist'][:,4,0,0]*(1.085-(1.2-0.343)) \
                                    + cable.variables['SoilMoist'][:,5,0,0]*2.872)/(4.6-1.5)

        status['SoilMoist_all'] = cable.variables['SoilMoist'][:,0,0,0]*zse[0]
        for i in np.arange(1,6):
            status['SoilMoist_all'] = status['SoilMoist_all'] + cable.variables['SoilMoist'][:,i,0,0]*zse[i]
        status['SoilMoist_all'] = status['SoilMoist_all']/sum(zse)

        status['SoilTemp_top']  = (   cable.variables['SoilTemp'][:,0,0,0]*0.022 \
                                    + cable.variables['SoilTemp'][:,1,0,0]*0.058\
                                    + cable.variables['SoilTemp'][:,2,0,0]*0.154 \
                                    + cable.variables['SoilTemp'][:,3,0,0]*(0.3-0.022-0.058-0.154) )/0.3\
                                  - 273.15
        status['SoilTemp_mid']  = (   cable.variables['SoilTemp'][:,3,0,0]*0.343 \
                                    + cable.variables['SoilTemp'][:,4,0,0]*(1.2-0.343) )/1.2\
                                  - 273.15
        status['SoilTemp_bot']  = (   cable.variables['SoilTemp'][:,4,0,0]*(1.085-(1.2-0.343)) \
                                    + cable.variables['SoilTemp'][:,5,0,0]*2.872)/(4.6-1.5) \
                                  - 273.15
        status['SoilTemp_all']  = cable.variables['SoilTemp'][:,0,0,0]*zse[0]
        for i in np.arange(1,6):
            status['SoilTemp_all']  = status['SoilTemp_all']  + cable.variables['SoilTemp'][:,i,0,0]*zse[i]
        status['SoilTemp_all']  = status['SoilTemp_all']/sum(zse) - 273.15

    elif layer == "31uni":

        status['SoilMoist_top']     = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                      + cable.variables['SoilMoist'][:,1,0,0]*0.15 )/0.3

        status['SoilMoist_mid']     = cable.variables['SoilMoist'][:,2,0,0]*0.15
        for i in np.arange(3,10):
            status['SoilMoist_mid'] = status['SoilMoist_mid'] + cable.variables['SoilMoist'][:,i,0,0]*0.15
        status['SoilMoist_mid']     = status['SoilMoist_mid'] /(1.5-0.3)

        status['SoilMoist_bot']     = cable.variables['SoilMoist'][:,10,0,0]*0.15
        for i in np.arange(11,30):
            status['SoilMoist_bot'] = status['SoilMoist_bot'] + cable.variables['SoilMoist'][:,i,0,0]*0.15
        status['SoilMoist_bot']     = (status['SoilMoist_bot'] + cable.variables['SoilMoist'][:,30,0,0]*0.1)/(4.6-1.5)

        status['SoilMoist_all']     = cable.variables['SoilMoist'][:,30,0,0]*0.1
        for i in np.arange(0,30):
            status['SoilMoist_all'] = status['SoilMoist_all'] + cable.variables['SoilMoist'][:,i,0,0]*zse[i]
        status['SoilMoist_all']     = status['SoilMoist_all']/4.6

        status['SoilTemp_top']      = ( cable.variables['SoilTemp'][:,0,0,0]*0.15 \
                                      + cable.variables['SoilTemp'][:,1,0,0]*0.15 )/0.3 \
                                      - 273.15
        status['SoilTemp_mid']  = cable.variables['SoilTemp'][:,2,0,0]*0.15
        for i in np.arange(3,10):
            status['SoilTemp_mid']  = status['SoilTemp_mid']  + cable.variables['SoilTemp'][:,i,0,0]*0.15
        status['SoilTemp_mid']    = status['SoilTemp_mid'] /(1.5-0.3) - 273.15

        status['SoilTemp_bot']   = cable.variables['SoilTemp'][:,10,0,0]*0.15
        for i in np.arange(11,30):
            status['SoilTemp_bot']   = status['SoilTemp_bot'] + cable.variables['SoilTemp'][:,i,0,0]*0.15
        status['SoilTemp_bot']   = (status['SoilTemp_bot'] + cable.variables['SoilTemp'][:,30,0,0]*0.1)/(4.6-1.5)\
                                    - 273.15

        status['SoilTemp_all']  = cable.variables['SoilTemp'][:,30,0,0]*0.1
        for i in np.arange(0,30):
            status['SoilTemp_all']  = status['SoilTemp_all']  + cable.variables['SoilTemp'][:,i,0,0]*zse[i]
        status['SoilTemp_all']     = status['SoilTemp_all']/4.6 - 273.15


    df['dates']     = nc.num2date(cable.variables['time'][:], cable.variables['time'].units)
    df              = df.set_index('dates')
    df              = df.resample("Y").agg('sum')

    status['dates']   = nc.num2date(cable.variables['time'][:], cable.variables['time'].units)
    status            = status.set_index('dates')
    status            = status.resample("Y").agg('mean')
    print(df)
    print(status)
    df     = df.iloc[0:6,:].mean(axis=0)
    status = status.iloc[0:6,:].mean(axis=0)
    print(df)
    print(status.shape)

    annual = np.zeros(19)
    for i in np.arange(19):
        if i <= 7:
            annual[i] = df.iloc[i]
        else:
            annual[i] = status.iloc[i-8]

    return annual;


if __name__ == "__main__":

    cases_6 = [
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6_litter",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_SM_6_litter"
              ]
    cases_31= [
              "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter",\
              "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x1-x1_litter",\
              "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter",\
              "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-exp",\
              "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_Hvrd",\
              "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-watpot"\
              ]
    rings  = ["amb"]#["R1","R2","R3","R4","R5","R6","amb","ele"]
    metrics= np.zeros((len(cases_6)+len(cases_31),16))
    annual = np.zeros((len(cases_6)+len(cases_31),19))
    for ring in rings:
        layer =  "6"
        for i,case_name in enumerate(cases_6):
            print(i)
            print(case_name)
            fcable ="%s/EucFACE_%s_out.nc" % (case_name, ring)
            metrics[i,:] = calc_metrics(fcable, case_name, ring, layer)
            annual[i,:]  = annual_value(fcable, case_name, ring, layer)
        print("i = %s" % str(i))
        j = i
        layer =  "31uni"
        for i,case_name in enumerate(cases_31):
            print(case_name)
            fcable ="%s/EucFACE_%s_out.nc" % (case_name, ring)
            metrics[i+j+1,:] = calc_metrics(fcable, case_name, ring, layer)
            annual[i+j+1,:]  = annual_value(fcable, case_name, ring, layer)
        #print(metrics)
        plotting(metrics,ring)
        #metrics.to_csv("EucFACE_amb_%slayers_%s_gw_on_or_on.csv" %(layers, case_name))
        np.savetxt("EucFACE_metrics_%s.csv" % (ring), metrics, delimiter=",")
        np.savetxt("EucFACE_annual_%s.csv" % (ring), annual, delimiter=",")
