#!/usr/bin/env python
"""
Purpose: read variable from observation data and cable output
"""

__author__  = "MU Mengyuan"
__version__ = "1.0 (2020-01-09)"
__email__   = "mu.mengyuan815@gmail.com"

import os
import sys
import glob
import shutil
import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime
from scipy.interpolate import griddata

def get_var_value(ref_var, output_file, layer, ring):
    print("carry on get_var_value")

    if ref_var == 'swc_25':
        cable_var = read_cable_swc_25cm(output_file, layer)
        obs_var   = read_obs_swc_tdr(ring)
    elif ref_var == 'swc_150':
        cable_var = read_SM_top_mid_bot(output_file, layer, ring)
        obs_var   = read_obs_neo_top_mid_bot(ring)
        print(cable_var)
        print(obs_var)
    elif ref_var == 'swc_all':
        cable_var = read_cable_swc_all(output_file, layer)
        obs_var   = read_obs_swc_neo(ring)
    elif ref_var == 'trans':
        cable_var = read_cable_var(output_file, 'TVeg')
        obs_var   = read_obs_trans(ring)
    elif ref_var == 'esoil':
        cable_var = read_cable_var(output_file, 'ESoil')
        obs_var   = read_obs_esoil(ring)
    elif ref_var == 'esoil2trans':
        cable_var = calc_cable_esoil2trans(output_file)
        obs_var   = calc_obs_esoil2trans(ring)

    return get_same_dates(cable_var, obs_var)

def get_same_dates(cable_var, obs_var):
    print("carry on get_same_dates")
    cable_var = cable_var['cable'].loc[cable_var.index.isin(obs_var.index)]
    obs_var   = obs_var['obs'].loc[obs_var.index.isin(cable_var.index)]
    mask      = np.any([np.isnan(cable_var), np.isnan(obs_var)],axis=0)

    cable_var = cable_var[mask == False]
    obs_var   = obs_var[mask == False]
    print(cable_var, obs_var)

    return cable_var, obs_var

def read_cable_swc_25cm(output_file, layer):

    """
    read the average swc in top 25cm from CABLE output
    """
    print("carry on read_cable_swc_25cm")

    cable = nc.Dataset(output_file, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['cable'])

    if layer == "6":
        SoilMoist['cable'] = (  cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.25-0.022-0.058-0.154) )/0.25
    elif layer == "31uni":
        SoilMoist['cable'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.10 )/0.25

    SoilMoist['Date'] = Time
    SoilMoist = SoilMoist.set_index('Date')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.sort_values(by=['Date'])

    print(SoilMoist)
    """
    The difference between SoilMoist['cable'] and SoilMoist is that
    SoilMoist has the column name "cable", but SoilMoist['cable'] doesn't.
    Both of them have 'dates' index
    """
    return SoilMoist

def read_SM_top_mid_bot(fcable, layer, ring):
    """
    Read CABLE ET and oil moisture for top mid bot blocks used in metrics calculation

    """
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    cable_data = pd.DataFrame(cable.variables['TVeg'][:,0,0]*1800., columns=['TVeg'])

    if layer == "6":
        '''
        cable_data['SM_top']  = (  cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058\
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.3-0.022-0.058-0.154) )/0.3
        cable_data['SM_mid']  = (  cable.variables['SoilMoist'][:,3,0,0]*0.343 \
                                 + cable.variables['SoilMoist'][:,4,0,0]*(1.2-0.343) )/1.2
        cable_data['SM_bot']  = (  cable.variables['SoilMoist'][:,4,0,0]*(1.085-(1.2-0.343)) \
                                 + cable.variables['SoilMoist'][:,5,0,0]*2.872)/(4.6-1.5)
        cable_data['SM_all'] = (   cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.409 \
                                 + cable.variables['SoilMoist'][:,4,0,0]*1.085 \
                                 + cable.variables['SoilMoist'][:,5,0,0]*2.872  )/4.6
        '''
        # SM_15m
        cable_data['cable']  = (  cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.409 \
                                 + cable.variables['SoilMoist'][:,4,0,0]*      \
                                 (1.5-0.022-0.058-0.154-0.409) )/1.5
    elif layer == "31uni":
        print("come in")
        '''
        cable_data['SM_top']  = (cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.15)/0.3
        cable_data['SM_mid']  = cable.variables['SoilMoist'][:,2,0,0]*0.15
        for i in np.arange(3,10):
            cable_data['SM_mid']  = cable_data['SM_mid'] + cable.variables['SoilMoist'][:,i,0,0]*0.15
        cable_data['SM_mid']  = cable_data['SM_mid']/(1.5-0.3)

        cable_data['SM_bot']  = cable.variables['SoilMoist'][:,10,0,0]*0.15
        for i in np.arange(11,30):
            cable_data['SM_bot']  = cable_data['SM_bot'] + cable.variables['SoilMoist'][:,i,0,0]*0.15
        cable_data['SM_bot']  = (cable_data['SM_bot'] + cable.variables['SoilMoist'][:,30,0,0]*0.1)/(4.6-1.5)

        cable_data['SM_all']  = cable.variables['SoilMoist'][:,30,0,0]*0.1
        for i in np.arange(0,30):
            cable_data['SM_all']  = cable_data['SM_all'] + cable.variables['SoilMoist'][:,i,0,0]*0.15
        cable_data['SM_all'] = cable_data['SM_all']/4.6
        '''
        # SM_15m
        cable_data['cable']  = cable.variables['SoilMoist'][:,0,0,0]*0.15
        for i in np.arange(1,10):
            cable_data['cable']  = cable_data['cable']+ cable.variables['SoilMoist'][:,i,0,0]*0.15
        cable_data['cable']  = cable_data['cable']/1.5

    cable_data['dates'] = Time
    cable_data = cable_data.set_index('dates')
    cable_data = cable_data.resample("D").agg('mean')
    cable_data.index = cable_data.index - pd.datetime(2011,12,31)
    cable_data.index = cable_data.index.days
    cable_data = cable_data.sort_values(by=['dates'])
    print("================")
    print(cable_data)

    return cable_data

def read_cable_swc_all(output_file, layer):

    """
    read swc from CABLE output and calculate the average swc of the whole soil columns
    """

    print("carry on read_cable_swc_all")

    cable = nc.Dataset(output_file, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['cable'])

    SoilMoist['cable'][:] = 0.

    if layer == "6":
        zse       = [ 0.022, 0.058, 0.154, 0.409, 1.085, 2.872 ]
    elif layer == "31uni":
        zse       = [ 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015, 0.015, 0.015, 0.015, 0.015, 0.015, \
                      0.015 ]
    elif layer == "31exp":
        zse       = [ 0.020440, 0.001759, 0.003957, 0.007035, 0.010993, 0.015829,\
                      0.021546, 0.028141, 0.035616, 0.043971, 0.053205, 0.063318,\
                      0.074311, 0.086183, 0.098934, 0.112565, 0.127076, 0.142465,\
                      0.158735, 0.175883, 0.193911, 0.212819, 0.232606, 0.253272,\
                      0.274818, 0.297243, 0.320547, 0.344731, 0.369794, 0.395737,\
                      0.422559 ]
    elif layer == "31para":
        zse       = [ 0.020000, 0.029420, 0.056810, 0.082172, 0.105504, 0.126808,\
                      0.146083, 0.163328, 0.178545, 0.191733, 0.202892, 0.212023,\
                      0.219124, 0.224196, 0.227240, 0.228244, 0.227240, 0.224196,\
                      0.219124, 0.212023, 0.202892, 0.191733, 0.178545, 0.163328,\
                      0.146083, 0.126808, 0.105504, 0.082172, 0.056810, 0.029420,\
                      0.020000 ]
    for i in np.arange(len(zse)):
        SoilMoist['cable'][:] =  SoilMoist['cable'][:] + cable.variables['SoilMoist'][:,i,0,0]*zse[i]

    SoilMoist['cable'][:] = SoilMoist['cable'][:]/sum(zse)

    SoilMoist['Date'] = Time
    SoilMoist = SoilMoist.set_index('Date')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.sort_values(by=['Date'])

    print(SoilMoist)
    return SoilMoist

def read_cable_var(output_file, var_name):

    """
    read transpiration or soil evaporation from CABLE output
    """

    print("carry on read_cable_var")
    cable = nc.Dataset(output_file, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    var = pd.DataFrame(cable.variables[var_name][:,0,0]*1800., columns=['cable'])

    var['Date'] = Time
    var = var.set_index('Date')
    var = var.resample("D").agg('sum')
    var.index = var.index - pd.datetime(2011,12,31)
    var.index = var.index.days
    var = var.sort_values(by=['Date'])
    print(var)

    return var

def calc_cable_esoil2trans(output_file):

    """
    calculate the ratio of esoil to trans of CABLE output
    """

    print("carry on calc_cable_esoil2trans")

    Esoil_cable = read_cable_var(output_file, "ESoil")
    Trans_cable = read_cable_var(output_file, "TVeg")

    Esoil_2_Trans = Esoil_cable
    Esoil_2_Trans['cable'] = Esoil_cable['cable']/Trans_cable['cable']
    print(Esoil_2_Trans)

    return Esoil_2_Trans

def read_obs_swc_tdr(ring):

    """
    read the 25 cm swc from tdr observation
    """

    print("carry on read_obs_swc_tdr")

    fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
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

    subset['swc.tdr'] = subset['swc.tdr'].clip(lower=0.)
    subset['swc.tdr'] = subset['swc.tdr'].replace(0., float('nan'))
    subset = subset.rename({'swc.tdr': 'obs'}, axis='columns')
    print(subset)

    return subset

def read_obs_swc_neo(ring):

    """
    read the neo swc observation and calculate the soil columns average
    """

    print("carry on read_obs_swc_neo")
    fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    neo = pd.read_csv(fobs, usecols = ['Ring','Depth','Date','VWC'])

    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)
    neo['Date'] = neo['Date'] - pd.datetime(2011,12,31)
    neo['Date'] = neo['Date'].dt.days
    neo = neo.sort_values(by=['Date','Depth'])

    if ring == 'amb':
        subset = neo[(neo['Ring'].isin(['R2','R3','R6'])) & (neo.Date > 366)]
    elif ring == 'ele':
        subset = neo[(neo['Ring'].isin(['R1','R4','R5'])) & (neo.Date > 366)]
    else:
        subset = neo[(neo['Ring'].isin(['Ring'])) & (neo.Date > 366)]
    print("------", subset)
    subset = subset.groupby(by=["Depth","Date"]).mean()
    #subset = subset.xs('VWC', axis=1, drop_level=True)
    subset[:] = subset[:]/100.
    subset['VWC'] = subset['VWC'].clip(lower=0.)
    subset['VWC'] = subset['VWC'].replace(0., float('nan'))

    zse_obs = [0.375, 0.25, 0.25, 0.25, 0.25, 0.375,\
               0.5, 0.5, 0.5, 0.5, 0.5, 0.35 ]
    layer_cm = [25, 50, 75, 100, 125, 150, 200, 250,\
                300, 350, 400, 450]

    neo_obs = subset.loc[25]

    neo_obs['VWC'][:] = 0.
    for i in np.arange(len(zse_obs)):
        print("i = ", i )
        print(subset.loc[layer_cm[i]]['VWC'])
        neo_obs['VWC'][:] = neo_obs['VWC'][:] + subset.loc[layer_cm[i]]['VWC']*zse_obs[i]
    neo_obs['VWC'][:] = neo_obs['VWC'][:]/4.6

    neo_obs = neo_obs.rename({'VWC' : 'obs'}, axis='columns')
    print(neo_obs)

    return neo_obs

def read_obs_neo_top_mid_bot(ring):

    """
    Read neo soil moisture for top mid and bot soil blocks used for metrics calculation
    """
    fobs_neo = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    neo = pd.read_csv(fobs_neo, usecols = ['Ring','Depth','Date','VWC'])
    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)
    neo['Date'] = neo['Date'] - pd.datetime(2011,12,31)
    neo['Date'] = neo['Date'].dt.days
    neo = neo.sort_values(by=['Date','Depth'])

    if ring == 'amb':
        subset = neo[neo['Ring'].isin(['R2','R3','R6'])]
    elif ring == 'ele':
        subset = neo[neo['Ring'].isin(['R1','R4','R5'])]
    else:
        subset = neo[neo['Ring'].isin([ring])]

    subset = subset.groupby(by=["Depth","Date"]).mean()
    subset = subset.xs('VWC', axis=1, drop_level=True)
    x     = subset.index.get_level_values(1).values
    y     = subset.index.get_level_values(0).values
    value = subset.values

    X     = subset[(25)].index.values[20:]
    Y     = np.arange(0.5,460,1)

    grid_X, grid_Y = np.meshgrid(X,Y)

    grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='nearest')

    neo_data = pd.DataFrame(subset[(25)].index.values[20:], columns=['dates'])
    '''
    neo_data["SM_top"] = np.mean(grid_data[0:30,:],axis=0)/100.
    neo_data["SM_mid"]  = np.mean(grid_data[30:150,:],axis=0)/100.
    neo_data["SM_bot"] = np.mean(grid_data[150:460,:],axis=0)/100.
    neo_data["WA_all"] = np.sum(grid_data[:,:]/100.*10.,axis=0)
    '''
    # SM_15m
    neo_data["obs"] = np.mean(grid_data[0:150,:],axis=0)/100.
    neo_data = neo_data.set_index('dates')
    print(neo_data)
    return neo_data

def read_obs_trans(ring):

    """
    read transpiration from observation, in G 2016
    """

    print("carry on read_obs_trans")

    fobs = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_sapflow.csv"
    est_trans = pd.read_csv(fobs, usecols = ['Ring','Date','volRing'])
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

    print(subs)

    return subs

def read_obs_esoil(ring):

    """
    read soil evaporation from observation, in G 2016
    """

    print("carry on read_obs_esoil")

    fobs = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_underET.csv"
    est_esoil = pd.read_csv(fobs, usecols = ['Ring','Date','wuTP'])
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
    print(subs)

    return subs

def calc_obs_esoil2trans(ring):

    """
    calculate the ratio of esoil to trans from observation
    """

    print("carry on calc_obs_esoil2trans")

    subs_Esoil = read_obs_esoil(ring)
    print(subs_Esoil)
    subs_Trans = read_obs_trans(ring)
    print(subs_Trans)

    # unify dates
    Esoil_obs = subs_Esoil.loc[subs_Esoil.index.isin(subs_Trans.index)]
    Trans_obs = subs_Trans.loc[subs_Trans.index.isin(subs_Esoil.index)]

    mask      = np.any([np.isnan(Esoil_obs['obs']), np.isnan(Trans_obs['obs'])],axis=0)
    Esoil_obs = Esoil_obs[mask == False]
    Trans_obs = Trans_obs[mask == False]

    Esoil_2_Trans = Esoil_obs
    Esoil_2_Trans['obs'] = Esoil_obs['obs']/Trans_obs['obs']

    print(Esoil_2_Trans)

    return Esoil_2_Trans
