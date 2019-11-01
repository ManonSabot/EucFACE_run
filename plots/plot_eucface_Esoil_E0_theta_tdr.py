#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-10-06"
__changefrom__ = 'plot_eucface_swc_cable_vs_obs.py'

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

def main(fobs, fcable, case_name, ring, layer):

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
    subset['swc.tdr']   = subset['swc.tdr'].clip(lower=0.)
    subset['swc.tdr']   = subset['swc.tdr'].replace(0., float('nan'))
    subset['Esoil']     = np.zeros(len(subset))
    subset['Esoil'][1:] = (subset['swc.tdr'].values[1:] - subset['swc.tdr'].values[:-1])*(-500.)
    subset['Esoil']     = subset['Esoil'].clip(lower=0.)
    subset['Esoil']     = subset['Esoil'].replace(0., float('nan'))
    #subset = subset.xs('swc.tdr', axis=1, drop_level=True)
    print(subset)

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    if layer == "6":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.5-0.022-0.058-0.154) )/0.5
    elif layer == "13":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.02 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.05 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.06 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.13 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.5-0.02-0.05-0.06-0.13) )/0.5
    elif layer == "31uni":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.05 )/0.5
    elif layer == "31exp":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.020440 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.001759 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.003957 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.007035 \
                                 + cable.variables['SoilMoist'][:,4,0,0]*0.010993 \
                                 + cable.variables['SoilMoist'][:,5,0,0]*0.015829 \
                                 + cable.variables['SoilMoist'][:,6,0,0]*0.021546 \
                                 + cable.variables['SoilMoist'][:,7,0,0]*0.028141 \
                                 + cable.variables['SoilMoist'][:,8,0,0]*0.035616 \
                                 + cable.variables['SoilMoist'][:,9,0,0]*0.043971 \
                                 + cable.variables['SoilMoist'][:,10,0,0]*0.053205 \
                                 + cable.variables['SoilMoist'][:,11,0,0]*0.063318 \
                                 + cable.variables['SoilMoist'][:,12,0,0]*0.074311 \
                                 + cable.variables['SoilMoist'][:,13,0,0]*0.086183 \
                                 + cable.variables['SoilMoist'][:,14,0,0]*(0.5-0.466304))/0.5
    elif layer == "31para":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.020440 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.001759 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.003957 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.007035 \
                                 + cable.variables['SoilMoist'][:,4,0,0]*0.010993 \
                                 + cable.variables['SoilMoist'][:,5,0,0]*0.015829 \
                                 + cable.variables['SoilMoist'][:,6,0,0]*(0.5-0.420714))/0.5

    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.sort_values(by=['dates'])

    ESoil = pd.DataFrame(cable.variables['ESoil'][:,0,0],columns=['ESoil'])
    ESoil = ESoil*1800.
    ESoil['dates'] = Time
    ESoil = ESoil.set_index('dates')
    ESoil = ESoil.resample("D").agg('sum')
    ESoil.index = ESoil.index - pd.datetime(2011,12,31)
    ESoil.index = ESoil.index.days
    print(ESoil)

    Rnet = pd.DataFrame(cable.variables['Rnet'][:,0,0],columns=['Rnet'])
    Rnet['dates'] = Time
    Rnet = Rnet.set_index('dates')
    Rnet = Rnet.resample("D").agg('mean')
    Rnet.index = Rnet.index - pd.datetime(2011,12,31)
    Rnet.index = Rnet.index.days
    print(Rnet*86400/2454000)

    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    Rainf.index = Rainf.index.days

    rain          = Rainf['Rainf'].loc[Rainf.index.isin(subset.index)]
    esoil         = ESoil['ESoil'].loc[ESoil.index.isin(subset.index)]
    rnet          = Rnet['Rnet'].loc[Rnet.index.isin(subset.index)]
    soilmoist     = SoilMoist['SoilMoist'].loc[SoilMoist.index.isin(subset.index)]
    esoil_tdr     = subset['Esoil'].loc[subset.index.isin(SoilMoist.index)]
    soilmoist_tdr = subset['swc.tdr'].loc[subset.index.isin(SoilMoist.index)]

    # exclude tdr soilmoisture < 0 or tdr esoil < 0
    mask      = np.any([np.isnan(soilmoist_tdr), np.isnan(esoil_tdr)],axis=0)
    print(mask)
    rain      = rain[mask == False]
    esoil     = esoil[mask == False]
    rnet      = rnet[mask == False]
    soilmoist = soilmoist[mask == False]
    esoil_tdr = esoil_tdr[mask == False]
    soilmoist_tdr = soilmoist_tdr[mask == False]
    print("any(rain>0.)")
    print(np.any(rain>0.))

    # exclude rainday and the after two days of rain
    mask      = np.ones((len(rain)), dtype=bool)
    print(rain)
    if rain.values[0] > 0. :
        mask[0] = False
    if rain.values[0] > 0. or rain.values[1] > 0.:
        mask[1] = False
    for i in np.arange(2,len(rain)):
        if rain.values[i] > 0. or rain.values[i-1] > 0. or rain.values[i-2] > 0. :
            mask[i] = False
    rain      = rain[mask == True]
    esoil     = esoil[mask == True]
    rnet      = rnet[mask == True]
    soilmoist = soilmoist[mask == True]
    esoil_tdr = esoil_tdr[mask == True]
    soilmoist_tdr = soilmoist_tdr[mask == True]
    print("any(rain>0.)")
    print(np.any(rain>0.))

    # exclude the days Rnet < 0.
    rnet = rnet.clip(lower=0.)
    rnet = rnet.replace(0., float('nan'))
    mask = np.isnan(rnet)

    esoil     = esoil[mask == False]
    rnet      = rnet[mask == False]
    soilmoist = soilmoist[mask == False]
    esoil_tdr = esoil_tdr[mask == False]
    soilmoist_tdr = soilmoist_tdr[mask == False]
    rate      = esoil/(rnet*86400/2454000)
    rate_tdr  = esoil_tdr/(rnet*86400/2454000)

    print("-------------------------------------------------")
    print(np.any(esoil < 0.))
    print(np.any(rnet < 0.))
    print(np.any(soilmoist < 0.))
    print(np.any(esoil_tdr < 0.))
    print(np.any(soilmoist_tdr < 0.))

    print(esoil)
    print(rnet)
    print(soilmoist)
    print(esoil_tdr)
    print(soilmoist_tdr)
    print(rate)
    print(rate_tdr)
    print("-------------------------------------------------")


# ____________________ Plot obs _______________________
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

    ax1 = fig.add_subplot(111)

    ax1.scatter(soilmoist, rate, s=2, marker='o', c='orange')
    ax1.scatter(soilmoist_tdr, rate_tdr, s=2, marker='o', c='green')
    ax1.set_ylim(-0.1,2.)

    fig.savefig("EucFACE_Esoil_E0_theta_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    layer =  "6"

    cases = ["ctl_met_LAI_vrt_SM",\
             "ctl_met_LAI_vrt_SM_swilt-watr", "ctl_met_LAI_vrt_SM_swilt-watr_Hvrd",\
             "ctl_met_LAI_vrt_SM_swilt-watr_Or-Off","default-met_only"]
    # 6
    # ["met_LAI_sand","met_LAI_clay","met_LAI_silt"\
    #  "ctl_met_LAI", "ctl_met_LAI_vrt", "ctl_met_LAI_vrt_SM",\
    #  "ctl_met_LAI_vrt_SM_swilt-watr", "ctl_met_LAI_vrt_SM_swilt-watr_Hvrd",\
    #  "ctl_met_LAI_vrt_SM_swilt-watr_Or-Off","default-met_only"]
    # 31para
    #["ctl_met_LAI_vrt_SM_swilt-watr_31para"]
    # 31exp
    #["ctl_met_LAI_vrt_SM_swilt-watr_31exp"]
    # 31uni
    #  ["ctl_met_LAI_vrt_SM_31uni","ctl_met_LAI_vrt_SM_swilt-watr_31uni",\
    #   "ctl_met_LAI_vrt_SM_swilt-watr_31uni_root-uni",\
    #   "ctl_met_LAI_vrt_SM_swilt-watr_31uni_root-log10"]

    rings = ["amb"]#["R1","R2","R3","R4","R5","R6","amb","ele"]
    for case_name in cases:
        for ring in rings:
            fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
            fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            main(fobs, fcable, case_name, ring, layer)




    '''


    TVeg = pd.DataFrame(cable.variables['TVeg'][:,0,0],columns=['TVeg'])
    TVeg = TVeg*1800.
    TVeg['dates'] = Time
    TVeg = TVeg.set_index('dates')
    TVeg = TVeg.resample("D").agg('sum')
    TVeg.index = TVeg.index - pd.datetime(2011,12,31)
    TVeg.index = TVeg.index.days

    Qrf = pd.DataFrame(cable.variables['Qs'][:,0,0],columns=['Qrf'])
    Qrf['Qrf'] = (cable.variables['Qs'][:,0,0]+ cable.variables['Qsb'][:,0,0])*1800.
    Qrf['dates'] = Time
    Qrf = Qrf.set_index('dates')
    Qrf = Qrf.resample("D").agg('sum')
    Qrf.index = Qrf.index - pd.datetime(2011,12,31)
    Qrf.index = Qrf.index.days

    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0],columns=['Tair'])
    Tair = Tair - 273.15
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')
    Tair = Tair.resample("D").agg('mean')
    Tair.index = Tair.index - pd.datetime(2011,12,31)
    Tair.index = Tair.index.days

    VegT = pd.DataFrame(cable.variables['VegT'][:,0,0],columns=['VegT'])
    VegT = VegT - 273.15
    VegT['dates'] = Time
    VegT = VegT.set_index('dates')
    VegT = VegT.resample("D").agg('mean')
    VegT.index = VegT.index - pd.datetime(2011,12,31)
    VegT.index = VegT.index.days

    Qair = pd.DataFrame(cable.variables['Qair'][:,0,0],columns=['Qair'])
    Qair['dates'] = Time
    Qair = Qair.set_index('dates')
    Qair = Qair.resample("D").agg('mean')
    Qair.index = Qair.index - pd.datetime(2011,12,31)
    Qair.index = Qair.index.days

    Wind = pd.DataFrame(cable.variables['Wind'][:,0,0],columns=['Wind'])
    Wind['dates'] = Time
    Wind = Wind.set_index('dates')
    Wind = Wind.resample("D").agg('mean')
    Wind.index = Wind.index - pd.datetime(2011,12,31)
    Wind.index = Wind.index.days

    Fwsoil = pd.DataFrame(cable.variables['Fwsoil'][:,0,0],columns=['Fwsoil'])
    Fwsoil['dates'] = Time
    Fwsoil = Fwsoil.set_index('dates')
    Fwsoil = Fwsoil.resample("D").agg('mean')
    Fwsoil.index = Fwsoil.index - pd.datetime(2011,12,31)
    Fwsoil.index = Fwsoil.index.days
    '''
