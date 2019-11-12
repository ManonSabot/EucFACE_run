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

    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    Rainf.index = Rainf.index.days
    rain        = Rainf['Rainf'].loc[Rainf.index.isin(subset.index)]

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
    soilmoist     = SoilMoist['SoilMoist'].loc[SoilMoist.index.isin(subset.index)]
    soilmoist_tdr = subset['swc.tdr'].loc[subset.index.isin(SoilMoist.index)]

    ESoil = pd.DataFrame(cable.variables['ESoil'][:,0,0],columns=['ESoil'])
    ESoil = ESoil*1800.
    ESoil['dates'] = Time
    ESoil = ESoil.set_index('dates')
    ESoil = ESoil.resample("D").agg('sum')
    ESoil.index = ESoil.index - pd.datetime(2011,12,31)
    ESoil.index = ESoil.index.days
    esoil       = ESoil['ESoil'].loc[ESoil.index.isin(subset.index)]
    esoil_tdr   = subset['Esoil'].loc[subset.index.isin(SoilMoist.index)]

    Evap = pd.DataFrame(cable.variables['Evap'][:,0,0],columns=['Evap'])
    Evap = Evap*1800.
    Evap['dates'] = Time
    Evap = Evap.set_index('dates')
    Evap = Evap.resample("D").agg('sum')
    Evap.index = Evap.index - pd.datetime(2011,12,31)
    Evap.index = Evap.index.days
    evap       = Evap['Evap'].loc[Evap.index.isin(subset.index)]

    # exclude tdr soilmoisture < 0 or tdr esoil < 0
    mask      = np.any([np.isnan(soilmoist_tdr), np.isnan(esoil_tdr)],axis=0)

    # exclude rainday and the after two days of rain
    day0      = np.zeros((len(rain)), dtype=bool)
    day1      = np.zeros((len(rain)), dtype=bool)
    day2      = np.zeros((len(rain)), dtype=bool)
    day3      = np.zeros((len(rain)), dtype=bool)
    day4      = np.zeros((len(rain)), dtype=bool)
    day5      = np.zeros((len(rain)), dtype=bool)
    day6      = np.zeros((len(rain)), dtype=bool)
    day7      = np.zeros((len(rain)), dtype=bool)

    print(day0)
    for i in np.arange(0,len(rain)-7):
        a = np.all([rain.values[i+1] == 0., rain.values[i+2] == 0., rain.values[i+3] == 0.,\
                    rain.values[i+4] == 0., rain.values[i+5] == 0., rain.values[i+6] == 0.,\
                    rain.values[i+7] == 0.])
        b = np.any([np.isnan(soilmoist_tdr.values[i+1]), np.isnan(soilmoist_tdr.values[i+2]), np.isnan(soilmoist_tdr.values[i+3]),\
                    np.isnan(soilmoist_tdr.values[i+4]), np.isnan(soilmoist_tdr.values[i+5]), np.isnan(soilmoist_tdr.values[i+6]),\
                    np.isnan(soilmoist_tdr.values[i+7])])
        c = np.any([np.isnan(esoil_tdr.values[i+1]), np.isnan(esoil_tdr.values[i+2]), np.isnan(esoil_tdr.values[i+3]),\
                    np.isnan(esoil_tdr.values[i+4]), np.isnan(esoil_tdr.values[i+5]), np.isnan(esoil_tdr.values[i+6]),\
                    np.isnan(esoil_tdr.values[i+7])])

        if (rain.values[i] > 0.5 and a and (not b) and (not c)):
            day0[i]   = True
            day1[i+1] = True
            day2[i+2] = True
            day3[i+3] = True
            day4[i+4] = True
            day5[i+5] = True
            day6[i+6] = True
            day7[i+7] = True


    esoil_rn        = np.zeros((len(esoil[day0 == True]),8))
    esoil_tdr_rn    = np.zeros((len(esoil[day0 == True]),8))
    soilmoist_rn    = np.zeros((len(esoil[day0 == True]),8))
    soilmoist_tdr_rn= np.zeros((len(esoil[day0 == True]),8))
    evap_rn         = np.zeros((len(esoil[day0 == True]),8))
    print("____________________________________________")
    print(len(esoil[day0 == True]))
    print("____________________________________________")
    esoil_rn[:,0] = esoil[day0 == True]
    esoil_rn[:,1] = esoil[day1 == True]
    esoil_rn[:,2] = esoil[day2 == True]
    esoil_rn[:,3] = esoil[day3 == True]
    esoil_rn[:,4] = esoil[day4 == True]
    esoil_rn[:,5] = esoil[day5 == True]
    esoil_rn[:,6] = esoil[day6 == True]
    esoil_rn[:,7] = esoil[day7 == True]

    esoil_tdr_rn[:,0] = esoil_tdr[day0 == True]
    esoil_tdr_rn[:,1] = esoil_tdr[day1 == True]
    esoil_tdr_rn[:,2] = esoil_tdr[day2 == True]
    esoil_tdr_rn[:,3] = esoil_tdr[day3 == True]
    esoil_tdr_rn[:,4] = esoil_tdr[day4 == True]
    esoil_tdr_rn[:,5] = esoil_tdr[day5 == True]
    esoil_tdr_rn[:,6] = esoil_tdr[day6 == True]
    esoil_tdr_rn[:,7] = esoil_tdr[day7 == True]

    soilmoist_rn[:,0] = soilmoist[day0 == True]
    soilmoist_rn[:,1] = soilmoist[day1 == True] - soilmoist_rn[:,0]
    soilmoist_rn[:,2] = soilmoist[day2 == True] - soilmoist_rn[:,0]
    soilmoist_rn[:,3] = soilmoist[day3 == True] - soilmoist_rn[:,0]
    soilmoist_rn[:,4] = soilmoist[day4 == True] - soilmoist_rn[:,0]
    soilmoist_rn[:,5] = soilmoist[day5 == True] - soilmoist_rn[:,0]
    soilmoist_rn[:,6] = soilmoist[day6 == True] - soilmoist_rn[:,0]
    soilmoist_rn[:,7] = soilmoist[day7 == True] - soilmoist_rn[:,0]
    soilmoist_rn[:,0] = soilmoist[day0 == True] - soilmoist_rn[:,0]

    soilmoist_tdr_rn[:,0] = soilmoist_tdr[day0 == True]
    soilmoist_tdr_rn[:,1] = soilmoist_tdr[day1 == True] - soilmoist_tdr_rn[:,0]
    soilmoist_tdr_rn[:,2] = soilmoist_tdr[day2 == True] - soilmoist_tdr_rn[:,0]
    soilmoist_tdr_rn[:,3] = soilmoist_tdr[day3 == True] - soilmoist_tdr_rn[:,0]
    soilmoist_tdr_rn[:,4] = soilmoist_tdr[day4 == True] - soilmoist_tdr_rn[:,0]
    soilmoist_tdr_rn[:,5] = soilmoist_tdr[day5 == True] - soilmoist_tdr_rn[:,0]
    soilmoist_tdr_rn[:,6] = soilmoist_tdr[day6 == True] - soilmoist_tdr_rn[:,0]
    soilmoist_tdr_rn[:,7] = soilmoist_tdr[day7 == True] - soilmoist_tdr_rn[:,0]
    soilmoist_tdr_rn[:,0] = soilmoist_tdr[day0 == True] - soilmoist_tdr_rn[:,0]

    evap_rn[:,0] = evap[day0 == True]
    evap_rn[:,1] = evap[day1 == True]/evap_rn[:,0]
    evap_rn[:,2] = evap[day2 == True]/evap_rn[:,0]
    evap_rn[:,3] = evap[day3 == True]/evap_rn[:,0]
    evap_rn[:,4] = evap[day4 == True]/evap_rn[:,0]
    evap_rn[:,5] = evap[day5 == True]/evap_rn[:,0]
    evap_rn[:,6] = evap[day6 == True]/evap_rn[:,0]
    evap_rn[:,7] = evap[day7 == True]/evap_rn[:,0]
    evap_rn[:,0] = evap[day0 == True]/evap_rn[:,0]

    return esoil_rn,esoil_tdr_rn,soilmoist_rn,soilmoist_tdr_rn,evap_rn;

if __name__ == "__main__":

    case_name = ["default-met_only_or-off",\
                 "ctl_met_LAI_vrt_SM_swilt-watr_31uni_HDM_or-off-litter_Hvrd"]

    rings = ["R1","R2","R3","R4","R5","R6","amb","ele"]

    for ring in rings:
        layer =  "6"
        fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[0], ring)
        esoil_rn1, esoil_tdr_rn1, soilmoist_rn1, soilmoist_tdr_rn1, evap_rn1 = \
                      main(fobs, fcable, case_name[0], ring, layer)

        layer =  "31uni"
        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[1], ring)
        esoil_rn2, esoil_tdr_rn2, soilmoist_rn2, soilmoist_tdr_rn2, evap_rn2 = \
                      main(fobs, fcable, case_name[1], ring, layer)

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

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        #ax1.scatter(lct1, esoil_rn1, s=2., marker='o', c='orange', label="def")
        #ax1.scatter(lct1, esoil_rn2, s=2., marker='o', c='green', label="imp")
        #ax1.scatter(lct1, esoil_tdr_rn1, s=2., marker='o', c='gray', label="obs")

        lct = [0,1,2,3,4,5,6,7]
        soilmoist1 = np.mean(soilmoist_rn1, axis=0)
        soilmoist2 = np.mean(soilmoist_rn2, axis=0)
        soilmoist_tdr = np.mean(soilmoist_tdr_rn1, axis=0)

        soilmoist_std1    = np.std(soilmoist_rn1,   axis=0, ddof = 1)
        soilmoist_std2    = np.std(soilmoist_rn2,   axis=0, ddof = 1)
        soilmoist_tdr_std = np.std(soilmoist_tdr_rn1,axis=0, ddof = 1)

        evap1 = np.mean(evap_rn1, axis=0)
        evap2 = np.mean(evap_rn2, axis=0)
        evap_std1 = np.std(evap_rn1, axis=0, ddof = 1)
        evap_std2 = np.std(evap_rn2, axis=0, ddof = 1)


        ax1.plot(lct, soilmoist1, c='orange', label="def")
        ax1.plot(lct, soilmoist2, c='green', label="imp")
        ax1.plot(lct, soilmoist_tdr, c='blue', label="obs")
        ax1.fill_between(lct, soilmoist1 - soilmoist_std1, soilmoist1 + soilmoist_std1,\
                        alpha=0.2, edgecolor='', facecolor='orange')
        ax1.fill_between(lct, soilmoist2 - soilmoist_std2, soilmoist2 + soilmoist_std2,\
                        alpha=0.2, edgecolor='', facecolor='green')
        ax1.fill_between(lct, soilmoist_tdr - soilmoist_tdr_std, soilmoist_tdr + soilmoist_tdr_std,\
                        alpha=0.2, edgecolor='', facecolor='blue')

        ax2.plot(lct, evap1, c='orange', label="def")
        ax2.plot(lct, evap2, c='green', label="imp")
        ax2.fill_between(lct, evap1 - evap_std1, evap1 + evap_std1, \
                alpha=0.2, edgecolor='', facecolor='orange')
        ax2.fill_between(lct, evap2 - evap_std2, evap2 + evap_std2, \
                alpha=0.2, edgecolor='', facecolor='green')

        '''
        ax1.scatter(lct, soilmoist1, marker='o', c='',edgecolors='orange', label="def") # s=2.,
        ax1.scatter(lct, soilmoist2, marker='o', c='',edgecolors='green', label="imp")
        ax1.scatter(lct, soilmoist_tdr, marker='o', c='',edgecolors='blue', label="obs")

        ax2.scatter(lct, evap1, marker='o', c='',edgecolors='orange',label="def")
        ax2.scatter(lct, evap2, marker='o', c='',edgecolors='green',label="imp")
        '''
        ax1.set_xlim(-0.5,7.5)
        ax2.set_xlim(-0.5,7.5)
        # ax1.set_ylim(-0.1,1.)
        ax1.legend()
        ax2.legend()
        fig.savefig("EucFACE_ET-SM_after_rain_%s.png" % (ring),bbox_inches='tight')#, pad_inches=0.1)
