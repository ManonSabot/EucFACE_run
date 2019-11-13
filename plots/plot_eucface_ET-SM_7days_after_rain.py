#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-11-12"
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

def main(fobs, fcable, case_name, ring, layer, ss):

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
    #rain        = Rainf['Rainf'].loc[Rainf.index.isin(subset.index)]
    print(Rainf)
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
    #soilmoist     = SoilMoist['SoilMoist'].loc[SoilMoist.index.isin(subset.index)]
    #soilmoist_tdr = subset['swc.tdr'].loc[subset.index.isin(SoilMoist.index)]

    ESoil = pd.DataFrame(cable.variables['ESoil'][:,0,0],columns=['ESoil'])
    ESoil = ESoil*1800.
    ESoil['dates'] = Time
    ESoil = ESoil.set_index('dates')
    ESoil = ESoil.resample("D").agg('sum')
    ESoil.index = ESoil.index - pd.datetime(2011,12,31)
    ESoil.index = ESoil.index.days
    #esoil       = ESoil['ESoil'].loc[ESoil.index.isin(subset.index)]
    #esoil_tdr   = subset['Esoil'].loc[subset.index.isin(SoilMoist.index)]

    Evap = pd.DataFrame(cable.variables['Evap'][:,0,0],columns=['Evap'])
    Evap = Evap*1800.
    Evap['dates'] = Time
    Evap = Evap.set_index('dates')
    Evap = Evap.resample("D").agg('sum')
    Evap.index = Evap.index - pd.datetime(2011,12,31)
    Evap.index = Evap.index.days
    #evap       = Evap['Evap'].loc[Evap.index.isin(subset.index)]

    if ss > 0 :
        season = np.arange('2013-01-01', '2019-07-02', dtype='datetime64[D]')
        season = np.datetime_as_string(season, unit='D')
        Season = np.arange(0, len(season))
        print(season)
        for i in np.arange(0,len(season),1):
            if season[i][5:7] in ['01','02','12']:
                Season[i] = 1
            elif season[i][5:7] in ['03','04','05']:
                Season[i] = 2
            elif season[i][5:7] in ['06','07','08']:
                Season[i] = 3
            elif season[i][5:7] in ['09','10','11']:
                Season[i] = 4
        print(Season)
    # exclude rainday and the after two days of rain
    day      = np.zeros((len(Rainf),8), dtype=bool)
    for i in np.arange(2,len(Rainf)-7):
        a = np.all([Rainf.values[i+1] == 0., Rainf.values[i+2] == 0., Rainf.values[i+3] == 0.,\
                    Rainf.values[i+4] == 0., Rainf.values[i+5] == 0., Rainf.values[i+6] == 0.,\
                    Rainf.values[i+7] == 0.])
        b = np.any([np.isnan(subset['swc.tdr'].values[i+1]), np.isnan(subset['swc.tdr'].values[i+2]), np.isnan(subset['swc.tdr'].values[i+3]),\
                    np.isnan(subset['swc.tdr'].values[i+4]), np.isnan(subset['swc.tdr'].values[i+5]), np.isnan(subset['swc.tdr'].values[i+6]),\
                    np.isnan(subset['swc.tdr'].values[i+7]), np.isnan(subset['swc.tdr'].values[i])])
        c = np.any([np.isnan(subset['Esoil'].values[i+1]), np.isnan(subset['Esoil'].values[i+2]), np.isnan(subset['Esoil'].values[i+3]),\
                    np.isnan(subset['Esoil'].values[i+4]), np.isnan(subset['Esoil'].values[i+5]), np.isnan(subset['Esoil'].values[i+6]),\
                    np.isnan(subset['Esoil'].values[i+7]), np.isnan(subset['Esoil'].values[i])])
        d = Rainf.values[i] > 0.5 or (Rainf.values[i-1] > 0.5 and Rainf.values[i] > 0.0) or (Rainf.values[i-2] > 10. and Rainf.values[i] > 0.0)
        if ss > 0:
            e = Season[i] == ss
        else:
            e = True
        if (e and d and a and (not b) and (not c) ):
            print(np.isnan(subset['swc.tdr'].values[i:i+7]))
            print(np.isnan(subset['Esoil'].values[i:i+7]))
            day[i,0]   = True
            day[i+1,1] = True
            day[i+2,2] = True
            day[i+3,3] = True
            day[i+4,4] = True
            day[i+5,5] = True
            day[i+6,6] = True
            day[i+7,7] = True

    event = len(ESoil[day[:,0] == True])
    esoil_rn        = np.zeros((event,8))
    esoil_tdr_rn    = np.zeros((event,8))
    soilmoist_rn    = np.zeros((event,8))
    soilmoist_tdr_rn= np.zeros((event,8))
    evap_rn         = np.zeros((event,8))
    print("____________________________________________")
    print(event)
    print("____________________________________________")

    for i in np.arange(0,8):
        esoil_rn[:,i]     = ESoil['ESoil'][day[:,i] == True].values
        esoil_tdr_rn[:,i] = subset['Esoil'][day[:,i] == True].values

        if i == 0 :
            # the rainday
            soilmoist_rn[:,i]     = SoilMoist['SoilMoist'][day[:,i] == True].values
            soilmoist_tdr_rn[:,i] = subset['swc.tdr'][day[:,i] == True].values
            evap_rn[:,i]          = Evap['Evap'][day[:,i] == True].values
        else:
            # 10 days after rainday
            soilmoist_rn[:,i]     = SoilMoist['SoilMoist'][day[:,i] == True].values - soilmoist_rn[:,0]
            soilmoist_tdr_rn[:,i] = subset['swc.tdr'][day[:,i] == True].values  - soilmoist_tdr_rn[:,0]
            evap_rn[:,i]          = Evap['Evap'][day[:,i] == True].values/evap_rn[:,0]

    soilmoist_rn[:,0]     = 0.
    soilmoist_tdr_rn[:,0] = 0.
    evap_rn[:,0]          = 1.

    return esoil_rn,esoil_tdr_rn,soilmoist_rn,soilmoist_tdr_rn,evap_rn;

if __name__ == "__main__":

    case_name = ["default-met_only_or-off",\
                 "ctl_met_LAI_vrt_SM_swilt-watr_31uni_HDM_or-off-litter_Hvrd"]

    rings = ["amb"]#["R1","R2","R3","R4","R5","R6","amb","ele"]
    ss    = 0
            # 1 summer
            # 2 autumn
            # 3 winter
            # 4 spring
            # 0 year
    for ring in rings:
        layer =  "6"
        fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[0], ring)
        esoil_rn1, esoil_tdr_rn1, soilmoist_rn1, soilmoist_tdr_rn1, evap_rn1 = \
                      main(fobs, fcable, case_name[0], ring, layer, ss)

        layer =  "31uni"
        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[1], ring)
        esoil_rn2, esoil_tdr_rn2, soilmoist_rn2, soilmoist_tdr_rn2, evap_rn2 = \
                      main(fobs, fcable, case_name[1], ring, layer, ss)

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

        #ax1.scatter(lct1, esoil_rn1, s=2., marker='o', c='orange', label="def")
        #ax1.scatter(lct1, esoil_rn2, s=2., marker='o', c='green', label="imp")
        #ax1.scatter(lct1, esoil_tdr_rn1, s=2., marker='o', c='gray', label="obs")

        lct = [0,1,2,3,4,5,6,7]
        soilmoist1 = np.mean(soilmoist_rn1, axis=0)
        soilmoist2 = np.mean(soilmoist_rn2, axis=0)
        soilmoist_tdr = np.mean(soilmoist_tdr_rn1, axis=0)
        print('===========================')
        print(soilmoist_tdr_rn1)
        print('===========================')
        soilmoist_std1    = np.std(soilmoist_rn1,   axis=0, ddof = 1)
        soilmoist_std2    = np.std(soilmoist_rn2,   axis=0, ddof = 1)
        soilmoist_tdr_std = np.std(soilmoist_tdr_rn1,axis=0, ddof = 1)

        evap1 = np.mean(evap_rn1, axis=0)
        evap2 = np.mean(evap_rn2, axis=0)
        evap_std1 = np.std(evap_rn1, axis=0, ddof = 1)
        evap_std2 = np.std(evap_rn2, axis=0, ddof = 1)
        print('===========================')
        print(evap_rn1)
        print('===========================')

        ax1.plot(lct, soilmoist1,lw= 2., c='orange', label="def")
        ax1.plot(lct, soilmoist2, lw= 2., c='green', label="imp")
        ax1.plot(lct, soilmoist_tdr, lw= 2., c='blue', label="obs")
        #ax1.fill_between(lct, soilmoist1 - soilmoist_std1, soilmoist1 + soilmoist_std1,\
        #                alpha=0.2, edgecolor='', facecolor='orange')
        #ax1.fill_between(lct, soilmoist2 - soilmoist_std2, soilmoist2 + soilmoist_std2,\
        #                alpha=0.2, edgecolor='', facecolor='green')
        #ax1.fill_between(lct, soilmoist_tdr - soilmoist_tdr_std, soilmoist_tdr + soilmoist_tdr_std,\
        #                alpha=0.2, edgecolor='', facecolor='blue')

        ax2.plot(lct, evap1, lw= 2., c='orange', label="def")
        ax2.plot(lct, evap2, lw= 2., c='green', label="imp")
        #ax2.fill_between(lct, evap1 - evap_std1, evap1 + evap_std1, \
        #        alpha=0.2, edgecolor='', facecolor='orange')
        #ax2.fill_between(lct, evap2 - evap_std2, evap2 + evap_std2, \
        #        alpha=0.2, edgecolor='', facecolor='green')

        '''
        ax1.scatter(lct, soilmoist1, marker='o', c='',edgecolors='orange', label="def") # s=2.,
        ax1.scatter(lct, soilmoist2, marker='o', c='',edgecolors='green', label="imp")
        ax1.scatter(lct, soilmoist_tdr, marker='o', c='',edgecolors='blue', label="obs")

        ax2.scatter(lct, evap1, marker='o', c='',edgecolors='orange',label="def")
        ax2.scatter(lct, evap2, marker='o', c='',edgecolors='green',label="imp")
        '''
        ax1.set_xlim(-0.5,7.5)
        ax1.set_ylim(-0.03,0.01)
        ax2.set_xlim(-0.5,7.5)
        ax2.set_ylim(0.,2.)

        title = ["Year","Summer","Autumn","Winter","Spring"]
        ax1.set_title(title[ss])
        cleaner_dates = ["0","1","2","3","4","5","6","7"]
        xtickslocs = [0,1,2,3,4,5,6,7]
        ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax1.set_ylabel("$VWC - VWC_{day0}  (m^{3} m^{-3})$")
        ax1.legend()

        ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax2.set_ylabel("$ET/ET_{day0}$ (-)")
        ax2.set_xlabel("Days After Rainfall")
        ax2.legend()
        fig.savefig("EucFACE_ET-SM_after_rain_7days_%s-%s.png" % (ring, title[ss]),bbox_inches='tight')#, pad_inches=0.1)
