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

def main(fobs, fcable, case_name, ring):

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

    Qh = pd.DataFrame(cable.variables['Qh'][:,0,0],columns=['Qh'])
    Qh['dates'] = Time
    Qh = Qh.set_index('dates')
    Qh = Qh.resample("D").agg('mean')
    Qh.index = Qh.index - pd.datetime(2011,12,31)
    Qh.index = Qh.index.days

    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')
    Tair = Tair.resample("D").agg('max')
    Tair.index = Tair.index - pd.datetime(2011,12,31)
    Tair.index = Tair.index.days

    # exclude rainday and the after two days of rain
    day0      = np.zeros((len(Qh)), dtype=bool)
    day1      = np.zeros((len(Qh)), dtype=bool)
    day2      = np.zeros((len(Qh)), dtype=bool)
    #day3      = np.zeros((len(Qh)), dtype=bool)
    #day4      = np.zeros((len(Qh)), dtype=bool)

    for i in np.arange(0,len(Qh)-2):
        if Tair.values[i] >= 35.:
            print(i)
            print(Tair.values[i])

        a = np.all([Tair.values[i] >= 35., Tair.values[i+1] >= 35., Tair.values[i+2] >= 35.])#, \
                   #Tair.values[i+3] >= 35., Tair.values[i+4] >= 35.])
        b = np.all([Rainf.values[i] == 0., Rainf.values[i+1] == 0., Rainf.values[i+2] == 0.])#, \
                  # Rainf.values[i+3] == 0., Rainf.values[i+4] == 0.])
        if (a and b):
            day0[i]   = True
            day1[i+1] = True
            day2[i+2] = True
            #day3[i+3] = True
            #day4[i+4] = True
    print(np.any(day0))
    qh      = np.zeros((len(Qh[day0 == True]),3))
    lct     = np.zeros((len(Qh[day0 == True]),3))
    qh[:,0] = Qh['Qh'].values[day0 == True]
    qh[:,1] = Qh['Qh'].values[day1 == True]
    qh[:,2] = Qh['Qh'].values[day2 == True]
    #qh[:,3] = Qh['Qh'].values[day3 == True]
    #qh[:,4] = Qh['Qh'].values[day4 == True]

    lct[:,0] = 0
    lct[:,1] = 1
    lct[:,2] = 2
    #lct[:,3] = 3
    #lct[:,4] = 4

    return qh,lct;

if __name__ == "__main__":

    case_name = ["ctl_met_LAI_vrt_SM_swilt-watr_31uni_HDM_or-off-litter_Hvrd",\
                 "default-met_only_or-off"]

    rings = ["R1","R2","R3","R4","R5","R6","amb","ele"]

    for ring in rings:
        fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[0], ring)
        qh1,lct1 = main(fobs, fcable, case_name[0], ring)

        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[1], ring)
        qh2,lct2 = main(fobs, fcable, case_name[1], ring)

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

        #ax1.scatter(lct1, qh1, marker='o', c='',edgecolors='orange',label="def")
        #ax1.scatter(lct1, qh2, marker='o', c='',edgecolors='green',label="imp")


        lct = [0,1,2]
        qh_mean1 = np.mean(qh1, axis=0)
        qh_mean2 = np.mean(qh2, axis=0)
        ax1.plot(lct, qh_mean1, c='orange', label="def")
        ax1.plot(lct, qh_mean2, c='green', label="imp")
        '''
        soilmoist_tdr = np.mean(soilmoist_tdr_rn1, axis=0)
        soilmoist_max1 = np.max(soilmoist_rn1, axis=0)
        soilmoist_max2 = np.max(soilmoist_rn2, axis=0)
        soilmoist_max_tdr = np.max(soilmoist_tdr_rn1, axis=0)
        soilmoist_min1 = np.min(soilmoist_rn1, axis=0)
        soilmoist_min2 = np.min(soilmoist_rn2, axis=0)
        soilmoist_min_tdr = np.min(soilmoist_tdr_rn1, axis=0)
        evap1 = np.mean(evap_rn1, axis=0)
        evap2 = np.mean(evap_rn2, axis=0)
        evap_max1 = np.max(evap_rn1, axis=0)
        evap_max2 = np.max(evap_rn2, axis=0)
        evap_min1 = np.max(evap_rn1, axis=0)
        evap_min2 = np.max(evap_rn2, axis=0)
        ax1.plot(lct, soilmoist1, c='orange', label="def")
        ax1.plot(lct, soilmoist2, c='green', label="imp")
        ax1.plot(lct, soilmoist_tdr, c='blue', label="obs")
        ax1.fill_between(lct, soilmoist_min1, soilmoist_max1, alpha=0.5, edgecolor='', facecolor='orange')
        ax1.fill_between(lct, soilmoist_min2, soilmoist_max2, alpha=0.5, edgecolor='', facecolor='green')
        ax1.fill_between(lct, soilmoist_min_tdr, soilmoist_max_tdr, alpha=0.5, edgecolor='', facecolor='blue')

        ax2.plot(lct, evap1, c='orange', label="def")
        ax2.plot(lct, evap2, c='green', label="imp")
        ax2.fill_between(lct, evap_min1, evap_max1, alpha=0.5, edgecolor='', facecolor='orange')
        ax2.fill_between(lct, evap_min2, evap_max2, alpha=0.5, edgecolor='', facecolor='green')
        '''
        '''
        ax1.scatter(lct, soilmoist1, marker='o', c='',edgecolors='orange', label="def") # s=2.,
        ax1.scatter(lct, soilmoist2, marker='o', c='',edgecolors='green', label="imp")
        ax1.scatter(lct, soilmoist_tdr, marker='o', c='',edgecolors='blue', label="obs")

        ax2.scatter(lct, evap1, marker='o', c='',edgecolors='orange',label="def")
        ax2.scatter(lct, evap2, marker='o', c='',edgecolors='green',label="imp")
        '''
        ax1.set_xlim(-0.5,2.5)
        # ax1.set_ylim(-0.1,1.)
        ax1.legend()
        fig.savefig("EucFACE_SH_during_HW_%s.png" % (ring),bbox_inches='tight')#, pad_inches=0.1)
