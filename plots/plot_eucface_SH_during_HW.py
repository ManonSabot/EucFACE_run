#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-11-13"
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

def main(fobs, fcable, case_name, ring, term):

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

    var = pd.DataFrame(cable.variables[term][:,0,0],columns=['var'])
    #var = pd.DataFrame(cable.variables['Rnet'][:,0,0]-cable.variables['Qg'][:,0,0],columns=['var'])
    var['dates'] = Time
    var = var.set_index('dates')
    var = var.resample("D").agg('mean')
    var.index = var.index - pd.datetime(2011,12,31)
    var.index = var.index.days

    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')
    Tair = Tair.resample("D").agg('max')
    Tair.index = Tair.index - pd.datetime(2011,12,31)
    Tair.index = Tair.index.days

    # exclude rainday and the after two days of rain
    day      = np.zeros((len(var)), dtype=bool)

    for i in np.arange(0,len(var)):
        if (Tair.values[i] >= 35. and Rainf.values[i] == 0.):
            day[i]   = True

    event = 0
    con_max = 0
    i = 0
    while i < len(var)-2:
        if np.all([day[i:i+3]]):
            event += 1
            i     += 3
            con   = 3
            while day[i]:
                con += 1
                i   += 1
        else:
            con = 0
            i += 1
        if con > con_max:
            con_max = con

    print(event)
    print(con_max)

    v      = np.zeros((event,con_max))
    lct    = np.zeros((event,con_max))
    v[:,:] = np.nan
    for con in np.arange(1,con_max+1):
        lct[:,con-1] = con

    i = 0
    j = 0
    while i < len(var)-2:
        if (np.all([day[i:i+3]])):
            print(i)
            print(var['var'].values[i])
            v[j,0] = var['var'].values[i]
            v[j,1] = var['var'].values[i+1]
            v[j,2] = var['var'].values[i+2]
            i = i + 3
            cont_day = 3
            while day[i]:
                v[j,cont_day] = var['var'].values[i]
                i += 1
                cont_day += 1
            j += 1
        else:
            i += 1
    print(v)
    print(lct)
    #return np.ravel(v),np.ravel(lct);
    return v,lct;

if __name__ == "__main__":

    case_name = ["ctl_met_LAI_vrt_SM_swilt-watr_31uni_HDM_or-off-litter_Hvrd",\
                 "default-met_only_or-off"]

    rings = ["amb"] #["R1","R2","R3","R4","R5","R6","amb","ele"]
    term  = "Qle" # "Qh" #"Qle"
    for ring in rings:
        fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[0], ring)
        v1,lct1 = main(fobs, fcable, case_name[0], ring, term)

        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" \
                    % (case_name[1], ring)
        v2,lct2 = main(fobs, fcable, case_name[1], ring, term)

        fig = plt.figure(figsize=[10,7])
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

        #ax1.scatter(lct1, v1, marker='o', c='',edgecolors='orange',label="def")
        #ax1.scatter(lct1, v2, marker='o', c='',edgecolors='green',label="imp")


        #lct = [0,1,2]
        #qh_mean1 = np.mean(v1, axis=0)
        #qh_mean2 = np.mean(v2, axis=0)
        #for i in np.arange(0,8):

        ax1.plot(lct1[1,:], v1[1,:], c='orange', ls= '-', label="HW1-def")
        ax1.plot(lct2[1,:], v2[1,:], c='green', ls= '-',  label="HW1-imp")
        ax1.plot(lct1[6,:], v1[6,:], c='orange', ls= '--', label="HW2-def")
        ax1.plot(lct2[6,:], v2[6,:], c='green', ls= '--',  label="HW2-imp")
        ax1.plot(lct1[0,:], v1[0,:], c='orange', ls= ':', label="HW3-def")
        ax1.plot(lct2[0,:], v2[0,:], c='green', ls= ':',  label="HW3-imp")
        '''
        ax1.plot(lct1[0,:], v1[0,:], c='orange', ls= ':', label="HW1-def")
        ax1.plot(lct2[0,:], v2[0,:], c='green', ls= ':',  label="HW1-imp")
        ax1.plot(lct1[1,:], v1[1,:], c='orange', ls= '-', label="HW2-def")
        ax1.plot(lct2[1,:], v2[1,:], c='green', ls= '-',  label="HW2-imp")
        ax1.plot(lct1[2,:], v1[2,:], c='orange', ls= '--', label="HW3-def")
        ax1.plot(lct2[2,:], v2[2,:], c='green', ls= '--',  label="HW3-imp")
        ax1.plot(lct1[3,:], v1[3,:], c='orange', ls= '-.', label="HW4-def")
        ax1.plot(lct2[3,:], v2[3,:], c='green', ls= '-.',  label="HW4-imp")
        ax1.plot(lct1[4,:], v1[4,:], c='red', ls= ':', label="HW5-def")
        ax1.plot(lct2[4,:], v2[4,:], c='blue', ls= ':',  label="HW5-imp")
        ax1.plot(lct1[5,:], v1[5,:], c='red', ls= '-', label="HW6-def")
        ax1.plot(lct2[5,:], v2[5,:], c='blue', ls= '-',  label="HW6-imp")
        ax1.plot(lct1[6,:], v1[6,:], c='red', ls= '--', label="HW7-def")
        ax1.plot(lct2[6,:], v2[6,:], c='blue', ls= '--',  label="HW7-imp")
        ax1.plot(lct1[7,:], v1[7,:], c='red', ls= '-.', label="HW8-def")
        ax1.plot(lct2[7,:], v2[7,:], c='blue', ls= '-.',  label="HW8-imp")
        '''
        ax1.set_xlim(0.5,5.5)

        ax1.set_title('')
        cleaner_dates = ["1","2","3","4","5"]
        xtickslocs = [1,2,3,4,5]
        ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        if term == "Qle":
            ax1.set_ylabel("Latent Heat $(W m^{-2})$")
            ax1.set_ylim(0,250)
        elif term == "Qh":
            ax1.set_ylabel("Sensible Heat $(W m^{-2})$")
            ax1.set_ylim(-100,100)
        ax1.set_xlabel("Days During Heatwave")
        ax1.legend()

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
        #ax1.set_xlim(-0.5,2.5)
        # ax1.set_ylim(-0.1,1.)

        fig.savefig("EucFACE_%s_during_HW_%s.png" % (term,ring),bbox_inches='tight')#, pad_inches=0.1)
        #ax1.set_ylim(-100,200)
        #fig.savefig("EucFACE_Rnet-G_during_HW_%s.png" % (ring),bbox_inches='tight')#, pad_inches=0.1)
