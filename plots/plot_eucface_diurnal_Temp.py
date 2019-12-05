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

def main(fobs, fcable):

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    var = pd.DataFrame(cable.variables['Tair'][:,0,0],columns=['Tair'])
    #var['Qle'] = cable.variables['Qg'][:,0,0] + cable.variables['Qle'][:,0,0] + cable.variables['Qh'][:,0,0]
    var['VegT'] = cable.variables['VegT'][:,0,0]
    var['SoilTemp'] = cable.variables['SoilTemp'][:,0,0,0]
    #var = pd.DataFrame(cable.variables['Rnet'][:,0,0]-cable.variables['Qg'][:,0,0],columns=['var'])
    var['dates'] = Time
    var = var.set_index('dates')
    var = var.resample("H").mean()
    print(var)
    t = lambda x : x.hour
    print(var.index.month)
    a = var.index.month == 1
    b = var.index.month == 2
    c = var.index.month == 12 #and var.index.month == 1 and var.index.month == 2

    var = var[var.index.month == 1]
    var = var.groupby(var.index.hour).mean()
    print(var)
    plt.plot(var['Tair'],c="orange",label="Tair")
    plt.plot(var['VegT'],c="green",label="TVeg")
    plt.plot(var['SoilTemp'],c="blue",label="SoilTemp")
    #plt.ylim((-200,500))
    plt.legend()
    plt.show()
    '''
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    Rainf.index = Rainf.index.days

    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')
    Tair = Tair.resample("D").agg('max')
    Tair.index = Tair.index - pd.datetime(2011,12,31)
    Tair.index = Tair.index.days
    '''
    #return v,lct;

if __name__ == "__main__":

    case_name = ["TumbaFluxnet_out.nc"]
                #["ctl_met_LAI_vrt_SM_swilt-watr_31uni_HDM_or-off-litter_Hvrd",\
                #"default-met_only_or-off"]

    fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    #fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/ctl_met_LAI_vrt_SM_swilt-watr_31uni_HDM_or-off-litter_Hvrd/EucFACE_amb_out.nc"
    #fcable = "/srv/ccrc/data25/z5218916/cable/EucFACE/test_PLUMBER/outputs/Mark_latest_gw-on_spinup-off_Hvrd_litter_met/TumbaFluxnet_out.nc"
    fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_SM_swilt-watr_31uni_Hvrd/EucFACE_amb_out.nc"

    main(fobs, fcable)


    '''
        fig = plt.figure(figsize=[10,7])
        fig.subplots_adjust(hspace=0.1)
        fig.subplots_adjust(wspace=0.05)
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = "Helvetica"
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['font.size'] = 16
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16

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

        #ax1.scatter(lct1, v1, marker='o', c='',edgecolors='orange',label="CTL")
        #ax1.scatter(lct1, v2, marker='o', c='',edgecolors='green',label="NEW")


        #lct = [0,1,2]
        #qh_mean1 = np.mean(v1, axis=0)
        #qh_mean2 = np.mean(v2, axis=0)
        #for i in np.arange(0,8):

        ax1.plot(lct1[1,:], v1[1,:], c='orange', ls= '-', label="HW1-CTL")
        ax1.plot(lct2[1,:], v2[1,:], c='green', ls= '-',  label="HW1-NEW")
        ax1.plot(lct1[6,:], v1[6,:], c='orange', ls= '--', label="HW2-CTL")
        ax1.plot(lct2[6,:], v2[6,:], c='green', ls= '--',  label="HW2-NEW")
        ax1.plot(lct1[7,:], v1[7,:], c='orange', ls= ':', label="HW3-CTL")
        ax1.plot(lct2[7,:], v2[7,:], c='green', ls= ':',  label="HW3-NEW")

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



        fig.savefig("EucFACE_%s_during_HW_%s.png" % (term,ring),bbox_inches='tight')#, pad_inches=0.1)
        #ax1.set_ylim(-100,200)
        #fig.savefig("EucFACE_Rnet-G_during_HW_%s.png" % (ring),bbox_inches='tight')#, pad_inches=0.1)
    '''
