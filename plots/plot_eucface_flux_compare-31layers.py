#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-8-27"
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

def main(fcable, case_name):

    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])
    SoilMoist['SWC_50cm'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,1,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,2,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,3,0,0]*0.05 )
    SoilMoist['SWC_200cm'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,1,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,2,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,3,0,0]*0.05 \
                             + cable.variables['SoilMoist'][:,4,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,5,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,6,0,0]*0.05 \
                             + cable.variables['SoilMoist'][:,7,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,8,0,0]*0.05 \
                             + cable.variables['SoilMoist'][:,9,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,10,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,11,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,12,0,0]*0.15 \
                             + cable.variables['SoilMoist'][:,13,0,0]*0.05 )

    SoilMoist['SWC'] = cable.variables['SoilMoist'][:,0,0,0]

    for i in np.arange(1,31):
        #print(i)
        SoilMoist['SWC'] = SoilMoist['SWC'] + cable.variables['SoilMoist'][:,i,0,0]
    SoilMoist['SWC'] = SoilMoist['SWC']*0.15

    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.sort_values(by=['dates'])

    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    Rainf.index = Rainf.index.days

    TVeg = pd.DataFrame(cable.variables['TVeg'][:,0,0],columns=['TVeg'])
    TVeg = TVeg*1800.
    TVeg['dates'] = Time
    TVeg = TVeg.set_index('dates')
    TVeg = TVeg.resample("D").agg('sum')
    TVeg.index = TVeg.index - pd.datetime(2011,12,31)
    TVeg.index = TVeg.index.days

    ESoil = pd.DataFrame(cable.variables['ESoil'][:,0,0],columns=['ESoil'])
    ESoil = ESoil*1800.
    ESoil['dates'] = Time
    ESoil = ESoil.set_index('dates')
    ESoil = ESoil.resample("D").agg('sum')
    ESoil.index = ESoil.index - pd.datetime(2011,12,31)
    ESoil.index = ESoil.index.days

    ECanop = pd.DataFrame(cable.variables['ECanop'][:,0,0],columns=['ECanop'])
    ECanop = ECanop*1800.
    ECanop['dates'] = Time
    ECanop = ECanop.set_index('dates')
    ECanop = ECanop.resample("D").agg('sum')
    ECanop.index = ECanop.index - pd.datetime(2011,12,31)
    ECanop.index = ECanop.index.days

    Qrecharge = pd.DataFrame(cable.variables['Qrecharge'][:,0,0],columns=['Qrecharge'])
    Qrecharge = Qrecharge*1800.
    Qrecharge['dates'] = Time
    Qrecharge = Qrecharge.set_index('dates')
    Qrecharge = Qrecharge.resample("D").agg('sum')
    Qrecharge.index = Qrecharge.index - pd.datetime(2011,12,31)
    Qrecharge.index = Qrecharge.index.days

    Qrf = pd.DataFrame(cable.variables['Qs'][:,0,0],columns=['Qrf'])
    Qrf['Qrf'] = (cable.variables['Qs'][:,0,0]+ cable.variables['Qsb'][:,0,0])*1800.
    #print(Qrf)
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

    Rnet = pd.DataFrame(cable.variables['Rnet'][:,0,0],columns=['Rnet'])
    Rnet['dates'] = Time
    Rnet = Rnet.set_index('dates')
    Rnet = Rnet.resample("D").agg('mean')
    Rnet.index = Rnet.index - pd.datetime(2011,12,31)
    Rnet.index = Rnet.index.days

    Fwsoil = pd.DataFrame(cable.variables['Fwsoil'][:,0,0],columns=['Fwsoil'])
    Fwsoil['dates'] = Time
    Fwsoil = Fwsoil.set_index('dates')
    Fwsoil = Fwsoil.resample("D").agg('mean')
    Fwsoil.index = Fwsoil.index - pd.datetime(2011,12,31)
    Fwsoil.index = Fwsoil.index.days

    var = np.zeros(15)
    # drought period 2017-11-1 - 2018-9-1
    var[0] = np.mean(SoilMoist['SWC_50cm'][1766:2070])*1000.
    var[1] = np.mean(SoilMoist['SWC_200cm'][1766:2070])*1000.
    var[2] = np.mean(SoilMoist['SWC'][1766:2070])*1000.
    var[3] = np.mean(Rainf['Rainf'][1766:2070])
    var[4] = np.mean(TVeg['TVeg'][1766:2070])
    var[5] = np.mean(ESoil['ESoil'][1766:2070])
    var[6] = np.mean(ECanop['ECanop'][1766:2070])
    var[7] = np.mean(Qrecharge['Qrecharge'][1766:2070])
    var[8] = np.mean(Qrf['Qrf'][1766:2070])
    var[9] = np.mean(Tair['Tair'][1766:2070])
    var[10] = np.mean(VegT['VegT'][1766:2070])
    var[11] = np.mean(Qair['Qair'][1766:2070])
    var[12] = np.mean(Wind['Wind'][1766:2070])
    var[13] = np.mean(Rnet['Rnet'][1766:2070])
    var[14] = np.mean(Fwsoil['Fwsoil'][1766:2070])

    #var.to_csv("EucFACE_amb_31-layer_.csv" %(layers, case_name))
    print(case_name)
    print(var[:])
    #print("%s %10.5f" %( case_name, var[:]))
    #print("%s %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f" %( case_name, var[:]))

if __name__ == "__main__":

    case = ["31-layer_exp","31-layer_para","bch0.5","bch1.5","ctl","froot_parabola",\
            "froot_triangle","froot_triangle_inverse","hyds0.1","hyds10","leafsize_eucalyptus",\
            "sfc0.5","sfc1.5","ssat=0.35_sfc=0.3_swilt=0.03_top1-layer","ssat=0.35_sfc=0.3_swilt=0.03_top3-layer",\
            "ssat0.75","ssat1.5","sucs0.5","sucs1.5","swilt0.5","swilt1.5","Cosby_multivariate",\
            "Cosby_univariate","HC_SWC"]


    for case_name in case:
        fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/31-layer/%s/EucFACE_amb_out.nc" % (case_name)
        main(fcable, case_name)
