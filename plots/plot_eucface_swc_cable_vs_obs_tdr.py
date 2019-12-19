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
import glob
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

#def main(fobs_Esoil, fobs, fcable, case_name, hk, b, ring, layer):
def main(fobs_Esoil, fobs_Trans, fobs, fcable, case_name, ring, layer):

    subs_Esoil = read_obs_esoil(fobs_Esoil, ring)
    subs_Trans = read_obs_trans(fobs_Trans, ring)
    subset     = read_obs_swc(fobs, ring)

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

    swilt = np.zeros(len(Rainf))
    sfc = np.zeros(len(Rainf))
    ssat = np.zeros(len(Rainf))

    if layer == "6":
        swilt[:] = ( cable.variables['swilt'][0]*0.022 + cable.variables['swilt'][1]*0.058 \
                   + cable.variables['swilt'][2]*0.154 + cable.variables['swilt'][3]*(0.5-0.022-0.058-0.154) )/0.5
        sfc[:] = ( cable.variables['sfc'][0]*0.022   + cable.variables['sfc'][1]*0.058 \
                   + cable.variables['sfc'][2]*0.154 + cable.variables['sfc'][3]*(0.5-0.022-0.058-0.154) )/0.5
        ssat[:] = ( cable.variables['ssat'][0]*0.022 + cable.variables['ssat'][1]*0.058 \
                   + cable.variables['ssat'][2]*0.154+ cable.variables['ssat'][3]*(0.5-0.022-0.058-0.154) )/0.5
    elif layer == "13":
        swilt[:] =(cable.variables['swilt'][0]*0.02 + cable.variables['swilt'][1]*0.05 \
                 + cable.variables['swilt'][2]*0.06 + cable.variables['swilt'][3]*0.13 \
                 + cable.variables['swilt'][4]*(0.5-0.02-0.05-0.06-0.13) )/0.5
        sfc[:] =(cable.variables['sfc'][0]*0.02 + cable.variables['sfc'][1]*0.05 \
               + cable.variables['sfc'][2]*0.06 + cable.variables['sfc'][3]*0.13 \
               + cable.variables['sfc'][4]*(0.5-0.02-0.05-0.06-0.13) )/0.5
        ssat[:] =(cable.variables['ssat'][0]*0.02 + cable.variables['ssat'][1]*0.05 \
                + cable.variables['ssat'][2]*0.06 + cable.variables['ssat'][3]*0.13 \
                + cable.variables['ssat'][4]*(0.5-0.02-0.05-0.06-0.13) )/0.5
    elif layer == "31uni":
        swilt[:] =(cable.variables['swilt'][0]*0.15 + cable.variables['swilt'][1]*0.15 \
                  + cable.variables['swilt'][2]*0.15 + cable.variables['swilt'][3]*0.05 )/0.5
        sfc[:] =(cable.variables['sfc'][0]*0.15 + cable.variables['sfc'][1]*0.15 \
                + cable.variables['sfc'][2]*0.15 + cable.variables['sfc'][3]*0.05 )/0.5
        ssat[:] =(cable.variables['ssat'][0]*0.15 + cable.variables['ssat'][1]*0.15 \
                 + cable.variables['ssat'][2]*0.15 + cable.variables['ssat'][3]*0.05 )/0.5
    elif layer == "31exp":
        swilt[:] = ( cable.variables['swilt'][0]*0.020440 + cable.variables['swilt'][1]*0.001759 \
                    + cable.variables['swilt'][2]*0.003957 + cable.variables['swilt'][3]*0.007035 \
                    + cable.variables['swilt'][4]*0.010993 + cable.variables['swilt'][5]*0.015829 \
                    + cable.variables['swilt'][6]*0.021546 + cable.variables['swilt'][7]*0.028141 \
                    + cable.variables['swilt'][8]*0.035616 + cable.variables['swilt'][9]*0.043971 \
                    + cable.variables['swilt'][10]*0.053205+ cable.variables['swilt'][11]*0.063318 \
                    + cable.variables['swilt'][12]*0.074311+ cable.variables['swilt'][13]*0.086183 \
                    + cable.variables['swilt'][14]*(0.5-0.466304))/0.5
        sfc[:] =   ( cable.variables['sfc'][0]*0.020440  + cable.variables['sfc'][1]*0.001759 \
                    + cable.variables['sfc'][2]*0.003957 + cable.variables['sfc'][3]*0.007035 \
                    + cable.variables['sfc'][4]*0.010993 + cable.variables['sfc'][5]*0.015829 \
                    + cable.variables['sfc'][6]*0.021546 + cable.variables['sfc'][7]*0.028141 \
                    + cable.variables['sfc'][8]*0.035616 + cable.variables['sfc'][9]*0.043971 \
                    + cable.variables['sfc'][10]*0.053205+ cable.variables['sfc'][11]*0.063318 \
                    + cable.variables['sfc'][12]*0.074311+ cable.variables['sfc'][13]*0.086183 \
                    + cable.variables['sfc'][14]*(0.5-0.466304))/0.5
        ssat[:] =  ( cable.variables['ssat'][0]*0.020440  + cable.variables['ssat'][1]*0.001759 \
                    + cable.variables['ssat'][2]*0.003957 + cable.variables['ssat'][3]*0.007035 \
                    + cable.variables['ssat'][4]*0.010993 + cable.variables['ssat'][5]*0.015829 \
                    + cable.variables['ssat'][6]*0.021546 + cable.variables['ssat'][7]*0.028141 \
                    + cable.variables['ssat'][8]*0.035616 + cable.variables['ssat'][9]*0.043971 \
                    + cable.variables['ssat'][10]*0.053205+ cable.variables['ssat'][11]*0.063318 \
                    + cable.variables['ssat'][12]*0.074311+ cable.variables['ssat'][13]*0.086183 \
                    + cable.variables['ssat'][14]*(0.5-0.466304))/0.5
    elif layer == "31para":
        swilt[:] =( cable.variables['swilt'][0]*0.020440 \
                  + cable.variables['swilt'][1]*0.001759 \
                  + cable.variables['swilt'][2]*0.003957 \
                  + cable.variables['swilt'][3]*0.007035 \
                  + cable.variables['swilt'][4]*0.010993 \
                  + cable.variables['swilt'][5]*0.015829 \
                  + cable.variables['swilt'][6]*(0.5-0.420714))/0.5
        sfc[:] =( cable.variables['sfc'][0]*0.020440 \
                 + cable.variables['sfc'][1]*0.001759 \
                 + cable.variables['sfc'][2]*0.003957 \
                 + cable.variables['sfc'][3]*0.007035 \
                 + cable.variables['sfc'][4]*0.010993 \
                 + cable.variables['sfc'][5]*0.015829 \
                 + cable.variables['sfc'][6]*(0.5-0.420714))/0.5
        ssat[:] =( cable.variables['ssat'][0]*0.020440 \
                 + cable.variables['ssat'][1]*0.001759 \
                 + cable.variables['ssat'][2]*0.003957 \
                 + cable.variables['ssat'][3]*0.007035 \
                 + cable.variables['ssat'][4]*0.010993 \
                 + cable.variables['ssat'][5]*0.015829 \
                 + cable.variables['ssat'][6]*(0.5-0.420714))/0.5

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

    ax1 = fig.add_subplot(311)
    #ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(312)
    #ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(313)
    #ax6 = fig.add_subplot(426)
    #ax7 = fig.add_subplot(427)
    #ax8 = fig.add_subplot(428)

    width = 1.
    x = Rainf.index
    #np.arange(len(Rainf.index))
    #print(x)
    #print(Rainf.values)

    ax1.plot(subset.index, subset.values,   c="green", lw=1.0, ls="-", label="tdr")
    ax1.plot(x, SoilMoist.values,c="orange", lw=1.0, ls="-", label="swc")

    tmp1 = SoilMoist['SoilMoist'].loc[SoilMoist.index.isin(subset.index)]
    tmp2 = subset.loc[subset.index.isin(SoilMoist.index)]
    mask = np.isnan(tmp2)
    print(mask)
    tmp1 = tmp1[mask == False]
    tmp2 = tmp2[mask == False]
    #print(np.isnan(tmp1).values.any())
    #print(np.isnan(tmp2).values.any())
    cor_tdr = stats.pearsonr(tmp1,tmp2)
    mse_tdr = mean_squared_error(tmp2, tmp1)
    ax1.set_title("r = % 5.3f , MSE = % 5.3f" %(cor_tdr[0], np.sqrt(mse_tdr)))
    print("-----------------------------------------------")
    print(mse_tdr)
    ax1.plot(x, swilt,           c="black", lw=1.0, ls="-", label="swilt")
    ax1.plot(x, sfc,             c="black", lw=1.0, ls="-.", label="sfc")
    ax1.plot(x, ssat,            c="black", lw=1.0, ls=":", label="ssat")
    ax3.plot(x, Fwsoil.values,   c="forestgreen", lw=1.0, ls="-", label="Fwsoil")
    ax5.plot(x, TVeg['TVeg'].rolling(window=5).mean(),     c="green", lw=1.0, ls="-", label="Trans") #.rolling(window=7).mean()
    ax5.plot(x, ESoil['ESoil'].rolling(window=5).mean(),    c="orange", lw=1.0, ls="-", label="ESoil") #.rolling(window=7).mean()
    ax5.scatter(subs_Trans.index, subs_Trans['volRing'], marker='o', c='',edgecolors='blue', s = 2., label="Trans Obs") # subs['EfloorPred']
    ax5.scatter(subs_Esoil.index, subs_Esoil['wuTP'], marker='o', c='',edgecolors='red', s = 2., label="ESoil Obs") # subs['EfloorPred']

    #ax7.plot(x, Tair['Tair'].rolling(window=7).mean(),     c="red",    lw=1.0, ls="-", label="Tair")
    #ax7.plot(x, VegT['VegT'].rolling(window=7).mean(),     c="orange", lw=1.0, ls="-", label="VegT")

    #rects2 = ax2.bar(x, Rainf['Rainf'], width, color='royalblue', label='Obs')
    #ax4.plot(x, Qair['Qair'].rolling(window=7).mean(),     c="royalblue", lw=1.0, ls="-", label="Qair")
    #ax6.plot(x, Wind['Wind'].rolling(window=7).mean(),     c="darkgoldenrod", lw=1.0, ls="-", label="Wind")
    #ax8.plot(x, Rnet['Rnet'].rolling(window=7).mean(),     c="crimson", lw=1.0, ls="-", label="Rnet")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    #plt.setp(ax1.get_xticklabels(), visible=False)
    #ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    #ax1.set_ylabel("CABLE Runoff (mm)")
    #ax1.axis('tight')
    #ax1.set_xlim(0,2374)

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.set_ylabel("VWC (m3/m3)")
    ax1.axis('tight')
    ax1.set_ylim(0,0.5)
    ax1.set_xlim(367,2739)
    ax1.legend()

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax3.set_ylabel("Fwsoil (-)")
    ax3.axis('tight')
    ax3.set_ylim(0.,1.2)
    ax3.set_xlim(367,2739)

    #plt.setp(ax5.get_xticklabels(), visible=False)
    ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax5.set_ylabel("ET ($mm d^{-1}$)")
    ax5.axis('tight')
    ax5.set_ylim(0.,4.5)
    ax5.set_xlim(367,2739)
    ax5.legend()
    '''
    ax7.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax7.set_ylabel("Tair, VegT (°C)")
    ax7.axis('tight')
    ax7.set_ylim(0.,30.)
    ax7.set_xlim(367,2739)
    ax7.legend()

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Rain (mm/day)")
    ax2.axis('tight')
    ax2.set_ylim(0.,150.)
    ax2.set_xlim(367,2739)

    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel("Qair (kg/kg)")
    ax4.axis('tight')
    ax4.set_ylim(0.,0.0002)
    ax4.set_xlim(367,2739)

    plt.setp(ax6.get_xticklabels(), visible=False)
    ax6.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax6.yaxis.tick_right()
    ax6.yaxis.set_label_position("right")
    ax6.set_ylabel("Wind (m/s)")
    ax6.axis('tight')
    ax6.set_ylim(0.,3.0)
    ax6.set_xlim(367,2739)

    ax8.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax8.yaxis.tick_right()
    ax8.yaxis.set_label_position("right")
    ax8.set_ylabel("Rnet (W/m²)")
    ax8.axis('tight')
    ax8.set_ylim(-20.,160.)
    ax8.set_xlim(367,2739)
    '''
    #fig.savefig("EucFACE_tdr_%s_hk=%s_b=%s_%s.png" % (case_name, hk, b, ring), bbox_inches='tight', pad_inches=0.1)
    fig.savefig("EucFACE_tdr_%s_%s.png" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)
    #fig.savefig("EucFACE_tdr_%s_%s.png" % (case_name, ring), bbox_inches='tight', pad_inches=0.1)


def read_obs_esoil(fobs_Esoil, ring):

   est_esoil = pd.read_csv(fobs_Esoil, usecols = ['Ring','Date','wuTP','EfloorPred'])
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
   subs['EfloorPred'] = subs['EfloorPred'].clip(lower=0.)
   subs['EfloorPred'] = subs['EfloorPred'].replace(0., float('nan'))

   return subs

def read_obs_trans(fobs_Trans, ring):
   est_trans = pd.read_csv(fobs_Trans, usecols = ['Ring','Date','volRing'])
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
   return subs

def read_obs_swc(fobs, ring):
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
    subset = subset.xs('swc.tdr', axis=1, drop_level=True)

    return subset


if __name__ == "__main__":

    layer =  "31uni"
    '''
    if layer == "6":
        cases = [ "met_only_fw-fixed_6_gw-off","met_only_6_gw-off"]
        """
        cases = [ "met_only_6_","met_LAI_6_","met_LAI_vrt_6_","met_LAI_vrt_swilt-watr-ssat_6_",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_6_","met_LAI_vrt_swilt-watr-ssat_SM_6_litter",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_6_Or","met_LAI_vrt_swilt-watr-ssat_SM_6_fw-Haverd",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_6_fw-hie-exp","met_LAI_vrt_swilt-watr-ssat_SM_6_fw-hie-watpot"]
        """
    elif layer == "31para":
        cases = ["ctl_met_LAI_vrt_SM_swilt-watr_31para"]
    elif layer == "31exp":
        cases = ["ctl_met_LAI_vrt_SM_swilt-watr_31exp"]
    elif layer == "31uni":
        cases = ["met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter","met_LAI_vrt_swilt-watr-ssat_SM_31uni_Or",\
                 "met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-hie-exp",\
                 "met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-Haverd","met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-hie-watpot"]
    '''
    cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs",\
                      "met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx01-hydsx*_fw-hie-exp"))
    rings = ["amb"]#["R1","R2","R3","R4","R5","R6","amb","ele"]

    for case_name in cases:
        for ring in rings:
            fobs_Esoil = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_underET.csv"
            fobs_Trans = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_sapflow.csv"
            fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
            #fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            #fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            #fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            fcable ="%s/EucFACE_%s_out.nc" % (case_name, ring)
            main(fobs_Esoil, fobs_Trans, fobs, fcable, case_name, ring, layer)
