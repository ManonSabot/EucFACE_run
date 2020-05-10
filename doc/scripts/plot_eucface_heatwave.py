#!/usr/bin/env python

"""
Pick out heatwaves and get heatwave plots

Include functions :
    find_Heatwave
    find_Heatwave_hourly
    plot_single_HW_event
    plot_EF_SM_HW
    get_day_time_Qle_Qh_EF
    find_all_Heatwave_days
    boxplot_Qle_Qh_EF_HW
    group_boxplot_Qle_Qh_EF_HW

"""
__author__ = "MU Mengyuan"
__version__ = "2020-03-19"

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import datetime as dt
import netCDF4 as nc
import seaborn as sns
import scipy.stats as stats
from matplotlib import cm
from matplotlib import ticker
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
from plot_eucface_get_var import *

def find_Heatwave(fcable, ring, layer):

    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    # Air temperature
    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')
    Tair = Tair.resample("D").agg('max')
    #Tair.index = Tair.index - pd.datetime(2011,12,31)
    #Tair.index = Tair.index.days

    # Precipitation
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0],columns=['Rainf'])
    Rainf = Rainf*1800.
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')
    Rainf = Rainf.resample("D").agg('sum')
    #Rainf.index = Rainf.index - pd.datetime(2011,12,31)
    #Rainf.index = Rainf.index.days

    Qle = read_cable_var(fcable, "Qle")
    Qh  = read_cable_var(fcable, "Qh")
    Rnet= read_cable_var(fcable, "Qle") + read_cable_var(fcable, "Qh")

    #Rnet["cable"] = np.where(Rnet["cable"].values < 1., Qle['cable'].values, Rnet["cable"].values)
    EF = pd.DataFrame(Qle['cable'].values/Rnet['cable'].values, columns=['EF'])
    #EF['EF'] = np.where(EF["EF"].values >10.0, 10., EF["EF"].values)
    SM = read_SM_top_mid_bot(fcable, ring, layer)

    # exclude rainday and the after two days of rain
    day = np.zeros((len(Tair)), dtype=bool)

    for i in np.arange(0,len(Tair)):
        if (Tair.values[i] >= 35.): # and Rainf.values[i] == 0.):
            day[i]   = True

    # calculate heatwave event
    HW = [] # create empty list

    i = 0
    while i < len(Tair)-2:
        HW_event = []
        if (np.all([day[i:i+3]])):
            # consistent 3 days > 35 degree
            for j in np.arange(i-2,i+3):

                event = ( Tair.index[j], Tair['Tair'].values[j], Rainf['Rainf'].values[j],
                          Qle['cable'].values[j], Qh['cable'].values[j],
                          EF['EF'].values[j], SM['SM_top'].values[j], SM['SM_mid'].values[j],
                          SM['SM_bot'].values[j], SM['SM_all'].values[j], SM['SM_15m'].values[j])
                HW_event.append(event)
            i = i + 3

            while day[i]:
                # consistent more days > 35 degree
                event = ( Tair.index[i], Tair['Tair'].values[i], Rainf['Rainf'].values[i],
                          Qle['cable'].values[i], Qh['cable'].values[i],
                          EF['EF'].values[i], SM['SM_top'].values[i], SM['SM_mid'].values[i],
                          SM['SM_bot'].values[i], SM['SM_all'].values[i], SM['SM_15m'].values[j] )
                HW_event.append(event)
                i += 1

            # post 2 days
            event = ( Tair.index[i], Tair['Tair'].values[i], Rainf['Rainf'].values[i],
                      Qle['cable'].values[i], Qh['cable'].values[i],
                      EF['EF'].values[i], SM['SM_top'].values[i], SM['SM_mid'].values[i],
                      SM['SM_bot'].values[i], SM['SM_all'].values[i], SM['SM_15m'].values[j] )
            HW_event.append(event)

            event = ( Tair.index[i+1], Tair['Tair'].values[i+1], Rainf['Rainf'].values[i+1],
                      Qle['cable'].values[i+1], Qh['cable'].values[i+1],
                      EF['EF'].values[i+1], SM['SM_top'].values[i+1], SM['SM_mid'].values[i+1],
                      SM['SM_bot'].values[i+1], SM['SM_all'].values[i+1], SM['SM_15m'].values[j] )
            HW_event.append(event)

            HW.append(HW_event)
        else:
            i += 1

    # The variable HW is a nested list, in Python accessing a nested list cannot\
    # be done by multi-dimensional slicing, i.e.: HW[1,2], instead one  would   \
    # write HW[1][2].
    # HW[:][0] does not work because HW[:] returns HW.

    return HW

def find_Heatwave_hourly(fcable, ring, layer):

    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    # Air temperature
    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')

    Tair_daily = Tair.resample("D").agg('max')

    # Precipitation
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0]*1800.,columns=['Rainf'])
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')

    Qle          = pd.DataFrame(cable.variables['Qle'][:,0,0],columns=['cable'])
    Qle['dates'] = Time
    Qle          = Qle.set_index('dates')

    Qh           = pd.DataFrame(cable.variables['Qh'][:,0,0],columns=['cable'])
    Qh['dates']  = Time
    Qh           = Qh.set_index('dates')

    Rnet         = Qle + Qh

    #print(Rnet)

    EF          = pd.DataFrame(Qle['cable'].values/Rnet['cable'].values, columns=['EF'])
    EF['dates'] = Time
    EF          = EF.set_index('dates')
    #EF['EF']    = np.where(np.all([Qle["cable"].values > 1., Qh["cable"].values > 1.],axis=0 ), EF['EF'].values, float('nan'))
    EF['EF']    = np.where(np.all([EF.index.hour >= 9., EF.index.hour <= 16., EF['EF'].values <= 5. ],axis=0 ), EF['EF'].values, float('nan'))
    #EF['EF'] = np.where(EF["EF"].values >10.0, 10., EF["EF"].values)
    SM = read_SM_top_mid_bot_hourly(fcable, ring, layer)

    #print(SM)

    # exclude rainday and the after two days of rain
    day = np.zeros((len(Tair_daily)), dtype=bool)

    for i in np.arange(0,len(Tair_daily)):
        if (Tair_daily.values[i] >= 35.): # and Rainf.values[i] == 0.):
            day[i]   = True

    # calculate heatwave event
    HW = [] # create empty list

    i = 0

    #while i < len(Tair_daily)-2:
    while i < len(Tair_daily)-1:
        HW_event = []

        if (np.all([day[i:i+3]])):

            #day_start = Tair_daily.index[i-2]
            day_start = Tair_daily.index[i-1]
            i = i + 3

            while day[i]:

                i += 1

            else:
                #if i+2 < len(Tair_daily.index):
                #    day_end = Tair_daily.index[i+2] # the third day after heatwave
                #elif i+1 < len(Tair_daily.index):
                if i+1 < len(Tair_daily.index):
                    day_end = Tair_daily.index[i+1]
                else:
                    day_end = Tair_daily.index[i]

                Tair_event  = Tair[np.all([Tair.index >= day_start,  Tair.index < day_end],axis=0)]
                Rainf_event = Rainf[np.all([Tair.index >= day_start, Tair.index < day_end],axis=0)]
                Qle_event   = Qle[np.all([Tair.index >= day_start,   Tair.index < day_end],axis=0)]
                Qh_event    = Qh[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]
                EF_event    = EF[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]
                SM_event    = SM[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]

                for hour_num in np.arange(len(Tair_event)):
                    hour_in_event = ( Tair_event.index[hour_num],
                                      Tair_event['Tair'].values[hour_num],
                                      Rainf_event['Rainf'].values[hour_num],
                                      Qle_event['cable'].values[hour_num],
                                      Qh_event['cable'].values[hour_num],
                                      EF_event['EF'].values[hour_num],
                                      SM_event['SM_top'].values[hour_num],
                                      SM_event['SM_mid'].values[hour_num],
                                      SM_event['SM_bot'].values[hour_num],
                                      SM_event['SM_all'].values[hour_num],
                                      SM_event['SM_15m'].values[hour_num] )
                    HW_event.append(hour_in_event)

            HW.append(HW_event)
        else:
            i += 1
    #print(HW[0])

    return HW

def find_Heatwave_hourly_beta(fcable, ring, layer):

    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    # Air temperature
    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')

    Tair_daily = Tair.resample("D").agg('max')

    # Precipitation
    Rainf = pd.DataFrame(cable.variables['Rainf'][:,0,0]*1800.,columns=['Rainf'])
    Rainf['dates'] = Time
    Rainf = Rainf.set_index('dates')

    Qle          = pd.DataFrame(cable.variables['Qle'][:,0,0],columns=['cable'])
    Qle['dates'] = Time
    Qle          = Qle.set_index('dates')

    Qh           = pd.DataFrame(cable.variables['Qh'][:,0,0],columns=['cable'])
    Qh['dates']  = Time
    Qh           = Qh.set_index('dates')

    Rnet         = Qle + Qh

    #print(Rnet)

    EF          = pd.DataFrame(cable.variables['Fwsoil'][:,0,0],columns=['EF'])
    EF['dates'] = Time

    SM = read_SM_top_mid_bot_hourly(fcable, ring, layer)

    #print(SM)

    # exclude rainday and the after two days of rain
    day = np.zeros((len(Tair_daily)), dtype=bool)

    for i in np.arange(0,len(Tair_daily)):
        if (Tair_daily.values[i] >= 35.): # and Rainf.values[i] == 0.):
            day[i]   = True

    # calculate heatwave event
    HW = [] # create empty list

    i = 0

    #while i < len(Tair_daily)-2:
    while i < len(Tair_daily)-1:
        HW_event = []

        if (np.all([day[i:i+3]])):

            #day_start = Tair_daily.index[i-2]
            day_start = Tair_daily.index[i-1]
            i = i + 3

            while day[i]:

                i += 1

            else:
                #if i+2 < len(Tair_daily.index):
                #    day_end = Tair_daily.index[i+2] # the third day after heatwave
                #elif i+1 < len(Tair_daily.index):
                if i+1 < len(Tair_daily.index):
                    day_end = Tair_daily.index[i+1]
                else:
                    day_end = Tair_daily.index[i]

                Tair_event  = Tair[np.all([Tair.index >= day_start,  Tair.index < day_end],axis=0)]
                Rainf_event = Rainf[np.all([Tair.index >= day_start, Tair.index < day_end],axis=0)]
                Qle_event   = Qle[np.all([Tair.index >= day_start,   Tair.index < day_end],axis=0)]
                Qh_event    = Qh[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]
                EF_event    = EF[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]
                SM_event    = SM[np.all([Tair.index >= day_start,    Tair.index < day_end],axis=0)]

                for hour_num in np.arange(len(Tair_event)):
                    hour_in_event = ( Tair_event.index[hour_num],
                                      Tair_event['Tair'].values[hour_num],
                                      Rainf_event['Rainf'].values[hour_num],
                                      Qle_event['cable'].values[hour_num],
                                      Qh_event['cable'].values[hour_num],
                                      EF_event['EF'].values[hour_num],
                                      SM_event['SM_top'].values[hour_num],
                                      SM_event['SM_mid'].values[hour_num],
                                      SM_event['SM_bot'].values[hour_num],
                                      SM_event['SM_all'].values[hour_num],
                                      SM_event['SM_15m'].values[hour_num] )
                    HW_event.append(hour_in_event)

            HW.append(HW_event)
        else:
            i += 1
    #print(HW[0])

    return HW

def plot_single_HW_event(time_scale, case_labels, i, date, Tair, Rainf, Qle, Qh, EF, SM_top, SM_mid, SM_bot, SM_all, SM_15m):

    # ======================= Plot setting ============================
    if time_scale == "daily":
        fig = plt.figure(figsize=[11,17.5])
    elif time_scale == "hourly":
        fig = plt.figure(figsize=[13,17.5])

    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.1)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # choose colormap
    colors = cm.Set2(np.arange(0,len(case_labels)))
    ls     = ['-','--','-','--','-','--','-','--']

    ax1  = fig.add_subplot(511)
    ax2  = fig.add_subplot(512,sharex=ax1)
    ax3  = fig.add_subplot(513,sharex=ax2)
    ax4  = fig.add_subplot(514,sharex=ax3)
    ax5  = fig.add_subplot(515,sharex=ax4)

    x    = date
    #ax6  = ax1.twinx()

    if time_scale == "daily":
        width  = 0.6
    elif time_scale == "hourly":
        width  = 1/48

    ax1.plot(x, Tair,   c="black", lw=1.5, ls="-", label="Air Temperature")#.rolling(window=30).mean()

    for case_num in np.arange(len(case_labels)):
        print(case_num)
        ax2.plot(x, EF[case_num, :],  c=colors[case_num], lw=1.5, ls=ls[case_num], label=case_labels[case_num])#.rolling(window=30).mean()
        ax3.plot(x, Qle[case_num, :], c=colors[case_num], lw=1.5, ls=ls[case_num], label=case_labels[case_num])#.rolling(window=30).mean()
        ax4.plot(x, Qh[case_num, :],  c=colors[case_num], lw=1.5, ls=ls[case_num], label=case_labels[case_num]) #, label=case_labels)#.rolling(window=30).mean()
        ax5.plot(x, SM_15m[case_num, :], c=colors[case_num], lw=1.5, ls=ls[case_num], label=case_labels[case_num]) #*1500.#.rolling(window=30).mean()

    if time_scale == "daily":
        ax1.set_ylabel('Max Air Temperature (°C)')
        ax1.set_ylim(20, 45)
    elif time_scale == "hourly":
        ax1.set_ylabel('$Tair$ (°C)')
        ax1.set_ylim(10, 45)

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim(date[0],date[-1])
    ax1.axhline(y=35.,c=almost_black, ls="--")
    #ax1.spines['top'].set_visible(False)
    #ax1.spines['right'].set_visible(False)
    #ax1.spines['bottom'].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel("$EF$")
    ax2.axis('tight')
    ax2.set_xlim(date[0],date[-1])
    if time_scale == "daily":
        ax2.set_ylim(0.,1.8)
    elif time_scale == "hourly":
        ax2.set_ylim(0,1.1)
    #ax2.spines['top'].set_visible(False)
    #ax2.spines['right'].set_visible(False)
    #ax2.spines['bottom'].set_visible(False)
    #ax2.get_xaxis().set_visible(False)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel('$Q_{E}$ (W m$^{-2}$)')
    ax3.axis('tight')
    ax3.set_xlim(date[0],date[-1])
    if time_scale == "daily":
        ax3.set_ylim(-50.,220)
    elif time_scale == "hourly":
        ax3.set_ylim(-40.,395.)
    #ax3.spines['top'].set_visible(False)
    #ax3.spines['right'].set_visible(False)
    #ax3.spines['bottom'].set_visible(False)
    #ax3.get_xaxis().set_visible(False)
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax3.legend( loc='best', frameon=False)

    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set_ylabel('$Q_{H}$ (W m$^{-2}$)')
    ax4.axis('tight')
    ax4.set_xlim(date[0],date[-1])
    if time_scale == "daily":
        ax4.set_ylim(-50.,220)
    elif time_scale == "hourly":
        ax4.set_ylim(-40.,395.)
    #ax4.spines['top'].set_visible(False)
    #ax4.spines['right'].set_visible(False)
    #ax4.spines['bottom'].set_visible(False)
    #ax4.get_xaxis().set_visible(False)
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax5.get_xticklabels(), visible=True)
    ax5.set_ylabel("$θ$ in 1.5m (m$^{3}$ m$^{-3}$)") #(m$^{3}$ m$^{-3}$)")
    ax5.axis('tight')
    #ax5.legend()
    ax5.set_xlim(date[0],date[-1])
    if time_scale == "daily":
        ax5.set_ylim(0.18,0.32)
        plt.suptitle('Heatwave in %s ~ %s ' % (str(date[2]), str(date[-3])))
    elif time_scale == "hourly":
        ax5.set_ylim(0.08,0.31)
    #ax5.spines['top'].set_visible(False)
    #ax5.spines['right'].set_visible(False)
    ax5.text(0.02, 0.95, '(e)', transform=ax5.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # ax6.set_ylabel('P (mm d$^{-1}$)')
    # ax6.bar(x, Rainf,  width, color='royalblue', alpha = 0.5, label='Rainfall')
    # if time_scale == "daily":
    #     ax6.set_ylim(0., 30.)
    # elif time_scale == "hourly":
    #     ax6.set_ylim(0, 10.)
    # ax6.spines['top'].set_visible(False)
    # ax6.spines['right'].set_visible(False)
    # ax6.spines['bottom'].set_visible(False)
    # ax6.get_xaxis().set_visible(False)

    fig.savefig("../plots/EucFACE_Heatwave_%s" % str(i) , bbox_inches='tight', pad_inches=0.02)

def plot_EF_SM_HW(fcables, case_labels, layers, ring, time_scale):

    # =========== Calc HW events ==========
    # save all cases and all heatwave events
    # struction : 1st-D  2st-D  3st-D  4st-D
    #             case   event  day    variables

    HW_all   = []
    case_sum = len(fcables)

    for case_num in np.arange(case_sum):
        if time_scale == "daily":
            HW = find_Heatwave(fcables[case_num], ring, layers[case_num])
        elif time_scale == "hourly":
            HW = find_Heatwave_hourly(fcables[case_num], ring, layers[case_num])
        HW_all.append(HW)
    #print(HW_all)
    #print(HW_all[0][1])

    # ============ Read vars ==============
    event_sum = len(HW_all[0])

    for event_num in np.arange(event_sum):

        day_sum = len(HW_all[0][event_num])
        if time_scale == "daily":
            date   = np.zeros(day_sum, dtype='datetime64[D]')
        elif time_scale == "hourly":
            date   = np.zeros(day_sum, dtype='datetime64[ns]')
        Tair   = np.zeros(day_sum)
        Rainf  = np.zeros(day_sum)
        Qle    = np.zeros([case_sum,day_sum])
        Qh     = np.zeros([case_sum,day_sum])
        EF     = np.zeros([case_sum,day_sum])
        SM_top = np.zeros([case_sum,day_sum])
        SM_mid = np.zeros([case_sum,day_sum])
        SM_bot = np.zeros([case_sum,day_sum])
        SM_all = np.zeros([case_sum,day_sum])
        SM_15m = np.zeros([case_sum,day_sum])

        # loop days in one event
        for day_num in np.arange(day_sum):
            date[day_num]      = HW_all[0][event_num][day_num][0].to_datetime64()
            #print(date[day_num])
            Tair[day_num]      = HW_all[0][event_num][day_num][1]
            Rainf[day_num]     = HW_all[0][event_num][day_num][2]
            #print(date)
            for case_num in np.arange(case_sum):

                Qle[case_num,day_num]     =  HW_all[case_num][event_num][day_num][3]
                Qh[case_num,day_num]      =  HW_all[case_num][event_num][day_num][4]
                EF[case_num,day_num]      =  HW_all[case_num][event_num][day_num][5]
                SM_top[case_num,day_num]  =  HW_all[case_num][event_num][day_num][6]
                SM_mid[case_num,day_num]  =  HW_all[case_num][event_num][day_num][7]
                SM_bot[case_num,day_num]  =  HW_all[case_num][event_num][day_num][8]
                SM_all[case_num,day_num]  =  HW_all[case_num][event_num][day_num][9]
                SM_15m[case_num,day_num]  =  HW_all[case_num][event_num][day_num][10]

        plot_single_HW_event(time_scale, case_labels, event_num, date, Tair, Rainf, Qle, Qh, EF, SM_top, SM_mid, SM_bot, SM_all, SM_15m)

def get_day_time_Qle_Qh_EF(fcable):

    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    Tair = pd.DataFrame(cable.variables['Tair'][:,0,0]-273.15,columns=['Tair'])
    Tair['dates'] = Time
    Tair = Tair.set_index('dates')
    Tair = Tair.resample("D").agg('max')

    Qle          = pd.DataFrame(cable.variables['Qle'][:,0,0],columns=['cable'])
    Qle['dates'] = Time
    Qle          = Qle.set_index('dates')

    Qh           = pd.DataFrame(cable.variables['Qh'][:,0,0],columns=['cable'])
    Qh['dates']  = Time
    Qh           = Qh.set_index('dates')

    Rnet         = Qle + Qh

    #print(Rnet)

    EF          = pd.DataFrame(Qle['cable'].values/Rnet['cable'].values, columns=['EF'])
    EF['dates'] = Time
    EF          = EF.set_index('dates')
    #EF['EF']    = np.where(np.all([Qle["cable"].values > 1., Qh["cable"].values > 1.],axis=0 ), EF['EF'].values, float('nan'))
    Qle['cable']= np.where(np.all([Qle.index.hour >= 9., Qle.index.hour <= 16.],axis=0 ), Qle['cable'].values, float('nan'))
    Qh['cable'] = np.where(np.all([Qh.index.hour >= 9., Qh.index.hour <= 16.  ],axis=0 ), Qh['cable'].values, float('nan'))
    EF['EF']    = np.where(np.all([EF.index.hour >= 9., EF.index.hour <= 16.  ],axis=0 ), EF['EF'].values, float('nan'))
    #print(Qle)
    Qle         = Qle.resample("D").agg('mean')
    Qh          = Qh.resample("D").agg('mean')
    EF          = EF.resample("D").agg('mean')
    #print(Qle)
    return Tair, Qle, Qh, EF;

def find_all_Heatwave_days(fcable):

    Tair, Qle, Qh, EF = get_day_time_Qle_Qh_EF(fcable)

    # exclude rainday and the after two days of rain
    day = np.zeros((len(Tair)), dtype=bool)


    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    Rain = pd.DataFrame(cable.variables['Rainf'][:,0,0]*60*30,columns=['Rain'])
    Rain['dates'] = Time
    Rain = Rain.set_index('dates')
    Rain = Rain.resample("D").agg('sum')

    for i in np.arange(1,len(Tair)):
        if (Tair.values[i] >= 35. and Rain.values[i] < 0.1 and Rain.values[i-1] < 0.1): # and Rainf.values[i] == 0.):
            day[i]   = True

    # calculate heatwave event
    HW_event = [] # create empty list

    i = 0
    while i < len(Tair)-2:
        if (np.all([day[i:i+3]])):
            # consistent 3 days > 35 degree
            for j in np.arange(i,i+3):
                event = ( Qle.index[j], Qle['cable'].values[j], Qh['cable'].values[j],
                          EF['EF'].values[j])
                HW_event.append(event)
            i = i + 3

            while day[i]:
                # consistent more days > 35 degree
                event = ( Qle.index[i], Qle['cable'].values[i], Qh['cable'].values[i],
                          EF['EF'].values[i] )
                print(event)
                HW_event.append(event)
                i += 1
        else:
            i += 1

    return HW_event

def boxplot_Qle_Qh_EF_HW(fcables, case_labels, time_scale):

    time_scale = "hw_days" # "hw_days" "all_days"
    case_sum = len(fcables)

    if time_scale == "hw_days":
        day_sum = len(find_all_Heatwave_days(fcables[0]))
        Qle = np.zeros([day_sum, case_sum])
        Qh  = np.zeros([day_sum, case_sum])
        EF  = np.zeros([day_sum, case_sum])

        for case_num in np.arange(case_sum):
            HW_event = find_all_Heatwave_days(fcables[case_num])
            for day_num in np.arange(day_sum):
                Qle[day_num,case_num] = HW_event[day_num][1]
                Qh[day_num,case_num]  = HW_event[day_num][2]
                EF[day_num,case_num]  = HW_event[day_num][3]

    elif time_scale == "all_days":
        Tair, Qle_tmp, Qh_tmp, EF_tmp = get_day_time_Qle_Qh_EF(fcables[0])
        day_sum = len(Qle_tmp)
        print(day_sum)
        Qle = np.zeros([day_sum, case_sum])
        Qh  = np.zeros([day_sum, case_sum])
        EF  = np.zeros([day_sum, case_sum])
        for case_num in np.arange(case_sum):
            Tair, Qle_tmp, Qh_tmp, EF_tmp = get_day_time_Qle_Qh_EF(fcables[case_num])
            #print(Qle_tmp.values)
            Qle[:,case_num] = Qle_tmp['cable'].values
            Qh[:,case_num]  = Qh_tmp['cable'].values
            EF[:,case_num]  = EF_tmp['EF'].values
        #print(Qle)
    fig = plt.figure(figsize=[7,9])

    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 14
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

    ax1  = fig.add_subplot(311)
    ax2  = fig.add_subplot(312)
    ax3  = fig.add_subplot(313)

    colors = cm.tab20(np.linspace(0,1,len(case_labels)))
    if time_scale == "hw_days":
        ax1.boxplot(Qle, widths = 0.4, showfliers=False)# c=colors[case_num], label=case_labels[case_num])
        ax2.boxplot(Qh, widths = 0.4, showfliers=False)# c=colors[case_num], label=case_labels[case_num])
        ax3.boxplot(EF, widths = 0.4, showfliers=False)# c=colors[case_num], label=case_labels[case_num])
    elif time_scale == "all_days":
        ax1.boxplot(Qle[:-2,:], widths = 0.4, showfliers=False)# c=colors[case_num], label=case_labels[case_num])
        ax2.boxplot(Qh[:-2,:], widths = 0.4, showfliers=False)# c=colors[case_num], label=case_labels[case_num])
        ax3.boxplot(EF[:-2,:], widths = 0.4, showfliers=False)# c=colors[case_num], label=case_labels[case_num])

    plt.setp(ax1.get_xticklabels(), visible=False)
    #ax1.set_xlim(date[0],date[-1])
    ax1.set_ylabel('Q$_{E}$ (W m$^{-2}$)')
    ax1.axis('tight')
    #ax1.set_xlim(date[0],date[-1])
    ax1.set_ylim(-50.,600.)
    ax1.set_xticks(np.arange(1,len(case_labels)+1,1))
    ax1.set_xticklabels(case_labels)

    plt.setp(ax2.get_xticklabels(), visible=False)
    #ax2.set_xlim(date[0],date[-1])
    ax2.set_ylabel('Q$_{H}$ (W m$^{-2}$)')
    ax2.axis('tight')
    #ax2.set_xlim(date[0],date[-1])
    ax2.set_ylim(-120.,400.)
    ax2.set_xticks(np.arange(1,len(case_labels)+1,1))
    ax2.set_xticklabels(case_labels)

    plt.setp(ax3.get_xticklabels(), visible=True)
    #ax3.set_xlim(date[0],date[-1])
    ax3.set_ylabel("EF")
    ax3.axis('tight')
    #ax3.set_xlim(date[0],date[-1])
    ax3.set_ylim(-0.3,1.6)
    #ax4.legend()
    ax3.set_xticks(np.arange(1,len(case_labels)+1,1))
    ax3.set_xticklabels(case_labels)

    if time_scale == "hw_days":
        fig.savefig("../plots/EucFACE_Qle_Qh_EF_HW" , bbox_inches='tight', pad_inches=0.02)
    elif time_scale == "all_days":
        fig.savefig("../plots/EucFACE_Qle_Qh_EF_all" , bbox_inches='tight', pad_inches=0.02)

def group_boxplot_Qle_Qh_EF_HW(fcables, case_labels):

    case_sum = len(fcables)
    day_hw   = len(find_all_Heatwave_days(fcables[0]))
    print("day_hw %f" % day_hw)
    print(find_all_Heatwave_days(fcables[0]))

    Tair, Qle_tmp, Qh_tmp, EF_tmp = get_day_time_Qle_Qh_EF(fcables[0])
    day_all = len(Qle_tmp)

    Tair, Qle_tmp, Qh_tmp, EF_tmp = get_day_time_Qle_Qh_EF(fcables[0])
    day_sum = len(Qle_tmp[np.any([Qle_tmp.index.month == 1, Qle_tmp.index.month == 2, Qle_tmp.index.month == 12],axis=0)])
    print(Qle_tmp[np.any([Qle_tmp.index.month == 1, Qle_tmp.index.month == 2, Qle_tmp.index.month == 12],axis=0)])

    Qle_hw = np.zeros([day_hw, case_sum])
    Qh_hw  = np.zeros([day_hw, case_sum])
    EF_hw  = np.zeros([day_hw, case_sum])

    Qle_all = np.zeros([day_all, case_sum])
    Qh_all  = np.zeros([day_all, case_sum])
    EF_all  = np.zeros([day_all, case_sum])

    Qle_sum = np.zeros([day_sum, case_sum])
    Qh_sum  = np.zeros([day_sum, case_sum])
    EF_sum  = np.zeros([day_sum, case_sum])

    for case_num in np.arange(case_sum):
        HW_event = find_all_Heatwave_days(fcables[case_num])
        for day_num in np.arange(day_hw):
            Qle_hw[day_num,case_num] = HW_event[day_num][1]
            Qh_hw[day_num,case_num]  = HW_event[day_num][2]
            EF_hw[day_num,case_num]  = HW_event[day_num][3]

    for case_num in np.arange(case_sum):
        Tair, Qle_tmp, Qh_tmp, EF_tmp = get_day_time_Qle_Qh_EF(fcables[case_num])
        Qle_all[:,case_num] = Qle_tmp['cable'].values
        Qh_all[:,case_num]  = Qh_tmp['cable'].values
        EF_all[:,case_num]  = EF_tmp['EF'].values

    for case_num in np.arange(case_sum):
        Tair, Qle_tmp, Qh_tmp, EF_tmp = get_day_time_Qle_Qh_EF(fcables[case_num])
        Qle_sum[:,case_num] = Qle_tmp[np.any([Qle_tmp.index.month == 1, Qle_tmp.index.month == 2, Qle_tmp.index.month == 12],axis=0)]['cable'].values
        Qh_sum[:,case_num]  = Qh_tmp[np.any([Qh_tmp.index.month == 1, Qh_tmp.index.month == 2, Qh_tmp.index.month == 12],axis=0)]['cable'].values
        EF_sum[:,case_num]  = EF_tmp[np.any([EF_tmp.index.month == 1, EF_tmp.index.month == 2, EF_tmp.index.month == 12],axis=0)]['EF'].values

    hw           = pd.DataFrame(np.zeros((day_hw+day_all)*case_sum),columns=['Qle'])
    hw['Qh']     = np.zeros((day_hw+day_all)*case_sum)
    hw['EF']     = np.zeros((day_hw+day_all)*case_sum)
    hw['day']    = [''] * ((day_hw+day_all)*case_sum)
    hw['exp']    = [''] * ((day_hw+day_all)*case_sum)

    s = 0

    for case_num in np.arange(case_sum):

        e  = s+day_hw
        hw['Qle'].iloc[s:e] = Qle_hw[:,case_num]
        hw['Qh'].iloc[s:e]  = Qh_hw[:,case_num]
        hw['EF'].iloc[s:e]  = EF_hw[:,case_num]
        hw['day'].iloc[s:e] = ['heatwave'] * day_hw
        hw['exp'].iloc[s:e] = [ case_labels[case_num]] * day_hw

        s  = e

        '''
        # all days
        e  = s+day_all
        hw['Qle'].iloc[s:e] = Qle_all[:,case_num]
        hw['Qh'].iloc[s:e]  = Qh_all[:,case_num]
        hw['EF'].iloc[s:e]  = EF_all[:,case_num]
        hw['day'].iloc[s:e] = ['all'] * day_all
        hw['exp'].iloc[s:e] = [ case_labels[case_num]] * day_all
        '''

        # summer
        e  = s+day_sum
        hw['Qle'].iloc[s:e] = Qle_sum[:,case_num]
        hw['Qh'].iloc[s:e]  = Qh_sum[:,case_num]
        hw['EF'].iloc[s:e]  = EF_sum[:,case_num]
        hw['day'].iloc[s:e] = ['summer'] * day_sum
        hw['exp'].iloc[s:e] = [ case_labels[case_num]] * day_sum

        s  =  e

    # ======================= Plot setting ============================
    fig = plt.figure(figsize=[7,11])
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.0)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    ax1  = fig.add_subplot(311)
    ax2  = fig.add_subplot(312,sharex=ax1)
    ax3  = fig.add_subplot(313,sharex=ax2)

    sns.boxplot(x="exp", y="Qle", hue="day", data=hw, palette="Set2",
                order=case_labels,  width=0.7, hue_order=['heatwave','summer'],
                ax=ax1, showfliers=False, color=almost_black)
    sns.boxplot(x="exp", y="Qh", hue="day", data=hw, palette="Set2",
                order=case_labels,  width=0.7, hue_order=['heatwave','summer'],
                ax=ax2, showfliers=False, color=almost_black)
    sns.boxplot(x="exp", y="EF", hue="day", data=hw, palette="Set2",
                order=case_labels,  width=0.7, hue_order=['heatwave','summer'],
                ax=ax3, showfliers=False, color=almost_black)

    ax1.set_ylabel('Q$_{E}$ (W m$^{-2}$)')
    ax1.axis('tight')
    #ax1.set_xlim(date[0],date[-1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylim(-50.,550.)
    ax1.legend(loc='best', frameon=False)
    #ax1.axhline(y=np.median(Qle_hw[:,0]) ,color=almost_black, ls="--")
    ax1.axhline(y=np.mean(Qle_hw[:,0]) ,color=almost_black, ls="--")
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.get_xaxis().set_visible(False)

    ax2.set_ylabel('Q$_{H}$ (W m$^{-2}$)')
    ax2.axis('tight')
    ax2.legend().set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    #ax2.set_xlim(date[0],date[-1])
    ax2.set_ylim(-120.,400.)
    #ax2.axhline(y=np.median(Qh_hw[:,0]) , color=almost_black, ls="--")
    ax2.axhline(y=np.mean(Qh_hw[:,0]) , color=almost_black, ls="--")
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax2.get_xaxis().set_visible(False)

    ax3.set_ylabel("EF")
    ax3.axis('tight')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.legend().set_visible(False)
    #ax3.set_xlim(date[0],date[-1])
    ax3.set_ylim(-0.3,1.6)
    #plt.legend()
    #ax3.axhline(y=np.median(EF_hw[:,0]) ,color=almost_black, ls="--")
    ax3.axhline(y=np.mean(EF_hw[:,0]) ,color=almost_black, ls="--")
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    fig.savefig("../plots/EucFACE_Qle_Qh_EF_summer_group_boxplot" , bbox_inches='tight', pad_inches=0.02)

def plot_case_study_HW_event(fcables, fcables_re, case_labels, ring, layers):

    # ======================= Plot setting ============================
    #fig = plt.figure(figsize=[13,17.5])
    fig = plt.figure(figsize=[13,7])
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.1)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # choose colormap
    #colors = cm.Set2(np.arange(0,len(case_labels)))
    colors = cm.Set2(np.arange(0,6))
    ls     = ['-','--','-','--','-','--']

    #ax1  = fig.add_subplot(511)
    #ax2  = fig.add_subplot(512,sharex=ax1)
    #ax3  = fig.add_subplot(513,sharex=ax2)
    #ax4  = fig.add_subplot(514,sharex=ax3)
    #ax5  = fig.add_subplot(515,sharex=ax4)
    ax3  = fig.add_subplot(211)
    ax4  = fig.add_subplot(212,sharex=ax3)
    # ========================= Read data ============================
    HW_all   = []
    case_sum = len(fcables)

    for case_num in np.arange(case_sum):
        HW = find_Heatwave_hourly(fcables[case_num], ring, layers[case_num])
        HW_all.append(HW)

    # read heatwave event 6 2018-1-19 - 2018-1-22
    event_num = 6
    day_sum = len(HW_all[0][event_num])
    Qle    = np.zeros([case_sum,day_sum])
    Qh     = np.zeros([case_sum,day_sum])
    EF     = np.zeros([case_sum,day_sum])
    SM_15m = np.zeros([case_sum,day_sum])

    # loop days in one event
    for day_num in np.arange(day_sum):
        for case_num in np.arange(case_sum):
            Qle[case_num,day_num]     =  HW_all[case_num][event_num][day_num][3]
            Qh[case_num,day_num]      =  HW_all[case_num][event_num][day_num][4]
            EF[case_num,day_num]      =  HW_all[case_num][event_num][day_num][5]
            SM_15m[case_num,day_num]  =  HW_all[case_num][event_num][day_num][10]

    # ======================= Read restart simulation ============================
    time_steps = 48*6
    case_sum   = len(fcables_re)
    Tair_re    = np.zeros([case_sum,time_steps])
    Rainf_re   = np.zeros([case_sum,time_steps])
    Qle_re     = np.zeros([case_sum,time_steps])
    Qh_re      = np.zeros([case_sum,time_steps])
    Rnet_re    = np.zeros([case_sum,time_steps])
    EF_re      = np.zeros([case_sum,time_steps])
    SM_15m_re  = np.zeros([case_sum,time_steps])

    for case_num in np.arange(case_sum):

        cable = nc.Dataset(fcables_re[case_num], 'r')
        Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

        Tair_re[case_num,:]  = cable.variables['Tair'][:,0,0]-273.15
        Rainf_re[case_num,:] = cable.variables['Rainf'][:,0,0]*1800.
        Qle_re[case_num,:]   = cable.variables['Qle'][:,0,0]
        Qh_re[case_num,:]    = cable.variables['Qh'][:,0,0]
        Rnet_re[case_num,:]  = Qle_re[case_num,:] + Qh_re[case_num,:]

        EF_pd          = pd.DataFrame(Qle_re[case_num,:]/Rnet_re[case_num,:], columns=['EF'])
        EF_pd['dates'] = Time
        EF_pd          = EF_pd.set_index('dates')
        EF_re[case_num,:] = np.where(np.all([EF_pd.index.hour >= 9., EF_pd.index.hour <= 16.,
                         EF_pd['EF'].values <= 5. ],axis=0 ),
                         EF_pd['EF'].values, float('nan'))
        SM_15m_re[case_num,:] = cable.variables['SoilMoist'][:,0,0,0]*0.15
        for i in np.arange(1,10):
            SM_15m_re[case_num,:]  = SM_15m_re[case_num,:] + cable.variables['SoilMoist'][:,i,0,0]*0.15
        SM_15m_re[case_num,:]  = SM_15m_re[case_num,:]/1.5

    # ========================== Plotting ================================
    x    = np.arange(0,time_steps)

    width  = 1/48

    #ax1.plot(x, Tair_re[0, :],   c="black", lw=1.5, ls="-", label="Air Temperature")#.rolling(window=30).mean()

    for case_num in np.arange(len(case_labels)):
        print(case_num)
        #ax2.plot(x, EF[case_num, :],  c=colors[case_num+2], lw=1.5, ls='-', label=case_labels[case_num])#.rolling(window=30).mean()
        ax3.plot(x, Qle[case_num, :], c=colors[case_num+2], lw=1.5, ls='-', label=case_labels[case_num])#.rolling(window=30).mean()
        ax4.plot(x, Qh[case_num, :],  c=colors[case_num+2], lw=1.5, ls='-', label=case_labels[case_num]) #, label=case_labels)#.rolling(window=30).mean()
        #ax5.plot(x, SM_15m[case_num, :], c=colors[case_num+2], lw=1.5, ls='-', label=case_labels[case_num]) #*1500.#.rolling(window=30).mean()

        #if case_num == 0:
        #ax2.plot(x, EF_re[case_num, :],  c=colors[case_num+2], lw=1.5, ls='--')#.rolling(window=30).mean()
        if case_num > 0:
            ax3.plot(x, Qle_re[case_num, :], c=colors[case_num+2], lw=1.5, ls='--')#.rolling(window=30).mean()
            ax4.plot(x, Qh_re[case_num, :],  c=colors[case_num+2], lw=1.5, ls='--') #, label=case_labels)#.rolling(window=30).mean()
            #ax5.plot(x, SM_15m_re[case_num, :], c=colors[case_num+2], lw=1.5, ls='--') #*1500.#.rolling(window=30).mean()

    cleaner_dates = ["2018-1-18","2018-1-19","2018-1-20","2018-1-21","2018-1-22","2018-1-23","2018-1-24"]
    xtickslocs    = [0,48,96,144,192,240,288]

    # ax1.set_ylabel('Tair (°C)')
    # ax1.set_ylim(10, 41)
    #
    # plt.setp(ax1.get_xticklabels(), visible=False)
    # ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    # ax1.set_xlim(0,288)
    # ax1.axhline(y=35.,c=almost_black, ls="--")
    # #ax1.spines['top'].set_visible(False)
    # #ax1.spines['right'].set_visible(False)
    # #ax1.spines['bottom'].set_visible(False)
    # ax1.get_xaxis().set_visible(False)
    # ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    #
    # plt.setp(ax2.get_xticklabels(), visible=False)
    # ax2.set_ylabel("EF")
    # ax2.axis('tight')
    # ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    # ax2.set_xlim(0,288)
    # ax2.set_ylim(0,1.1)
    # #ax2.spines['top'].set_visible(False)
    # #ax2.spines['right'].set_visible(False)
    # #ax2.spines['bottom'].set_visible(False)
    # #ax2.get_xaxis().set_visible(False)
    # ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel('$Q_{E}$ (W m$^{-2}$)')
    ax3.axis('tight')
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_xlim(0,288)
    ax3.set_ylim(-40.,395.)
    #ax3.spines['top'].set_visible(False)
    #ax3.spines['right'].set_visible(False)
    #ax3.spines['bottom'].set_visible(False)
    #ax3.get_xaxis().set_visible(False)
    #ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax3.text(0.02, 0.95, '(a)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax3.legend( loc='best', frameon=False)

    plt.setp(ax4.get_xticklabels(), visible=True)
    ax4.set_ylabel('$Q_{H}$ (W m$^{-2}$)')
    ax4.axis('tight')
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_xlim(0,288)
    ax4.set_ylim(-40.,395.)
    # #ax4.spines['top'].set_visible(False)
    # #ax4.spines['right'].set_visible(False)
    # #ax4.spines['bottom'].set_visible(False)
    # #ax4.get_xaxis().set_visible(False)
    # ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax4.text(0.02, 0.95, '(b)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # plt.setp(ax5.get_xticklabels(), visible=True)
    # ax5.set_ylabel("VWC in 1.5m (m$^{3}$ m$^{-3}$)") #(m$^{3}$ m$^{-3}$)")
    # ax5.axis('tight')
    # #ax5.legend()
    # ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    # ax5.set_xlim(0,288)
    # ax5.set_ylim(0.10,0.31)
    # #ax5.spines['top'].set_visible(False)
    # #ax5.spines['right'].set_visible(False)
    # ax5.text(0.02, 0.95, '(e)', transform=ax5.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # ax6.set_ylabel('P (mm d$^{-1}$)')
    # ax6.bar(x, Rainf,  width, color='royalblue', alpha = 0.5, label='Rainfall')
    # if time_scale == "daily":
    #     ax6.set_ylim(0., 30.)
    # elif time_scale == "hourly":
    #     ax6.set_ylim(0, 10.)
    # ax6.spines['top'].set_visible(False)
    # ax6.spines['right'].set_visible(False)
    # ax6.spines['bottom'].set_visible(False)
    # ax6.get_xaxis().set_visible(False)

    fig.savefig("../plots/EucFACE_Heatwave_2018-1-18-23_LH-SH" , bbox_inches='tight', pad_inches=0.02)

def plot_case_study_HW_event_beta(fcables, fcables_re, case_labels, ring, layers):

    # ======================= Plot setting ============================
    #fig = plt.figure(figsize=[13,17.5])
    fig = plt.figure(figsize=[13,14])
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.1)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']  = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # choose colormap
    #colors = cm.Set2(np.arange(0,len(case_labels)))
    colors = cm.Set2(np.arange(0,6))
    ls     = ['-','--','-','--','-','--']

    #ax1  = fig.add_subplot(511)
    #ax2  = fig.add_subplot(512,sharex=ax1)
    #ax3  = fig.add_subplot(513,sharex=ax2)
    #ax4  = fig.add_subplot(514,sharex=ax3)
    #ax5  = fig.add_subplot(515,sharex=ax4)
    ax2  = fig.add_subplot(411)
    ax3  = fig.add_subplot(412,sharex=ax2)
    ax4  = fig.add_subplot(413,sharex=ax3)
    ax5  = fig.add_subplot(414,sharex=ax4)
    # ========================= Read data ============================
    HW_all   = []
    case_sum = len(fcables)

    for case_num in np.arange(case_sum):
        HW = find_Heatwave_hourly_beta(fcables[case_num], ring, layers[case_num])
        HW_all.append(HW)

    # read heatwave event 6 2018-1-19 - 2018-1-22
    event_num = 6
    day_sum = len(HW_all[0][event_num])
    Qle    = np.zeros([case_sum,day_sum])
    Qh     = np.zeros([case_sum,day_sum])
    EF     = np.zeros([case_sum,day_sum])
    SM_15m = np.zeros([case_sum,day_sum])

    # loop days in one event
    for day_num in np.arange(day_sum):
        for case_num in np.arange(case_sum):
            Qle[case_num,day_num]     =  HW_all[case_num][event_num][day_num][3]
            Qh[case_num,day_num]      =  HW_all[case_num][event_num][day_num][4]
            EF[case_num,day_num]      =  HW_all[case_num][event_num][day_num][5]
            SM_15m[case_num,day_num]  =  HW_all[case_num][event_num][day_num][10]

    # ======================= Read restart simulation ============================
    time_steps = 48*6
    case_sum   = len(fcables_re)
    Tair_re    = np.zeros([case_sum,time_steps])
    Rainf_re   = np.zeros([case_sum,time_steps])
    Qle_re     = np.zeros([case_sum,time_steps])
    Qh_re      = np.zeros([case_sum,time_steps])
    Rnet_re    = np.zeros([case_sum,time_steps])
    EF_re      = np.zeros([case_sum,time_steps])
    SM_15m_re  = np.zeros([case_sum,time_steps])

    for case_num in np.arange(case_sum):

        cable = nc.Dataset(fcables_re[case_num], 'r')
        Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

        Tair_re[case_num,:]  = cable.variables['Tair'][:,0,0]-273.15
        Rainf_re[case_num,:] = cable.variables['Rainf'][:,0,0]*1800.
        Qle_re[case_num,:]   = cable.variables['Qle'][:,0,0]
        Qh_re[case_num,:]    = cable.variables['Qh'][:,0,0]
        Rnet_re[case_num,:]  = Qle_re[case_num,:] + Qh_re[case_num,:]

        EF_re[case_num,:] = cable.variables['Fwsoil'][:,0,0]
        SM_15m_re[case_num,:] = cable.variables['SoilMoist'][:,0,0,0]*0.15
        for i in np.arange(1,10):
            SM_15m_re[case_num,:]  = SM_15m_re[case_num,:] + cable.variables['SoilMoist'][:,i,0,0]*0.15
        SM_15m_re[case_num,:]  = SM_15m_re[case_num,:]/1.5

    # ========================== Plotting ================================
    x    = np.arange(0,time_steps)

    width  = 1/48

    #ax1.plot(x, Tair_re[0, :],   c="black", lw=1.5, ls="-", label="Air Temperature")#.rolling(window=30).mean()

    for case_num in np.arange(len(case_labels)):
        print(case_num)
        ax2.plot(x, EF[case_num, :],  c=colors[case_num+2], lw=1.5, ls='-', label=case_labels[case_num])#.rolling(window=30).mean()
        ax3.plot(x, Qle[case_num, :], c=colors[case_num+2], lw=1.5, ls='-', label=case_labels[case_num])#.rolling(window=30).mean()
        ax4.plot(x, Qh[case_num, :],  c=colors[case_num+2], lw=1.5, ls='-', label=case_labels[case_num]) #, label=case_labels)#.rolling(window=30).mean()
        ax5.plot(x, SM_15m[case_num, :], c=colors[case_num+2], lw=1.5, ls='-', label=case_labels[case_num]) #*1500.#.rolling(window=30).mean()

        #if case_num == 0:
        #
        #if case_num > 0:
        ax2.plot(x, EF_re[case_num, :],  c=colors[case_num+2], lw=1.5, ls='--')#.rolling(window=30).mean()
        ax3.plot(x, Qle_re[case_num, :], c=colors[case_num+2], lw=1.5, ls='--')#.rolling(window=30).mean()
        ax4.plot(x, Qh_re[case_num, :],  c=colors[case_num+2], lw=1.5, ls='--') #, label=case_labels)#.rolling(window=30).mean()
        ax5.plot(x, SM_15m_re[case_num, :], c=colors[case_num+2], lw=1.5, ls='--') #*1500.#.rolling(window=30).mean()

    cleaner_dates = ["2018-1-18","2018-1-19","2018-1-20","2018-1-21","2018-1-22","2018-1-23","2018-1-24"]
    xtickslocs    = [0,48,96,144,192,240,288]

    # ax1.set_ylabel('Tair (°C)')
    # ax1.set_ylim(10, 41)
    #
    # plt.setp(ax1.get_xticklabels(), visible=False)
    # ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    # ax1.set_xlim(0,288)
    # ax1.axhline(y=35.,c=almost_black, ls="--")
    # #ax1.spines['top'].set_visible(False)
    # #ax1.spines['right'].set_visible(False)
    # #ax1.spines['bottom'].set_visible(False)
    # ax1.get_xaxis().set_visible(False)
    # ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    #
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel("$β$")
    ax2.axis('tight')
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_xlim(0,48)
    ax2.set_ylim(0,1.1)
    #ax2.spines['top'].set_visible(False)
    #ax2.spines['right'].set_visible(False)
    #ax2.spines['bottom'].set_visible(False)
    #ax2.get_xaxis().set_visible(False)
    ax2.text(0.02, 0.95, '(a)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel('$Q_{E}$ (W m$^{-2}$)')
    ax3.axis('tight')
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_xlim(0,48)
    ax3.set_ylim(-40.,395.)
    #ax3.spines['top'].set_visible(False)
    #ax3.spines['right'].set_visible(False)
    #ax3.spines['bottom'].set_visible(False)
    #ax3.get_xaxis().set_visible(False)
    #ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax3.text(0.02, 0.95, '(b)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax3.legend( loc='best', frameon=False)

    plt.setp(ax4.get_xticklabels(), visible=True)
    ax4.set_ylabel('$Q_{H}$ (W m$^{-2}$)')
    ax4.axis('tight')
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_xlim(0,48)
    ax4.set_ylim(-40.,395.)
    # #ax4.spines['top'].set_visible(False)
    # #ax4.spines['right'].set_visible(False)
    # #ax4.spines['bottom'].set_visible(False)
    # #ax4.get_xaxis().set_visible(False)
    # ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax4.text(0.02, 0.95, '(c)', transform=ax4.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.setp(ax5.get_xticklabels(), visible=True)
    ax5.set_ylabel("VWC in 1.5m (m$^{3}$ m$^{-3}$)") #(m$^{3}$ m$^{-3}$)")
    ax5.axis('tight')
    #ax5.legend()
    ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax5.set_xlim(0,48)
    ax5.set_ylim(0.10,0.31)
    #ax5.spines['top'].set_visible(False)
    #ax5.spines['right'].set_visible(False)
    ax5.text(0.02, 0.95, '(d)', transform=ax5.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # ax6.set_ylabel('P (mm d$^{-1}$)')
    # ax6.bar(x, Rainf,  width, color='royalblue', alpha = 0.5, label='Rainfall')
    # if time_scale == "daily":
    #     ax6.set_ylim(0., 30.)
    # elif time_scale == "hourly":
    #     ax6.set_ylim(0, 10.)
    # ax6.spines['top'].set_visible(False)
    # ax6.spines['right'].set_visible(False)
    # ax6.spines['bottom'].set_visible(False)
    # ax6.get_xaxis().set_visible(False)

    fig.savefig("../plots/EucFACE_Heatwave_2018-1-18-23_LH-SH-beta" , bbox_inches='tight', pad_inches=0.02)
