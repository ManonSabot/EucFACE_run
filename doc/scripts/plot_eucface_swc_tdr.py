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

def plot_tdr(fcable, case_name, ring, layer):

    subset = read_obs_swc(ring)

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    if layer == "6":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.5-0.022-0.058-0.154) )/0.5
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

    swilt = np.zeros(len(SoilMoist))
    sfc = np.zeros(len(SoilMoist))
    ssat = np.zeros(len(SoilMoist))

    if layer == "6":
        swilt[:] = ( cable.variables['swilt'][0]*0.022 + cable.variables['swilt'][1]*0.058 \
                   + cable.variables['swilt'][2]*0.154 + cable.variables['swilt'][3]*(0.5-0.022-0.058-0.154) )/0.5
        sfc[:] = ( cable.variables['sfc'][0]*0.022   + cable.variables['sfc'][1]*0.058 \
                   + cable.variables['sfc'][2]*0.154 + cable.variables['sfc'][3]*(0.5-0.022-0.058-0.154) )/0.5
        ssat[:] = ( cable.variables['ssat'][0]*0.022 + cable.variables['ssat'][1]*0.058 \
                   + cable.variables['ssat'][2]*0.154+ cable.variables['ssat'][3]*(0.5-0.022-0.058-0.154) )/0.5
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

    ax = fig.add_subplot(111)

    x   = SoilMoist.index

    ax.plot(subset.index, subset.values,   c="green", lw=1.0, ls="-", label="tdr")
    ax.plot(x, SoilMoist.values,c="orange", lw=1.0, ls="-", label="swc")
    '''
    tmp1 = SoilMoist['SoilMoist'].loc[SoilMoist.index.isin(subset.index)]
    tmp2 = subset.loc[subset.index.isin(SoilMoist.index)]
    mask = np.isnan(tmp2)
    print(mask)
    tmp1 = tmp1[mask == False]
    tmp2 = tmp2[mask == False]

    cor_tdr = stats.pearsonr(tmp1,tmp2)
    mse_tdr = mean_squared_error(tmp2, tmp1)
    ax.set_title("r = % 5.3f , MSE = % 5.3f" %(cor_tdr[0], np.sqrt(mse_tdr)))
    print("-----------------------------------------------")
    print(mse_tdr)
    '''
    ax.plot(x, swilt,           c="black", lw=1.0, ls="-", label="swilt")
    ax.plot(x, sfc,             c="black", lw=1.0, ls="-.", label="sfc")
    ax.plot(x, ssat,            c="black", lw=1.0, ls=":", label="ssat")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax.set_ylabel("VWC (m3/m3)")
    ax.axis('tight')
    ax.set_ylim(0,0.5)
    ax.set_xlim(367,2739)
    ax.legend()

    fig.savefig("../plots/EucFACE_tdr_%s_%s" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)

def plot_Fwsoil(fcbl_def, fcbl_fw_def, fcbl_fw_hie, ring):

    fw1 = read_cable_var(fcbl_def, "Fwsoil")
    fw2 = read_cable_var(fcbl_fw_def, "Fwsoil")
    fw3 = read_cable_var(fcbl_fw_hie, "Fwsoil")

    fig = plt.figure(figsize=[15,10])

    ax  = fig.add_subplot(111)

    x = fw1.index

    ax.plot(x, fw1["cable"],   c="orange", lw=1.0, ls="-", label="Def_fw-std")
    ax.plot(x, fw2["cable"],   c="blue", lw=1.0, ls="-", label="Best_fw-std")
    ax.plot(x, fw3["cable"],   c="forestgreen", lw=1.0, ls="-", label="Best_fw-hie")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax.set_ylabel("Fwsoil (-)")
    ax.axis('tight')
    ax.set_ylim(0.,1.2)
    ax.set_xlim(367,2739)
    ax.legend()

    fig.savefig("../plots/EucFACE_fwsoil_comp_%s" % ring, bbox_inches='tight', pad_inches=0.1)

def plot_ET(fcable, case_name, ring):

    subs_Esoil = read_obs_esoil(ring)
    subs_Trans = read_obs_trans(ring)

    TVeg  = read_cable_var(fcable, "TVeg")
    ESoil = read_cable_var(fcable, "ESoil")

    fig = plt.figure(figsize=[15,10])

    ax  = fig.add_subplot(111)

    x = TVeg.index

    ax.plot(x, TVeg['cable'],     c="green", lw=1.0, ls="-", label="Trans") #.rolling(window=5).mean() .rolling(window=7).mean()
    ax.plot(x, ESoil['cable'],    c="orange", lw=1.0, ls="-", label="ESoil") #.rolling(window=7).mean()
    ax.scatter(subs_Trans.index, subs_Trans['obs'], marker='o', c='',edgecolors='blue', s = 2., label="Trans Obs") # subs['EfloorPred']
    ax.scatter(subs_Esoil.index, subs_Esoil['obs'], marker='o', c='',edgecolors='red', s = 2., label="ESoil Obs") # subs['EfloorPred']

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax.set_ylabel("ET ($mm d^{-1}$)")
    ax.axis('tight')
    ax.set_ylim(0.,4.5)
    ax.set_xlim(367,1098)
    ax.legend()

    fig.savefig("../plots/EucFACE_ET_%s_%s" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)

def plot_Rain(fcable, case_name, ring):

    Rain  = read_cable_var(fcable, "Rainf")
    fig   = plt.figure(figsize=[15,10])
    ax    = fig.add_subplot(111)
    x     = Rain.index
    width = 1.

    ax.bar(x, Rain['cable'], width, color='royalblue', label='Obs')

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.set_ylabel("Rain (mm/day)")
    ax.axis('tight')
    ax.set_ylim(0.,150.)
    ax.set_xlim(367,2739)

    fig.savefig("../plots/EucFACE_Rainfall", bbox_inches='tight', pad_inches=0.1)


def plot_Rain_Fwsoil(fcbl_def, fcbl_fw_def, fcbl_fw_hie, ring):

    fw1 = read_cable_var(fcbl_def, "Fwsoil")
    fw2 = read_cable_var(fcbl_fw_def, "Fwsoil")
    fw3 = read_cable_var(fcbl_fw_hie, "Fwsoil")
    Rain= read_cable_var(fcbl_def, "Rainf")

    fig = plt.figure(figsize=[15,10])

    ax1  = fig.add_subplot(211)
    ax2  = fig.add_subplot(212)

    x    = Rain.index
    width= 1.

    ax1.bar(x, Rain['cable'], width, color='royalblue', label='Obs')

    ax2.plot(x, fw1['cable'],   c="orange", lw=1.0, ls="-", label="Def_fw-std")
    ax2.plot(x, fw2['cable'],   c="blue", lw=1.0, ls="-", label="Best_fw-std")
    ax2.plot(x, fw3['cable'],   c="forestgreen", lw=1.0, ls="-", label="Best_fw-hie")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [367,732,1097,1462,1828,2193,2558]

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")
    ax1.set_ylabel("Rain (mm/day)")
    ax1.axis('tight')
    ax1.set_ylim(0.,150.)
    ax1.set_xlim(367,1098)

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_ylabel("Fwsoil (-)")
    ax2.axis('tight')
    ax2.set_ylim(0.,1.2)
    ax2.set_xlim(367,1098)
    ax2.legend()

    fig.savefig("../plots/EucFACE_Rain_Fwsoil_%s" % ring, bbox_inches='tight', pad_inches=0.1)


def read_cable_var(fcable, var_name):

    """
    read a var from CABLE output
    """

    print("carry on read_cable_var")
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    if var_name in ["TVeg", "ESoil", "Rainf"]:
        var = pd.DataFrame(cable.variables[var_name][:,0,0]*1800., columns=['cable'])
    else:
        var = pd.DataFrame(cable.variables[var_name][:,0,0], columns=['cable'])
    var['Date'] = Time
    var = var.set_index('Date')
    if var_name in ["TVeg", "ESoil", "Rainf"]:
        var = var.resample("D").agg('sum')
    elif var_name in ["Fwsoil"]:
        var = var.resample("D").agg('mean')
    var.index = var.index - pd.datetime(2011,12,31)
    var.index = var.index.days
    var = var.sort_values(by=['Date'])

    return var

def read_obs_esoil(ring):

    fobs_Esoil = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_underET.csv"

    est_esoil = pd.read_csv(fobs_Esoil, usecols = ['Ring','Date','wuTP'])
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

    return subs

def read_obs_trans(ring):

    fobs_Trans = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_sapflow.csv"

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
    subs = subs.rename({'volRing' : 'obs'}, axis='columns')

    return subs

def read_obs_swc(ring):

    fobs   = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
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
    subset = subset.rename({'swc.tdr' : 'obs'}, axis='columns')
    return subset


'''

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

'''
