#!/usr/bin/env python

"""


"""

__author__  = "MU Mengyuan"
__version__ = "2020-09-10"
__email__   = 'mengyuan.mu815@gmail.com'

import os
import sys
import glob
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from sklearn.metrics import mean_squared_error


def get_cable_value(output_file, layer):

    cable = nc.Dataset(output_file, 'r')
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
    cable.close()
    return SoilMoist['SoilMoist'].values

def read_obs_swc(ring):

    fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
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

    return subset.values


def plot_3d(rmse, txt_info):

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

    #almost_black = '#262626'
    ## change the tick colors also to the almost black
    #plt.rcParams['ytick.color'] = almost_black
    #plt.rcParams['xtick.color'] = almost_black

    ## change the text colors also to the almost black
    #plt.rcParams['text.color'] = almost_black

    ## Change the default axis colors from black to a slightly lighter black,
    ## and a little thinner (0.5 instead of 1)
    #plt.rcParams['axes.edgecolor'] = almost_black
    #plt.rcParams['axes.labelcolor'] = almost_black

    ax = fig.add_subplot(111, projection='3d')

    #x = np.linspace(2.,13.,12)
    x = np.linspace(2.,8.,7)
    X,XX,XXX = np.meshgrid(x, x, x)
    print(X.ravel().shape)
    print(XX.ravel().shape)
    print(XXX.ravel().shape)
    print(rmse.ravel().shape)
    ax.scatter(X.ravel(), XX.ravel(), XXX.ravel(), c=rmse.ravel(), cmap=plt.hot(), alpha=0.5)#, edgecolors = 'none')
    #ax.scatter(x, x, x)#, cmap=plt.hot(), alpha=0.5, edgecolors = 'none')
    print("it is fine")
    ax.set_title(txt_info)
    ax.view_init(azim=230)
    ax.set_xlabel('bch top')
    ax.set_ylabel('bch mid')
    ax.set_zlabel('bch bottom')
    print("it is ok")
    #plt.show()
    fig.savefig("EucFACE_error_surface_bch-bch-bch-top50.png", bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":

    bch_names1 = ['2','3','4','5','6','7','8','9','10','11','12','13']
    bch_names2 = ['2','3','4','5','6','7','8']
    rmse      = np.zeros((len(bch_names1), len(bch_names2),len(bch_names2)))
    #rmse      =np.random.rand(len(bch_names), len(bch_names),len(bch_names))
    ring      = "amb"
    layer     = "31uni"

    min_rmse = 9999.
    min_i    = -1
    min_j    = -1
    min_k    = -1
    obs_swc = read_obs_swc(ring)

    for i,bch1 in enumerate(bch_names1):
        for j,bch2 in enumerate(bch_names2):
       	    for k,bch3 in enumerate(bch_names2):
                print("%s-%s-%s" % (bch1,bch2,bch3))
                #output_file = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_3bch/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=%s-%s-%s_fw-hie-exp_fix/EucFACE_amb_out.nc"\
                #              % (bch1, bch2, bch3)
                output_file = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_3bch-top50/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=%s-%s-%s_fw-hie-exp_fix/EucFACE_amb_out.nc"\
                              % (bch1, bch2, bch3)
                cable_swc   = get_cable_value(output_file, layer)

                rmse[i,j,k] = np.sqrt(np.mean((obs_swc - cable_swc)**2))

                if rmse[i,j,k] < min_rmse:
                    min_rmse = rmse[i,j,k]
                    min_i    = i
                    min_j    = j
                    min_k    = k

    txt_info= "when bch are %s, %s and %s, the min rmse is %s" % ( str(bch_names1[min_i]), str(bch_names2[min_j]), str(bch_names2[min_k]), str(min_rmse))
    print("txt_info is ", txt_info)

    #plot_3d(rmse, txt_info)
