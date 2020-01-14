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
from cable_get_var import *

def plot_2d(x, rmse, ref_var, case_name, txt_info):
    # _____________ Make plot _____________
    fig = plt.figure(figsize=(8,6))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax = fig.add_subplot(111)

    ax.plot(x, rmse)
    ax.set_title(txt_info)

    fig.savefig('EucFACE_error_surface_2d_%s_%s' % (ref_var, case_name), bbox_inches='tight',pad_inches=0.1)

def plot_3d(x, y, rmse, var_names, ref_var, case_name, txt_info):

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

    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(y, x)

    print(X.shape)
    print(Y.shape)
    print(rmse.shape)
    ax.plot_surface(X, Y, rmse, cmap='viridis', alpha=0.5, edgecolor='none')
    ax.set_title(txt_info)
    ax.view_init(azim=230)
    ax.set_xlabel('%s' % var_names[0])
    ax.set_ylabel('%s' % var_names[1])
    ax.set_zlabel('RMSE')

    fig.savefig("EucFACE_error_surface_3d_%s_%s" % (ref_var, case_name), bbox_inches='tight', pad_inches=0.1)

def plot_4d(x, y, z, rmse, var_names, ref_var, case_name, txt_info):

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

    ax = fig.add_subplot(111, projection='3d')

    X,XX,XXX = np.meshgrid(x, y, z)
    ax.scatter(X.ravel(), XX.ravel(), XXX.ravel(), c=rmse.ravel(), cmap=plt.hot(), alpha=0.5)#, edgecolors = 'none')
    #ax.scatter(x, x, x)#, cmap=plt.hot(), alpha=0.5, edgecolors = 'none')
    print("it is fine")
    ax.set_title(txt_info)
    ax.view_init(azim=200)
    ax.set_xlabel('%s' % var_names[0])
    ax.set_ylabel('%s' % var_names[1])
    ax.set_zlabel('%s' % var_names[2])
    print("it is ok")
    #plt.show()
    fig.savefig("EucFACE_error_surface_4d_%s_%s" % (ref_var, case_name), bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":

    dim_info = "4d" # "3d", "4d"
    layer    = "31uni"
    ring     = "amb"
    ref_var  = 'swc_all'

    if dim_info == "2d":
        var_names  = ["hyds"] # ['bch']#
        case_name  = "EucFACE_run_opt_31uni_hyds-bot"
        #var_values = ['15','20','25','30','35','40','45','50','55','60','65','70','75','80','85',
        #              '90','95','100','105','110','115','120','125','130']
        var_values = ['-100','-95','-90','-85','-80','-75','-70','-65','-60','-55',
                      '-50','-45','-40','-35','-30','-25','-20','-15','-10','-05',
                      '00','05','10','15','20','25','30','35','40','45','50']
        rmse = np.zeros(len(var_values))
        min_rmse = 9999.
        min_i    = -1
        x = np.linspace(-10.,5.,31)
        #x = np.linspace(1.5,13.,24)

        for i in np.arange(len(var_values)):
            output_file = "/srv/ccrc/data25/z5218916/cable/EucFACE/%s/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hyds^%s_fw-hie-exp/EucFACE_amb_out.nc"\
                         % (case_name, var_values[i])
            #output_file = "/srv/ccrc/data25/z5218916/cable/EucFACE/%s/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=%s_fw-hie-exp/EucFACE_amb_out.nc"\
            #              % (case_name, var_values[i])
            cable_var, obs_var = get_var_value(ref_var, output_file, layer, ring)

            rmse[i] = np.sqrt(np.mean((obs_var - cable_var)**2))

            if rmse[i] < min_rmse:
                min_rmse = rmse[i]
                min_i    = i

        txt_info= "when %s is %s, the min rmse is %s" % ( var_names, str(x[min_i]), str(min_rmse))
        print("txt_info is ", txt_info)
        plot_2d(x, rmse, ref_var, case_name, txt_info)

    if dim_info == "3d":
        var_names  = ["bch","hyds"]
        case_name  = "std" #"EucFACE_run_sen_31uni_2bch-mid-bot"
        var_values1 = ["20","30","40","50","60","70","80","90","100","110","120","130"]
        var_values2 = ["-80","-75","-70","-65","-60","-55","-50","-45","-40","-35","-30","-25","-20","-15","-10","-05",
                      "00","05","10","15","20"]
        ref_var     = ['swc_50','swc_all','trans','esoil','esoil2trans']

        x = np.linspace(2.,13.,12)
        y = np.linspace(-8.,2.,21)

        for k in np.arange(5):
            rmse = np.zeros((len(var_values1), len(var_values2)))
            min_rmse = 9999.
            min_i    = -1
            min_j    = -1
            for i in np.arange(len(var_values1)):
                for j in np.arange(len(var_values2)):
                    #output_file = "/srv/ccrc/data25/z5218916/cable/EucFACE/%s/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=%s-%s_fw-hie-exp_fix/EucFACE_amb_out.nc"\
                    #  % (case_name, var_values1[i],var_values2[j])
                    output_file = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_bch-hyds-top1/%s/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=%s-hyds^%s_std/EucFACE_amb_out.nc"\
                                   % (case_name, var_values1[i],var_values2[j])
                    cable_var, obs_var = get_var_value(ref_var[k], output_file, layer, ring)
                    rmse[i,j] = np.sqrt(np.mean((obs_var - cable_var)**2))
                    if rmse[i,j] < min_rmse:
                        min_rmse = rmse[i,j]
                        min_i    = i
                        min_j    = j

            txt_info= "when %s are %s and %s, the min rmse is %s" % ( var_names[0], str(x[min_i]), str(y[min_j]), str(min_rmse))
            print("txt_info is ", txt_info)
            plot_3d(x, y, rmse, var_names, ref_var[k], case_name, txt_info)

    if dim_info == "4d":
        var_names  = ["hyds","hyds","hyds"] # ["bch","bch","bch"] #
        case_name  = "EucFACE_run_sen_31uni_3hyds" #"EucFACE_run_sen_31uni_3bch" # "EucFACE_run_sen_31uni_3hyds-top50" # "EucFACE_run_sen_31uni_3bch-top50"

        var_values1 = ["-8","-7","-6","-5","-4","-3","-2","-1","0","1","2"]
        var_values2 = ["-8","-7","-6","-5","-4","-3","-2","-1","0","1","2"]
        var_values3 = ["-8","-7","-6","-5","-4","-3","-2","-1","0","1","2"]
        #["2","3","4","5","6","7","8","9","10","11","12","13"]
        #["2","3","4","5","6","7","8"]
        #["-8","-7","-6","-5","-4","-3","-2","-1","0","1","2"]

        x = np.linspace(-8.,2.,11)
        y = np.linspace(-8.,2.,11)
        z = np.linspace(-8.,2.,11)
        #np.linspace(2.,8.,7) # np.linspace(2.,13.,12) # np.linspace(-8.,2.,11)

        rmse     = np.zeros((len(var_values1), len(var_values2),len(var_values3)))
        min_rmse = 9999.
        min_i    = -1
        min_j    = -1
        min_k    = -1

        for i in np.arange(len(var_values1)):
            for j in np.arange(len(var_values2)):
                for k in np.arange(len(var_values3)):
                    print("i = ", i, " j = ",j , " k = ", k)
                    output_file = "/srv/ccrc/data25/z5218916/cable/EucFACE/%s/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hyds^%s-%s-%s_fw-hie-exp_fix/EucFACE_amb_out.nc"\
                                  % (case_name, var_values1[i],var_values2[j],var_values3[k])
                    cable_var, obs_var = get_var_value(ref_var, output_file, layer, ring)

                    rmse[i,j,k] = np.sqrt(np.mean((obs_var - cable_var)**2))

                    if rmse[i,j,k] < min_rmse:
                        min_rmse = rmse[i,j,k]
                        min_i    = i
                        min_j    = j
                        min_k    = k

        txt_info= "when %s are %s, %s and %s, the min rmse is %s" % ( var_names[0], str(x[min_i]), str(y[min_j]), str(z[min_k]), str(min_rmse))
        print("txt_info is ", txt_info)
        plot_4d(x, y, z, rmse, var_names, ref_var, case_name, txt_info)
