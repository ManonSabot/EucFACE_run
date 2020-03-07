#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-10-06"
__changefrom__ = 'plot_eucface_swc_cable_vs_obs_obsved_dates-13-layer.py'

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
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from plot_eucface_swc_cable_vs_obs_profile import *

if __name__ == "__main__":

    layer   = "31uni"
    contour = False
    rings   = ["amb"]#"R1","R2","R3","R4","R5","R6",,"ele"
    plot_var= "difference" #"CABLE" # "difference"
    pyth    = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_3hyds/outputs"

    value1      = "2"
    sen_values2 = ["-2","-1","0","1","2","3"]
    #["0","05","1","15","2","25","3","35","4","45","5"]
    #["-8","-7","-6","-5","-4","-3","-2","-1","0","1","2"]
    sen_values3 = ["-2","-1","0","1","2","3"]
    sen_range  = ["-2.","-1.","0.","1.","2.","3."]
    #["0.","0.5","1.","1.5","2.","2.5","3.","3.5","4.","4.5","5."]
# ____________________ Plot obs _______________________

    fig, axes = plt.subplots(len(sen_values2), len(sen_values3),figsize=(80,40))

    #fig.figure(figsize=[40,40])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize']  = 2000
    plt.rcParams['font.size']       = 2000
    plt.rcParams['legend.fontsize'] = 2000
    plt.rcParams['xtick.labelsize'] = 2000
    plt.rcParams['ytick.labelsize'] = 2000

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

    if plot_var == "CABLE":
        cmap = plt.cm.viridis_r
    elif plot_var == "difference":
        cmap = plt.cm.BrBG

    Y    = np.arange(0,465,5)
    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [0,19,37,52,66,74,86]

    for ring in rings:
        for i,value2 in enumerate(sen_values2):
            for j,value3 in enumerate(sen_values3):
                case_name = "%s/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hyds^%s-%s-%s_litter" % (pyth, value1, value2, value3)

                fcable    ="%s/EucFACE_%s_out.nc" % (case_name, ring)
                grid_data, grid_cable, difference = read_profile_data(fcable, case_name, ring, contour, layer)
                if plot_var == "CABLE":
                    if contour:
                        levels   = np.arange(0.,52.,2.)
                        axes[i,j].contourf(grid_data, cmap=cmap, origin="upper", levels=levels)
                        Y_labels = np.flipud(Y)
                    else:
                        axes[i,j].imshow(grid_data, cmap=cmap, vmin=0, vmax=52, origin="upper", interpolation='nearest')
                        Y_labels = Y
                if plot_var == "difference":
                    if contour:
                        levels = np.arange(-30.,30.,2.)
                        axes[i,j].contourf(difference, cmap=cmap, origin="upper", levels=levels)
                        Y_labels = np.flipud(Y)
                    else:
                        axes[i,j].imshow(difference, cmap=cmap, vmin=-30, vmax=30, origin="upper", interpolation='nearest')
                        Y_labels = Y

                plt.setp(axes[i,j].get_xticklabels(), visible=False)
                plt.setp(axes[i,j].get_yticklabels(), visible=False)

                if i == len(sen_values2)-1:
                    plt.setp(axes[i,j].get_yticklabels(), visible=True)
                    axes[i,j].set_yticks(np.arange(len(Y))[::10])
                    axes[i,j].set_yticklabels(Y_labels[::10])
                    axes[i,j].set_xlabel("10 ^ %s" % sen_range[j])

                if j == 0:
                    plt.setp(axes[i,j].get_xticklabels(), visible=True)
                    axes[i,j].set(xticks=xtickslocs, xticklabels=cleaner_dates)
                    axes[i,j].set_ylabel("10 ^ %s" % sen_range[i])

                axes[i,j].axis('tight')

        #cbar = fig.colorbar(axes, orientation="vertical", pad=0.02, shrink=.6) #"horizontal"
        #cbar.set_label('VWC CABLE (%)')#('Volumetric soil water content (%)')
        #tick_locator = ticker.MaxNLocator(nbins=5)
        #cbar.locator = tick_locator
        #cbar.update_ticks()

        fig.savefig("EucFACE_profile_%s_multiplot_%s.png" % (plot_var, ring) , bbox_inches='tight', pad_inches=0.1)
