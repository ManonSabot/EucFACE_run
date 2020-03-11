#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-10-5"
__changefrom__ = 'plot_eucface_swc_cable_vs_obs.py'

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import datetime as dt
import netCDF4 as nc
from matplotlib import cm
from matplotlib import ticker
from plot_eucface_get_var import *

def plot_fwsoil_SM( fcables, layers, case_labels, ring):

    fig = plt.figure(figsize=[12,9])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
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

    ax = fig.add_subplot(111)
    colors = cm.tab20(np.linspace(0,1,len(case_labels)))
    #rainbow nipy_spectral Set1
    for case_num in np.arange(len(fcables)):
        SM  = read_cable_SM(fcables[case_num], layers[case_num])
        fw  = read_cable_var(fcables[case_num], "Fwsoil")
        print(SM)
        if layers[case_num] == "6":
            sm =(  SM.iloc[:,0]*0.022 + SM.iloc[:,1]*0.058 \
                 + SM.iloc[:,2]*0.154 + SM.iloc[:,3]*0.409 \
                 + SM.iloc[:,4]*(1.5-0.022-0.058-0.154-0.409) )/1.5
        elif layers[case_num] == "31uni":
            sm = SM.iloc[:,0:10].mean(axis = 1)

        ax.scatter(sm, fw,  s=3., marker='o', c=colors[case_num],label=case_labels[case_num])

    ax.set_xlim(0.1,0.45)
    ax.set_ylim(0.,1.1)
    ax.set_ylabel("β (-)")
    ax.set_xlabel("volumetric water content in top 1.5 m (m3/m3)")
    ax.legend(numpoints=1, loc='lower right')

    fig.savefig("../plots/EucFACE_fwsoil_vs_SM_%s.png" % ring , bbox_inches='tight', pad_inches=0.1)

def plot_Trans_SM(fcables, layers, case_labels, ring):

    fig = plt.figure(figsize=[12,9])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
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

    ax = fig.add_subplot(111)
    colors = cm.tab20(np.linspace(0,1,len(case_labels)))
    #rainbow
    for case_num in np.arange(len(fcables)):
        SM  = read_cable_SM(fcables[case_num], layers[case_num])
        fw  = read_cable_var(fcables[case_num], "TVeg")
        print(SM)
        if layers[case_num] == "6":
            sm =(  SM.iloc[:,0]*0.022 + SM.iloc[:,1]*0.058 \
                 + SM.iloc[:,2]*0.154 + SM.iloc[:,3]*0.409 \
                 + SM.iloc[:,4]*(1.5-0.022-0.058-0.154-0.409) )/1.5
        elif layers[case_num] == "31uni":
            sm = SM.iloc[:,0:10].mean(axis = 1)

        ax.scatter(sm, fw,  s=3., marker='o',c=colors[case_num],label=case_labels[case_num])

        if fcables[case_num] == fcables[-1] :
            Trans = read_obs_trans(ring)
            sm_trans = sm[sm.index.isin(Trans.index)]
            ax.scatter(sm_trans, Trans,  s=3., marker='o', c="red",label="Obs")

    ax.set_xlim(0.1,0.45)
    ax.set_ylim(0.,4.)
    ax.set_ylabel("Transpiration (mm d$^{-1}$)")
    ax.set_xlabel("volumetric water content in top 1.5 m (m3/m3)")
    ax.legend(numpoints=1, loc='lower right')

    fig.savefig("../plots/EucFACE_Trans_vs_SM_%s.png" % ring , bbox_inches='tight', pad_inches=0.1)

def plot_Esoil_SM(fcables, layers, case_labels, ring):

    fig = plt.figure(figsize=[12,9])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
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

    ax = fig.add_subplot(111)
    colors = cm.tab20(np.linspace(0,1,len(case_labels)))

    #rainbow Paired
    for case_num in np.arange(len(fcables)):
        SM  = read_cable_SM(fcables[case_num], layers[case_num])
        fw  = read_cable_var(fcables[case_num], "ESoil")
        print(SM)
        if layers[case_num] == "6":
            sm =(  SM.iloc[:,0]*0.022 + SM.iloc[:,1]*0.058 \
                 + SM.iloc[:,2]*0.154 + SM.iloc[:,3]*0.409 \
                 + SM.iloc[:,4]*(1.5-0.022-0.058-0.154-0.409) )/1.5
        elif layers[case_num] == "31uni":
            sm = SM.iloc[:,0:10].mean(axis = 1)

        ax.scatter(sm, fw,  s=3., marker='o', c=colors[case_num], label=case_labels[case_num]) # alpha = 0.5,
        if fcables[case_num] == fcables[-1]:
            Esoil = read_obs_esoil(ring)
            sm_esoil = sm[sm.index.isin(Esoil.index)]
            print("==========")
            print(Esoil)
            print("----------")
            print(sm_esoil)
            #print(sm_esoil)
            ax.scatter(sm_esoil, Esoil,  s=3., marker='o',  c="red",label="Obs") # alpha = 0.5,


    ax.set_xlim(0.1,0.45)
    ax.set_ylim(0.,5.)
    ax.set_ylabel("Soil evaporation (mm d$^{-1}$)")
    ax.set_xlabel("volumetric water content in top 1.5 m (m3/m3)")
    ax.legend(numpoints=1, loc='lower right')

    fig.savefig("../plots/EucFACE_Esoil_vs_SM_%s.png" % ring , bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    ring = "amb"

    case_1 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6"
    fcbl_1 ="%s/EucFACE_%s_out.nc" % (case_1, ring)

    case_2 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6_litter"
    fcbl_2 ="%s/EucFACE_%s_out.nc" % (case_2, ring)

    case_3 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_SM_6_litter"
    fcbl_3 ="%s/EucFACE_%s_out.nc" % (case_3, ring)

    case_4 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_31uni_litter"
    fcbl_4 ="%s/EucFACE_%s_out.nc" % (case_4, ring)

    case_5 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter"
    fcbl_5 ="%s/EucFACE_%s_out.nc" % (case_5, ring)

    case_6 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x1-x1_litter"
    fcbl_6 ="%s/EucFACE_%s_out.nc" % (case_6, ring)

    case_7 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter"
    fcbl_7 ="%s/EucFACE_%s_out.nc" % (case_7, ring)

    case_8 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_Hvrd"
    fcbl_8 ="%s/EucFACE_%s_out.nc" % (case_8, ring)

    case_9 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-exp"
    fcbl_9 ="%s/EucFACE_%s_out.nc" % (case_9, ring)

    case_10 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-watpot"
    fcbl_10 ="%s/EucFACE_%s_out.nc" % (case_10, ring)

    fcables     = [fcbl_1, fcbl_2, fcbl_5,   fcbl_6,    fcbl_7, fcbl_8,  fcbl_9 ]
    case_labels = ["Ctl",  "Lit",  "Hi-Res", "Opt-top", "Opt",  "β-hvrd","β-exp"]
    layers      = ["6",    "6",    "31uni",  "31uni",   "31uni","31uni", "31uni"]

    plot_fwsoil_SM(fcables, layers, case_labels, ring)
    plot_Trans_SM(fcables, layers, case_labels, ring)
    plot_Esoil_SM(fcables, layers, case_labels, ring)
