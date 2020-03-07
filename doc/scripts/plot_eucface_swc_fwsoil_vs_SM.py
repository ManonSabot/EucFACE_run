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
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from plot_eucface_get_var import *

def plot_fwsoil_SM( fstd, fhvrd, fexp, fwatpot, layer, ring):

    SM_std  = read_cable_SM(fstd, layer)
    fw_std  = read_cable_var(fstd, "Fwsoil")
    print(SM_std)
    print(fw_std)
    sm_std = SM_std.iloc[:,0:10].mean(axis = 1)

    SM_hvrd  = read_cable_SM(fhvrd, layer)
    fw_hvrd  = read_cable_var(fhvrd, "Fwsoil")
    sm_hvrd = SM_hvrd.iloc[:,0:10].mean(axis = 1)

    SM_exp  = read_cable_SM(fexp, layer)
    fw_exp  = read_cable_var(fexp, "Fwsoil")
    sm_exp = SM_exp.iloc[:,0:10].mean(axis = 1)

    SM_watpot  = read_cable_SM(fwatpot, layer)
    fw_watpot  = read_cable_var(fwatpot, "Fwsoil")
    sm_watpot = SM_watpot.iloc[:,0:10].mean(axis = 1)

    print(sm_watpot)

# _____________ Plot _____________
    fig = plt.figure(figsize=[8,6])
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

    ax = fig.add_subplot(111)

    ax.scatter(sm_std, fw_std,  s=1, marker='o', c='orange',label="β-std")
    ax.scatter(sm_hvrd,fw_hvrd, s=1, marker='o', c='blue',label="β-hvrd")
    ax.scatter(sm_exp, fw_exp,  s=1, marker='o', c='green',label="β-exp")
    ax.scatter(sm_watpot,fw_watpot, s=1, marker='o', c='red',label="β-watpot")

    ax.set_xlim(0.,0.35)
    ax.set_ylim(0.,1.1)
    ax.set_ylabel("β")
    ax.set_xlabel("volumetric water content in top 1.5 m (m3/m3)")

    ax.legend(numpoints=1, loc='lower right')

    fig.savefig("EucFACE_fwsoil_vs_SM_%s.png" % ring , bbox_inches='tight', pad_inches=0.1)



if __name__ == "__main__":

    ring = "amb"
    layer= "31uni"
    case_5 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter"
    fstd ="%s/EucFACE_%s_out.nc" % (case_5, ring)

    case_6 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_Hvrd"
    fhvrd ="%s/EucFACE_%s_out.nc" % (case_6, ring)

    case_7 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_hie-exp"
    fexp ="%s/EucFACE_%s_out.nc" % (case_7, ring)

    case_8 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_hie-watpot"
    fwatpot ="%s/EucFACE_%s_out.nc" % (case_8, ring)

    plot_fwsoil_SM( fstd, fhvrd, fexp, fwatpot, layer, ring)
