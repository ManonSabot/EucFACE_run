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


def main(fobs, fcable, case_name):

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)

    ESoil = pd.DataFrame(cable.variables['ESoil'][:,0,0],columns=['ESoil'])
    ESoil = ESoil*1800.
    ESoil['dates'] = Time
    ESoil = ESoil.set_index('dates')
    ESoil = ESoil.resample("D").agg('sum')
    ESoil.index = ESoil.index - pd.datetime(2011,12,31)
    ESoil.index = ESoil.index.days

    Tveg = pd.DataFrame(cable.variables['TVeg'][:,0,0],columns=['Tveg'])
    Tveg = Tveg*1800.
    Tveg['dates'] = Time
    Tveg = Tveg.set_index('dates')
    Tveg = Tveg.resample("D").agg('sum')
    Tveg.index = Tveg.index - pd.datetime(2011,12,31)
    Tveg.index = Tveg.index.days

    Rnet = pd.DataFrame(cable.variables['Rnet'][:,0,0],columns=['Rnet'])
    Rnet['dates'] = Time
    Rnet = Rnet.set_index('dates')
    Rnet = Rnet.resample("D").agg('mean')
    Rnet.index = Rnet.index - pd.datetime(2011,12,31)
    Rnet.index = Rnet.index.days

    Qle = pd.DataFrame(cable.variables['Qle'][:,0,0],columns=['Qle'])
    Qle['dates'] = Time
    Qle = Qle.set_index('dates')
    Qle = Qle.resample("D").agg('mean')
    Qle.index = Qle.index - pd.datetime(2011,12,31)
    Qle.index = Qle.index.days

    Qh = pd.DataFrame(cable.variables['Qh'][:,0,0],columns=['Qh'])
    Qh['dates'] = Time
    Qh = Qh.set_index('dates')
    Qh = Qh.resample("D").agg('mean')
    Qh.index = Qh.index - pd.datetime(2011,12,31)
    Qh.index = Qh.index.days

    Qg = pd.DataFrame(cable.variables['Qg'][:,0,0],columns=['Qg'])
    Qg['dates'] = Time
    Qg = Qg.set_index('dates')
    Qg = Qg.resample("D").agg('mean')
    Qg.index = Qg.index - pd.datetime(2011,12,31)
    Qg.index = Qg.index.days


# _________________________ Obs ___________________________
    obs = nc.Dataset(fobs, 'r')
    Time  = nc.num2date(obs.variables['time'][:],obs.variables['time'].units)

    Rnet_obs = pd.DataFrame(obs.variables['Rnet'][:,0,0],columns=['Rnet'])
    Rnet_obs['dates'] = Time
    Rnet_obs = Rnet_obs.set_index('dates')
    Rnet_obs = Rnet_obs.resample("D").agg('mean')
    Rnet_obs.index = Rnet_obs.index - pd.datetime(2011,12,31)
    Rnet_obs.index = Rnet_obs.index.days

    Qle_obs = pd.DataFrame(obs.variables['Qle'][:,0,0],columns=['Qle'])
    Qle_obs['dates'] = Time
    Qle_obs = Qle_obs.set_index('dates')
    Qle_obs = Qle_obs.resample("D").agg('mean')
    Qle_obs.index = Qle_obs.index - pd.datetime(2011,12,31)
    Qle_obs.index = Qle_obs.index.days

    Qh_obs = pd.DataFrame(obs.variables['Qh'][:,0,0],columns=['Qh'])
    Qh_obs['dates'] = Time
    Qh_obs = Qh_obs.set_index('dates')
    Qh_obs = Qh_obs.resample("D").agg('mean')
    Qh_obs.index = Qh_obs.index - pd.datetime(2011,12,31)
    Qh_obs.index = Qh_obs.index.days

    Qg_obs = pd.DataFrame(obs.variables['Qg'][:,0,0],columns=['Qg'])
    Qg_obs['dates'] = Time
    Qg_obs = Qg_obs.set_index('dates')
    Qg_obs = Qg_obs.resample("D").agg('mean')
    Qg_obs.index = Qg_obs.index - pd.datetime(2011,12,31)
    Qg_obs.index = Qg_obs.index.days

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
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    width = 1.
    x = Qle.index
    y = Rnet_obs.index
    ax1.plot(x, Tveg['Tveg'].rolling(window=5).mean(),  c="green", lw=1.0, ls="-", label="Trans")
    ax1.plot(x, ESoil['ESoil'].rolling(window=5).mean(), c="orange", lw=1.0, ls="-", label="Esoil")
    #ax2.plot(x, Rnet["Rnet"],     c="red", lw=1.0, ls="-", label="Rnet")
    #ax2.plot(y, Rnet_obs["Rnet"], c="red", lw=1.0, ls="-.", label="Rnet Obs")
    ax2.plot(x, Qh["Qh"],       c="orange", lw=1.0, ls="-", label="Qh")
    ax2.plot(y, Qh_obs["Qh"],   c="red", lw=1.0, ls="-.", label="Qh Obs")
    ax3.plot(x, Qle["Qle"],      c="green", lw=1.0, ls="-", label="Qle")
    ax3.plot(y, Qle_obs["Qle"],  c="blue", lw=1.0, ls="-.", label="Qle Obs")
    #ax2.plot(x, Qg["Qg"],       c="blue", lw=1.0, ls="-", label="Qg")
    #ax2.plot(y, Qg_obs["Qg"],   c="blue", lw=1.0, ls="-.", label="Qg Obs")

    cleaner_dates = ["2013","2014","2015"]
    xtickslocs    = [367,732,1097]

    plt.setp(ax1.get_xticklabels(), visible=True)
    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_ylabel("Trans and Esoil")
    ax1.axis('tight')
    ax1.set_ylim(0,5.)
    #ax1.set_xlim(367,2739)
    ax1.legend()

    plt.setp(ax2.get_xticklabels(), visible=True)
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax2.set_ylabel("Qh")
    ax2.axis('tight')
    ax2.set_ylim(-20.,200.)
    #ax3.set_xlim(367,2739)
    ax2.legend()

    plt.setp(ax3.get_xticklabels(), visible=True)
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
    ax3.set_ylabel("Qle")
    ax3.axis('tight')
    ax3.set_ylim(0.,250.)
    #ax3.set_xlim(367,2739)
    ax3.legend()

    fig.savefig("Cumberland_ET_Qle_Qh_%s.png" %case_name , bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":


    cases = ["Cumberland_default", "Cumberland_litter"]

    for case_name in cases:
        fobs   = "/srv/ccrc/data25/z5218916/data/Cumberland_OzFlux/CumberlandPlainsOzFlux2.0_flux.nc"
        fcable = "/srv/ccrc/data25/z5218916/cable/EucFACE/test_Cumberland/outputs/%s/CumberlandPlainsOzFlux2_out.nc" \
                % (case_name)
        main(fobs, fcable, case_name)
