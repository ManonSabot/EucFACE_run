#!/usr/bin/env python

"""
Turn the MAESPA input file into a CABLE netcdf file. Aim to swap MAESPA data
for the raw data later when I have more time...

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (04.08.2018)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import netCDF4 as nc

fig = plt.figure(figsize=[9,9])
fig.subplots_adjust(hspace=0.15)
fig.subplots_adjust(wspace=0.05)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#fm = matplotlib.font_manager.json_load(os.path.expanduser("~/.cache/matplotlib/fontlist-v310.json"))
#fm.findfont('sans-serif', rebuild_if_missing=False)

plt.rcParams['text.usetex']     = False
plt.rcParams['font.family']     = "sans-serif"
plt.rcParams['font.serif']      = "Helvetica"
plt.rcParams['axes.linewidth']  = 1.5
plt.rcParams['axes.labelsize']  = 14
plt.rcParams['font.size']       = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
ax1 = fig.add_subplot(111)

fname1 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_grass_6_sres_watr/EucFACE_amb_out.nc"
# fname2 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI-08_6/EucFACE_met_R3.nc"
# fname3 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI-08_6/EucFACE_met_R6.nc"
# fname4 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/met_LAI-08_6/EucFACE_met_amb.nc"
var_name = "TVeg"


cable1 = nc.Dataset(fname1, 'r')
# cable2 = nc.Dataset(fname2, 'r')
# cable3 = nc.Dataset(fname3, 'r')
# cable4 = nc.Dataset(fname4, 'r')
Time  = nc.num2date(cable1.variables['time'][:],cable1.variables['time'].units)

Var1 = pd.DataFrame(cable1.variables[var_name][:,0,0],columns=['var'])
Var1 = Var1*1800.
Var1['dates'] = Time
Var1 = Var1.set_index('dates')
# print(Var1.min())
# print(Var1.max())
# print(Var1.mean())
Var1 = Var1.resample("Y").agg('sum')
print(Var1)
print(Var1.mean())
# Var1.index = Var1.index - pd.datetime(2011,12,31)
# Var1.index = Var1.index.days
#
# # Var2 = pd.DataFrame(cable2.variables[var_name][:,0,0],columns=['var'])
# # Var2 = Var2#*1800.
# # Var2['dates'] = Time
# # Var2 = Var2.set_index('dates')
# # Var2 = Var2.resample("D").agg('mean')
# # Var2.index = Var2.index - pd.datetime(2011,12,31)
# # Var2.index = Var2.index.days
# #
# # Var3 = pd.DataFrame(cable3.variables[var_name][:,0,0],columns=['var'])
# # Var3 = Var3#*1800.
# # Var3['dates'] = Time
# # Var3 = Var3.set_index('dates')
# # Var3 = Var3.resample("D").agg('mean')
# # Var3.index = Var3.index - pd.datetime(2011,12,31)
# # Var3.index = Var3.index.days
# #
# # Var4 = pd.DataFrame(cable4.variables[var_name][:,0,0],columns=['var'])
# # Var4 = Var4#*1800.
# # Var4['dates'] = Time
# # Var4 = Var4.set_index('dates')
# # Var4 = Var4.resample("D").agg('mean')
# # Var4.index = Var4.index - pd.datetime(2011,12,31)
# # Var4.index = Var4.index.days
#
# # .rolling(window=30).mean()
#
# ax1.plot(Var1["var"], c="green", lw=1.0, ls="-", label="R2") #.rolling(window=7).mean()
# # ax1.plot(Var2["var"], c="orange",lw=1.0, ls="-", label="R3") #.rolling(window=7).mean()
# # ax1.plot(Var3["var"], c="blue", lw=1.0, ls="-", label="R6") #.rolling(window=7).mean()
# # ax1.plot(Var4["var"], c="red",lw=1.0, ls="-", label="amb") #.rolling(window=7).mean()
# ax1.legend()

plt.savefig("check_grass_%s" %var_name, bbox_inches='tight', pad_inches=0.1)
