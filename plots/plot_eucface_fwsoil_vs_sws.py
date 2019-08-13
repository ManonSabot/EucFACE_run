#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc

file_path = "/g/data1a/w35/mm3972/cable/EucFACE/EucFACE_run/outputs/calculated_para_gridinfo_ununi_gw_on/after_changing_cable_input_as_abs_sucs_vec_1000/depth_varied_para_gw_on_or_off/"
famb = "EucFACE_amb_out.nc"
fele = "EucFACE_ele_out.nc"

amb = nc.Dataset(os.path.join(file_path,famb), 'r')
ele = nc.Dataset(os.path.join(file_path,fele), 'r')

zse                 = [0.02, 0.05,0.06,0.13,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.75,1.50]

df_amb              = pd.DataFrame(amb.variables['SoilMoist'][:,0,0], columns=['SoilMoist'])
df_ele              = pd.DataFrame(ele.variables['SoilMoist'][:,0,0], columns=['SoilMoist'])
df_amb['SoilMoist'] = 0.0
df_ele['SoilMoist'] = 0.0
'''
SM_up_amb = np.zeros(len(df_amb))
SM_up_ele = np.zeros(len(df_amb))
SM_bottom_amb = np.zeros(len(df_amb))
SM_bottom_ele = np.zeros(len(df_amb))
print(amb.variables['SoilMoist'][:,1,0]*zse[1]*1000.)
'''
for i in np.arange(0,13,1):
    df_amb = df_amb + amb.variables['SoilMoist'][:,i,0]*zse[i]*1000.
    df_ele = df_ele + ele.variables['SoilMoist'][:,i,0]*zse[i]*1000.
'''
for i in np.arange(0,6,1):
    SM_up_amb = SM_up_amb + amb.variables['SoilMoist'][:,i,0]*zse[i]*1000.
    SM_up_ele = SM_up_ele + ele.variables['SoilMoist'][:,i,0]*zse[i]*1000.

for i in np.arange(6,13,1):
    SM_bottom_amb = SM_bottom_amb + amb.variables['SoilMoist'][:,i,0]*zse[i]*1000.
    SM_bottom_ele = SM_bottom_ele + ele.variables['SoilMoist'][:,i,0]*zse[i]*1000.


df_amb['SM_up'] = SM_up_amb
df_ele['SM_up'] = SM_up_ele
df_amb['SM_bottom'] = SM_bottom_amb
df_ele['SM_bottom'] = SM_bottom_ele
'''
df_amb['Fwsoil'] = amb.variables['Fwsoil'][:,0]
df_ele['Fwsoil'] = ele.variables['Fwsoil'][:,0]
df_amb['dates']  = nc.num2date(amb.variables['time'][:],amb.variables['time'].units)
df_amb           = df_amb.set_index('dates')
df_ele['dates']  = nc.num2date(ele.variables['time'][:],ele.variables['time'].units)
df_ele           = df_ele.set_index('dates')

df_amb           = df_amb.resample("M").agg('mean')
#df_amb           = df_amb.drop(df_amb.index[len(df_amb)-1])
#df_amb.index     = df_amb.index.strftime('%Y-%m-%d')
#turn DatetimeIndex into the formatted strings specified by date_format

df_ele           = df_ele.resample("M").agg('mean')
#df_ele           = df_ele.drop(df_ele.index[len(df_ele)-1])
#df_ele.index     = df_ele.index.strftime('%Y-%m-%d')
#turn DatetimeIndex into the formatted strings specified by date_format

var = df_amb

fig = plt.figure(figsize=(6,9))
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

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(var.index, var['Fwsoil'], c="blue", lw=2.0, ls="-")
ax2.plot(var.index, var['SoilMoist'], c="blue", lw=2.0, ls="-")
#ax2.plot(var.index, var['SoilMoist','SM_up','SM_bottom'], c=["blue",'turquoise','darkcyan'], lw=[2.0,2.0,2.0], ls=["-","-","-"])
ax1.set_title('Fwsoil', fontsize=12)
ax2.set_title('Water Storage (mm)', fontsize=12)
#ax2.legend()
plt.show()

fig.savefig("fwsoil_sws_amb_or_off.png", bbox_inches='tight', pad_inches=0.1)
