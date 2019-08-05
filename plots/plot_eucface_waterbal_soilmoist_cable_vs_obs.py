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

file_path = "/g/data1a/w35/mm3972/cable/EucFACE/EucFACE_run/outputs/calculated_para_gridinfo_ununi_gw_on_or_off/after_changing_cable_input_as_abs_sucs_vec_1000/depth_varied_para_gw_on_or_off/"
famb = "EucFACE_amb_out.nc"
fele = "EucFACE_ele_out.nc"

amb = nc.Dataset(os.path.join(file_path,famb), 'r')
ele = nc.Dataset(os.path.join(file_path,fele), 'r')

zse                    = [0.02, 0.05,0.06,0.13,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.75,1.50]
df_SM_amb              = pd.DataFrame(amb.variables['SoilMoist'][:,0,0], columns=['SoilMoist'])
df_SM_ele              = pd.DataFrame(ele.variables['SoilMoist'][:,0,0], columns=['SoilMoist'])
df_SM_amb['SoilMoist'] = 0.0
df_SM_ele['SoilMoist'] = 0.0

for i in np.arange(0,13,1):
    df_SM_amb = df_SM_amb + amb.variables['SoilMoist'][:,i,0]*zse[i]*1000.
    df_SM_ele = df_SM_ele + ele.variables['SoilMoist'][:,i,0]*zse[i]*1000.

df_SM_amb['dates']    = nc.num2date(amb.variables['time'][:],amb.variables['time'].units)
df_SM_amb             = df_SM_amb.set_index('dates')
df_SM_index_amb       = df_SM_amb.index.strftime('%Y-%m-%d %H:%M')

df_SM_ele['dates']    = nc.num2date(ele.variables['time'][:],ele.variables['time'].units)
df_SM_ele             = df_SM_ele.set_index('dates')
df_SM_index_ele       = df_SM_ele.index.strftime('%Y-%m-%d %H:%M')

# monthly soil water content and monthly changes
df_SM_mth_laststep_amb         = df_SM_amb.resample("M").agg('mean')
#df_SM_mth_laststep_amb.iloc[0] = df_SM_amb.iloc[0]
j = 0
for i in np.arange(0,len(df_SM_amb),1):
    if df_SM_amb.index.is_month_end[i] and df_SM_index_amb[i][11:16] == '23:30':
        print(df_SM_amb.index[i])
        print(df_SM_index_amb[i])
        df_SM_mth_laststep_amb.iloc[j] = df_SM_amb.iloc[i]
        j       += 1
#df_SM_mth_chg_amb = pd.DataFrame((df_SM_mth_laststep_amb.iloc[1:].values - df_SM_mth_laststep_amb.iloc[0:-1].values), \
#                          columns=['SoilMoist'])#,'SoilMoist_up','SoilMoist_lw'])
#df_SM_mth_chg_amb['dates'] = df_SM_mth_laststep_amb.index[0:-1]
#df_SM_mth_chg_amb          = df_SM_mth_chg_amb.set_index('dates')

# monthly soil water content and monthly changes
df_SM_mth_laststep_ele         = df_SM_ele.resample("M").agg('mean')
#df_SM_mth_laststep_ele.iloc[0] = df_SM_ele.iloc[0]
j = 0
for i in np.arange(0,len(df_SM_ele),1):
    if df_SM_ele.index.is_month_end[i] and df_SM_index_ele[i][11:16] == '23:30':
        df_SM_mth_laststep_ele.iloc[j] = df_SM_ele.iloc[i]
        j       += 1
#df_SM_mth_chg_ele = pd.DataFrame((df_SM_mth_laststep_ele.iloc[1:].values - df_SM_mth_laststep_ele.iloc[0:-1].values), \
#                          columns=['SoilMoist'])#,'SoilMoist_up','SoilMoist_lw'])
#df_SM_mth_chg_ele['dates'] = df_SM_mth_laststep_ele.index[0:-1]
#df_SM_mth_chg_ele          = df_SM_mth_chg_ele.set_index('dates')


df_SM_mth_laststep_amb.to_csv("EucFACE_amb_SM_gw_on_or_off.csv")
df_SM_mth_laststep_ele.to_csv("EucFACE_ele_SM_gw_on_or_off.csv")
