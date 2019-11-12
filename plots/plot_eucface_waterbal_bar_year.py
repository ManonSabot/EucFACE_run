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


fdfl = "EucFACE_amb_default-met_only_or-off.csv"
fctl = "EucFACE_amb_ctl_met_LAI_vrt_SM_swilt-watr_31uni_HDM_or-off-litter_Hvrd.csv"

dfl = pd.read_csv(fdfl, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])
ctl = pd.read_csv(fctl, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])

dfl['Qs'] = dfl['Qs']+dfl['Qsb']
ctl['Qs'] = ctl['Qs']+ctl['Qsb']

dfl = dfl.drop(['Year','Season','Qsb'], axis=1)
ctl = ctl.drop(['Year','Season','Qsb'], axis=1)

dfl = dfl.drop([0])
ctl = ctl.drop([0])

print(dfl)
print(ctl)

obs = [[155,153,99,34,20,0,0,-113],\
       [84,84,61,19,3,0,0,-45],\
       [250,120,75,24,21,0,0,114],\
       [151,159,106,36,16,0,0,-149],\
       [170,132,76,27,30,0,0,-26],\
       [150,80,50,13,18,0,0,25]]
       # Autum-2013
       # Winter-2013
       # Spring-2013
       # Summer-2014
       # Autum-2014
       # Winter-2014
print(np.sum(dfl.iloc[0:4].values,axis=0))
print(np.sum(obs[1:5],axis=0))

title = 'Water Balance'

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

labels = ['Rain','Evap','TVeg','ESoil','ECanop','Runoff','Rechrg','ΔS']
x = np.arange(len(labels))  # the label locations
width = 0.6                # the width of the bars

rects1 = ax.bar(x - 0.2, np.sum(obs[0:4],axis=0), width/3, color='blue', label='obs')
rects2 = ax.bar(x      , np.sum(dfl.iloc[1:5].values,axis=0), width/3, color='orange', label='def')
rects3 = ax.bar(x + 0.2, np.sum(ctl.iloc[1:5].values,axis=0), width/3, color='green', label='imp')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('$mm y^{-1}$')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#fig.tight_layout()

plt.show()

fig.savefig('water_balance_2013_obs-def-ctl', bbox_inches='tight',pad_inches=0.1)
