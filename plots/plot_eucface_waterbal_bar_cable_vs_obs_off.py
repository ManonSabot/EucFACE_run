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

famb = "EucFACE_amb_gw_on_or_off.csv"
fele = "EucFACE_ele_gw_on_or_off.csv"

amb = pd.read_csv(famb, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])
ele = pd.read_csv(fele, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])

amb['Qs'] = amb['Qs']+amb['Qsb']
ele['Qs'] = ele['Qs']+ele['Qsb']

amb = amb.drop(['Year','Season','Qsb'], axis=1)
ele = ele.drop(['Year','Season','Qsb'], axis=1)

amb = amb.drop([0])
ele = ele.drop([0])

print(amb)
print(ele)

amb_obs = [[155,153,99,34,20,0,0,-113],\
            [84,84,61,19,3,0,0,-45],\
            [250,120,75,24,21,0,0,114],\
            [151,159,106,36,16,0,0,-149],\
            [170,132,76,27,30,0,0,-26],\
            [150,80,50,13,18,0,0,25]]
ele_obs = [[155,151,100,28,24,0,0,-89],\
            [84,81,57,19,5,0,0,-48],\
            [250,126,69,24,32,0,0,110],\
            [151,172,111,38,23,0,0,-139],\
            [170,142,89,15,28,0,0,-34],\
            [150,87,54,13,20,0,0,14]]
print(amb_obs)
print(ele_obs)

i = 5
print(amb.iloc[i].values)
print(amb_obs[i])

title = 'Water balance of Winter-2014'

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
width = 0.8                # the width of the bars

rects1 = ax.bar(x - 0.3, amb.iloc[i].values, width/4, color='royalblue', label='amb_cable')
rects2 = ax.bar(x - 0.1, ele.iloc[i].values, width/4, color='orangered', label='ele_cable')
rects3 = ax.bar(x + 0.1, amb_obs[i], width/4, color='deepskyblue', label='amb_obs')
rects4 = ax.bar(x + 0.3, ele_obs[i], width/4, color='lightsalmon',label='ele_obs')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('mm')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#fig.tight_layout()

plt.show()

fig.savefig('water_balance_Winter-2014_or_off', bbox_inches='tight',pad_inches=0.1)
