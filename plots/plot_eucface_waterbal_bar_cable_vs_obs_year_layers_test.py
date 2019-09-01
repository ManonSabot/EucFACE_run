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

# _____________________ choose cases __________________________
#case = ["hyds0.01", "hyds0.1", "hyds","hyds10","hyds100"]
case = "hyds0.01"

# _____________________ observation ___________________________
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


fcase1 = "EucFACE_amb_6layers_%s_gw_on_or_on.csv" % (case)
fcase2 = "EucFACE_amb_13layers_%s_gw_on_or_on.csv" % (case)
fcase3 = "EucFACE_amb_31layers_%s_gw_on_or_on.csv" % (case)

# _____________________ Make plot ___________________________
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

title = 'Water balance of 2013'

ax = fig.add_subplot(111)

labels = ['Rain','Evap','TVeg','ESoil','ECanop','Runoff','Rechrg','ΔS']
x = np.arange(len(labels))  # the label locations

width = 0.8                # the width of the bars

f1 = pd.read_csv(fcase1, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])
f2 = pd.read_csv(fcase2, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])
f3 = pd.read_csv(fcase3, usecols = ['Year','Season','Rainf','Evap','TVeg','ESoil','ECanop','Qs','Qsb','Qrecharge','soil_storage_chg'])

f1['Qs'] = f1['Qs']+f1['Qsb']
f2['Qs'] = f2['Qs']+f2['Qsb']
f3['Qs'] = f3['Qs']+f3['Qsb']

f1 = f1.drop(['Year','Season','Qsb'], axis=1)
f2 = f2.drop(['Year','Season','Qsb'], axis=1)
f3 = f3.drop(['Year','Season','Qsb'], axis=1)

f1 = f1.drop([0])
f2 = f2.drop([0])
f3 = f3.drop([0])

rects1 = ax.bar(x - 0.3, np.sum(amb_obs[0:4],axis=0), width/4, color='blueviolet', label='Obs')
rects2 = ax.bar(x - 0.1, np.sum(f1.iloc[0:4].values,axis=0), width/4, color='orangered', label='6-l')
rects3 = ax.bar(x + 0.1, np.sum(f2.iloc[0:4].values,axis=0), width/4, color='orange', label='13-l')
rects4 = ax.bar(x + 0.3, np.sum(f3.iloc[0:4].values,axis=0), width/4, color='darkturquoise', label='31-l')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('mm / year')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#fig.tight_layout()

plt.show()

fig.savefig('water_balance_2013_layers-test_%s.png' % (case), bbox_inches='tight',pad_inches=0.1)
