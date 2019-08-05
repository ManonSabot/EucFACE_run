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

file_path = "/g/data/w35/mm3972/cable/EucFACE/EucFACE_run/met/"
famb = "EucFACE_met_amb.nc"

amb = nc.Dataset(os.path.join(file_path,famb), 'r')

step_2_sec = 30.*60.

df_amb              = pd.DataFrame(amb.variables['Rainf'][:,0], columns=['Rainf'])
df_amb['dates']     = nc.num2date(amb.variables['time'][:],amb.variables['time'].units)
df_amb              = df_amb.set_index('dates')

df_amb              = df_amb*step_2_sec
df_amb              = df_amb.resample("A").agg('sum')
#df_amb              = df_amb.drop(df_amb.index[len(df_amb)-1])
df_amb.index        = df_amb.index.strftime('%Y')

print(df_amb)
fig, ax = plt.subplots()

ax.scatter(df_amb.index, df_amb.values)

for i, txt in enumerate(df_amb.values):
    ax.annotate(txt, (df_amb.index[i], df_amb.values[i]))

#plt.plot(df_amb.index, df_amb.values , 'ro')
ax.axis([-1, 7, 0, 1200])
fig.savefig("check_rainfall.png")
