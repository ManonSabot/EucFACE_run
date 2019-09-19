#!/usr/bin/env python

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

fcable = "/short/w35/mm3972/cable/src/CABLE-AUX/offline/gridinfo_mmy_MD_elev_orig_std_avg-sand_mask.nc"

cable = nc.Dataset(fcable, 'r')
var = cable.variables['bch'][:,:]
print(np.unique(var).shape)

