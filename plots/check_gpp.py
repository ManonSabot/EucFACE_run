#!/usr/bin/env python

"""
Plot visual benchmark (average seasonal cycle) of old vs new model runs.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (18.10.2017)"
__email__ = "mdekauwe@gmail.com"

import netCDF4 as nc
import matplotlib.pyplot as plt
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import os
import datetime
def main(fname, plot_fname=None):

    df = read_cable_file(fname)

    fig = plt.figure(figsize=(9,6))
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

    ax1 = fig.add_subplot(111)
    #ax1.plot(df.index, df.GPP, c="black", lw=2.0, ls="-")
    ax1.plot(df.index, df.PAR, c="green", lw=2.0, ls="-")

    #ax1.plot(df.index, df.PAR_sunlit, c="red", lw=1.0, ls="-")
    #ax1.plot(df.index, df.PAR_shaded, c="blue", lw=1.0, ls="-")
    ax1.plot(df.index,  df.PAR_sunlit+df.PAR_shaded, c="red", lw=1.0, ls="-")


    ax1.set_xlim(datetime.date(2013,1,1), datetime.date(2014, 1, 1))
    ax1.set_title("GPP", fontsize=12)

    plt.show()


def read_cable_file(fname):

    f = nc.Dataset(fname)
    #print(f.variables['iveg'][0,0])
    #sys.exit()
    time = nc.num2date(f.variables['time'][:],
                        f.variables['time'].units)
    df = pd.DataFrame(f.variables['GPP'][:,0], columns=['GPP'])
    df['GPP_shaded'] = f.variables['GPP_shaded'][:,0]
    df['GPP_sunlit'] = f.variables['GPP_sunlit'][:,0]
    df['PAR_shaded'] = f.variables['PAR_shaded'][:,0]
    df['PAR_sunlit'] = f.variables['PAR_sunlit'][:,0]
    df['PAR'] = f.variables['SWdown'][:,0] #* 2.3
    print(df.PAR)
    df['dates'] = time
    df = df.set_index('dates')

    return df



if __name__ == "__main__":

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--fname", dest="fname",
                      action="store", help="filename",
                      type="string")
    parser.add_option("-p", "--plot_fname", dest="plot_fname", action="store",
                      help="Benchmark plot filename", type="string")
    (options, args) = parser.parse_args()

    main(options.fname, options.plot_fname)
