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


    df = df.between_time("08:00", "19:00")
    #df = df[(df.PAR_sunlit > 0.01) & (df.PAR_shaded > 0.01)]


    plt.hist(df.cica_sha, label="Shaded", alpha=0.7)
    plt.hist(df.cica_sun, label="Sunlit", alpha=0.7)
    plt.legend(numpoints=1, loc="best")

    plt.xlabel("Ci/Ca (-)")
    plt.ylabel("Count")
    plt.show()


    cica_hod_sha = df.cica_sha.groupby(df.index.hour).mean()
    cica_hod_sun = df.cica_sun.groupby(df.index.hour).mean()


    plt.plot(cica_hod_sha, label="Shaded")
    plt.plot(cica_hod_sun, label="Sunlit")
    plt.legend(numpoints=1, loc="best")
    plt.ylabel("Ci/Ca (-)")
    plt.xlabel("Hour of day")
    plt.show()


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
    #ax1.plot(df.index, df.PAR, c="green", lw=2.0, ls="-")


    ax1.plot(df.index, df.cica_sun, c="red", lw=1.0, ls="-", label="Sunlit")
    ax1.plot(df.index, df.cica_sha, c="blue", lw=1.0, ls="-", label="Shaded")
    ax1.legend(numpoints=1, loc="best")
    #ax1.set_xlim(datetime.date(2002,1,1), datetime.date(2003, 1, 1))
    ax1.set_title("Ci/Ca", fontsize=12)

    plt.show()


def read_cable_file(fname):

    f = nc.Dataset(fname)
    time = nc.num2date(f.variables['time'][:],
                        f.variables['time'].units)
    df = pd.DataFrame(f.variables['ci_over_ca_sun'][:,0], columns=['cica_sun'])
    df['cica_sha'] = f.variables['ci_over_ca_sha'][:,0]
    df['dates'] = time
    df = df.set_index('dates')

    return df

def resample_to_seasonal_cycle(df, OBS=False):

    UMOL_TO_MOL = 1E-6
    MOL_C_TO_GRAMS_C = 12.0
    SEC_2_DAY = 86400.

    method = {'cica_sun':'mean', 'cica_sha':'mean'}
    df = df.resample("M").agg(method).groupby(lambda x: x.month).mean()
    df['month'] = np.arange(1,13)

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
