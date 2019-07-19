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

    """
    PAR_shaded_hod = df.PAR_shaded.groupby(df.index.hour).mean()
    PAR_sunlit_hod = df.PAR_sunlit.groupby(df.index.hour).mean()
    plt.plot(PAR_shaded_hod, label="Shaded")
    plt.plot(PAR_sunlit_hod, label="Sunlit")
    plt.legend(numpoints=1, loc="best")
    plt.ylabel("PAR ($\mathrm{\mu}$mol m$^{-2}$ s$^{-1}$)")
    plt.xlabel("Hour of day")
    plt.show()
    sys.exit()
    """
    df = df.between_time("08:00", "19:00")
    #df = df[(df.PAR_sunlit > 0.01) & (df.PAR_shaded > 0.01)]


    plt.hist(df.PAR_shaded, label="Shaded", alpha=0.7)
    plt.hist(df.PAR_sunlit, label="Sunlit", alpha=0.7)
    plt.legend(numpoints=1, loc="best")
    plt.xlabel("PAR ($\mathrm{\mu}$mol m$^{-2}$ s$^{-1}$)")
    plt.ylabel("Count")
    plt.show()
    sys.exit()

    PAR_shaded_hod = df.PAR_shaded.groupby(df.index.hour).mean()
    PAR_sunlit_hod = df.PAR_sunlit.groupby(df.index.hour).mean()

    print(PAR_shaded_hod)


    plt.plot(PAR_shaded_hod, label="Shaded")
    plt.plot(PAR_sunlit_hod, label="Sunlit")
    plt.legend(numpoints=1, loc="best")
    plt.ylabel("PAR ($\mathrm{\mu}$mol m$^{-2}$ s$^{-1}$)")
    plt.xlabel("Hour of day")
    plt.show()
    sys.exit()

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

    ax1.plot(df.index, df.PAR_sunlit, c="red", lw=1.0, ls="-")
    ax1.plot(df.index, df.PAR_shaded, c="blue", lw=1.0, ls="-")

    ax1.set_xlim(datetime.date(2002,1,1), datetime.date(2003, 1, 1))
    ax1.set_title("GPP", fontsize=12)

    plt.show()


def read_cable_file(fname):

    f = nc.Dataset(fname)
    time = nc.num2date(f.variables['time'][:],
                        f.variables['time'].units)
    df = pd.DataFrame(f.variables['PAR_shaded'][:,0], columns=['PAR_shaded'])
    df['PAR_sunlit'] = f.variables['PAR_sunlit'][:,0]
    df['PAR'] = f.variables['SWdown'][:,0] * 2.3
    df['dates'] = time
    df = df.set_index('dates')

    return df

def resample_to_seasonal_cycle(df, OBS=False):

    UMOL_TO_MOL = 1E-6
    MOL_C_TO_GRAMS_C = 12.0
    SEC_2_DAY = 86400.

    # umol/m2/s -> g/C/d
    df['PAR_shaded'] *= SEC_2_DAY
    df['PAR_sunlit'] *= SEC_2_DAY

    method = {'PAR_shaded':'mean', 'PAR_sunlit':'mean',}
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
