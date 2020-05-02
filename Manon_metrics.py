#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code that performs the selection of the best parameterisation and/or
configuration for a given variable.

This file is part of the TractLSM project.

Copyright (c) 2019 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

The  logic is based on PLUMBER:
    * Best, M. J., Abramowitz, G., Johnson, H. R., Pitman, A. J.,
      Balsamo, G., Boone, A., ... & Ek, M. (2015). The plumbing of land
      surface models: benchmarking model performance. Journal of
      Hydrometeorology, 16(3), 1425-1442.

"""

__title__ = "best performance for any given parameter/configuration"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (08.10.2019)"
__email__ = "m.e.b.sabot@gmail.com"
 

#=======================================================================

# import general modules
import os  # check for files, paths
import sys  # check for files, paths
import netCDF4 as nc  # open netcdf
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
from scipy import stats  # compute statistical metrics and p-ranks
import itertools


#=======================================================================

def main(idata, odata, variable):

    """
    Main function: Generates 'perf_scores.csv' in the base_dir, which
                   contains information on the performance of different
                   configurations of the model, in the form of
                   statistical metrics and ranks. The logic is based on
                   PLUMBER (Best et al., 2015).

    Arguments:
    ----------
    idata: array
        absolute paths to the input data

    odata: array
        absolute paths to the output data

    variable: string
        variable that is being calibrated for

    Returns:
    --------
    'best.txt' in the output project directory.

    """

    # compare the fluxes to the data, calibrate parameter
    best = performance_scores(idata, odata, variable)

    return


#=======================================================================

# ~~~ Other functions are defined here ~~~

def read_netcdf(idata, variable):

    cable = nc.Dataset(idata, 'r')
    #Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    data = pd.DataFrame(cable.variables[variable][0:17000,0,0], columns=[variable])

    return data


def find_root(arr):

    """
    Finds substring common to a whole array of strings.

    Arguments:
    ----------
    arr: array
        list of strings amongst which we want to find common elem.

    Returns:
    --------
    res: string
        the common substring for the whole array of strings

    """

    # determine size of the array
    n = len(arr)

    # take first word from array as reference
    s = arr[0]
    l = len(s)

    res = ''

    for i in range(l):

        for j in range(i + 1, l + 1):

            # all possible substrings of our reference string
            stem = s[i:j]
            k = 1

            for k in range(1, n):

                # check if the generated stem is common to all strings
                if stem not in arr[k]:
                    break

            # current substring present in all strings?
            if (k + 1 == n) and (len(res) < len(stem)):
                res = stem

    return res


def performance_scores(idata, odata, variables):

    """
    Computes statistical measures for different configurations of the
    model (logic based on PLUMBER; Best et al., 2015).

    Arguments:
    ----------
    idata: array
        absolute paths to the input data

    odata: array
        absolute paths to the output data

    variable: string
        variable that is being calibrated for

    Returns:
    --------
    best: array
        string names of all the best parameter calibrations

    """

    # performance metrics
    index = pd.MultiIndex.from_tuples(list(itertools.product(variables, odata)))
    df = pd.DataFrame(index=index,
                      columns=['NMSE', 'MAE', 'SD', 'P5', 'P95'])

    # ranks
    df2 = pd.DataFrame(index=index,
                       columns=['rNMSE', 'rMAE', 'rSD', 'rP5', 'rP95'])

    for variable in variables:

        obs = read_netcdf(idata, variable)
        obs.fillna(0., inplace=True)

        # get all the sims
        for i in range(len(odata)):

            sim = read_netcdf(odata[i], variable)
            sim.fillna(0., inplace=True)

            # deal with missing or negative values
            mask = np.logical_and(obs < -999., sim < -999.)
            obs = obs[~mask]
            sim = sim[~mask]
            obs.fillna(0., inplace=True)
            sim.fillna(0., inplace=True)

            # metrics
            nmse = np.mean((sim - obs) ** 2. / (np.mean(sim) * np.mean(obs)))
            mae = np.mean(np.abs(sim - obs))
            sd = np.abs(1. - np.std(sim) / np.std(obs))
            #p5 = np.abs(np.percentile(sim, 5) - np.percentile(obs, 5))
            #p95 = np.abs(np.percentile(sim, 95) - np.percentile(obs, 95))

            df.loc[(variable, odata[i]), 'NMSE'] = nmse.values[0]
            df.loc[(variable, odata[i]), 'MAE'] = mae.values[0]
            df.loc[(variable, odata[i]), 'SD'] = sd.values[0]
            #df.loc[(variable, odata[i]), 'P5'] = p5
            #df.loc[(variable, odata[i]), 'P95'] = p95

    print(df)

    for i in range(len(df.columns)):

        x = np.abs(df.loc[variable, df.columns[i]].values.flatten())
        print(x)
        y = np.array([stats.percentileofscore(x, a, 'weak') if not
                      pd.isna(a) else -1. for a in x])
        print(y)

    exit(1)

    return df, df2


#=======================================================================

if __name__ == "__main__":

    idata = 'test_in.nc'  # proxy obs file
    odata = ['EucFACE_amb_out.nc', 'test_in.nc']  # evaluating this
    variables = ['TVeg', 'ESoil']

    main(idata, odata, variables)
