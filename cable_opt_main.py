#!/usr/bin/env python

__author__    = "MU Mengyuan"

import os
import sys
import glob
import shutil
import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime
import scipy.optimize as optimize
from cable_run_sen_opt import RunCable
from run_sen_opt import *

def residuals(sen_value, met_case, met_dir, met_subset, sen_para, operator, obs):

    output_file = main(sen_para, sen_value, operator, met_dir, met_subset)

    cable_swc   = get_cable_value(output_file, met_case)

    return (obs - cable_swc)

def get_cable_value(output_file, met_case):

    cable = nc.Dataset(output_file, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,0,0,0], columns=['SoilMoist'])

    layer = os.path.basename(met_case).split("_")[-1]

    if layer == "6":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.5-0.022-0.058-0.154) )/0.5
    elif layer == "13":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.02 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.05 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.06 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.13 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.5-0.02-0.05-0.06-0.13) )/0.5
    elif layer == "31uni":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.05 )/0.5
    elif layer == "31exp":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.020440 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.001759 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.003957 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.007035 \
                                 + cable.variables['SoilMoist'][:,4,0,0]*0.010993 \
                                 + cable.variables['SoilMoist'][:,5,0,0]*0.015829 \
                                 + cable.variables['SoilMoist'][:,6,0,0]*0.021546 \
                                 + cable.variables['SoilMoist'][:,7,0,0]*0.028141 \
                                 + cable.variables['SoilMoist'][:,8,0,0]*0.035616 \
                                 + cable.variables['SoilMoist'][:,9,0,0]*0.043971 \
                                 + cable.variables['SoilMoist'][:,10,0,0]*0.053205 \
                                 + cable.variables['SoilMoist'][:,11,0,0]*0.063318 \
                                 + cable.variables['SoilMoist'][:,12,0,0]*0.074311 \
                                 + cable.variables['SoilMoist'][:,13,0,0]*0.086183 \
                                 + cable.variables['SoilMoist'][:,14,0,0]*(0.5-0.466304))/0.5
    elif layer == "31para":
        SoilMoist['SoilMoist'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.020440 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.001759 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.003957 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.007035 \
                                 + cable.variables['SoilMoist'][:,4,0,0]*0.010993 \
                                 + cable.variables['SoilMoist'][:,5,0,0]*0.015829 \
                                 + cable.variables['SoilMoist'][:,6,0,0]*(0.5-0.420714))/0.5

    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2011,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.sort_values(by=['dates'])

    return SoilMoist.values

def read_obs_swc(ring):

    fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    tdr = pd.read_csv(fobs, usecols = ['Ring','Date','swc.tdr'])
    tdr['Date'] = pd.to_datetime(tdr['Date'],format="%Y-%m-%d",infer_datetime_format=False)
    tdr['Date'] = tdr['Date'] - pd.datetime(2011,12,31)
    tdr['Date'] = tdr['Date'].dt.days
    tdr = tdr.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
        subset = tdr[(tdr['Ring'].isin(['R2','R3','R6'])) & (tdr.Date > 366)]
    elif ring == 'ele':
        subset = tdr[(tdr['Ring'].isin(['R1','R4','R5'])) & (tdr.Date > 366)]
    else:
        subset = tdr[(tdr['Ring'].isin([ring]))  & (tdr.Date > 366)]

    subset = subset.groupby(by=["Date"]).mean()/100.
    subset = subset.xs('swc.tdr', axis=1, drop_level=True)

    return subset.values


if __name__ == "__main__":


    sen_para   = "hyds"
    operator   = "="
    aa = np.array([0.09])
    sen_value  = aa[0] # initial_guess

    ring       = "amb"
    met_case   = "met_LAI_vrt_swilt-watr-ssat_SM_31uni"
    met_dir    = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_opt_fw-hie-exp_31uni/met/%s" % met_case
    met_subset = "EucFACE_met_%s.nc" % ring

    obs = read_obs_swc(ring)
    print(obs.dtype)
    print(sen_value.dtype)
    
    (popt, pcov,info, mesg, success) = optimize.leastsq(residuals, sen_value, \
                                       args=( met_case, met_dir, met_subset,  \
                                       sen_para, operator, obs),full_output=True)

    if success != 1:
        print("No fit!")
        print(mesg)
        sys.exit()
    '''
    # calculate final chi square
    chisq = sum(info["fvec"] * info["fvec"])
    dof = len(obs) - len(p0)
    print( "Converged with chi squared ", chisq )
    print( "degrees of freedom, dof ", dof)
    print "RMS of residuals (i.e. sqrt(chisq/dof)) ", np.sqrt(chisq / dof)
    print "Reduced chisq (i.e. variance of residuals) ", chisq / dof
    '''
    # uncertainties are calculated as per gnuplot, "fixing" the result
    # for non unit values of the reduced chisq.
    # values at min match gnuplot
    #print "Fitted parameters at minimum, with 68% C.I.:"
    #for i,pmin in enumerate(p2):
    #    print "%2i %-10s %12f +/- %10f"%(i,"vfix",pmin,np.sqrt(cov[i,i])*np.sqrt(chisq/dof))
    #print
