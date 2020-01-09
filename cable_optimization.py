#!/usr/bin/env python

"""
Purpose: provide 3 optimization methods (MCMC, ,range) to find the optimal parameter suiting the
         observation
"""

__author__  = "MU Mengyuan"
__version__ = "1.0 (2020-01-09)"
__email__   = "mu.mengyuan815@gmail.com"

import os
import sys
import glob
import shutil
import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime
from cable_run_optimization import RunCable

def residuals(sen_values, *args):

    (met_case, met_dir, met_subset, sen_para, operator, obs) = args

    output_file = main(sen_para, sen_value, operator, met_dir, met_subset)

    cable_var, obs_var = get_cable_value(ref_var, output_file, met_case)

    metric = get_metric(metric_type, cable_var, obs_var)

    return metric

def get_metric(metric_type, cable_var, obs_var):
    if metric_type == "rmse":
        metric = np.sqrt(np.mean((obs_var - cable_var)**2))
    elif metric_type == "r":

    return metric

def get_var_value(ref_var, output_file, met_case):

    choose_cable_var = {
                     'swc_50'     : read_cable_swc_50cm(output_file, met_case),
                     'swc_all'    : read_cable_swc_all(output_file, met_case),
                     'trans'      : read_cable_var(output_file, 'TVeg'),
                     'esoil'      : read_cable_var(output_file, 'ESoil'),
                     'esoil2trans': calc_cable_esoil2trans(output_file)
                     }
    choose_obs_var = {
                     'swc_50'     : read_obs_swc_tdr(ring),
                     'swc_all'    : read_obs_swc_neo(ring),
                     'trans'      : read_obs_trans(ring),
                     'esoil'      : read_obs_esoil(ring),
                     'esoil2trans': calc_obs_esoil2trans(ring)
                     }

    cable_var = choose_cable_var.get(ref_var, 'default')
    obs_var   = choose_obs_var.get(ref_var, 'default')

    return get_same_dates(cable_var, obs_var)

def get_same_dates(cable_var, obs_var):

    return cable_var, obs_var

def optimization():

    ring       = "amb"
    met_case   = "met_LAI_vrt_swilt-watr-ssat_SM_31uni"
    met_dir    = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_opt_fw-hie-exp_31uni/met/%s" % met_case
    met_subset = "EucFACE_met_%s.nc" % ring

    optimize    = "" # "leastsq" ; "minimize"; "MCMC"; "range"
    ref_var     = "" # "swc_50"; "swc_all"; "trans"; "esoil"; "esoil2trans"

    # "leastsq" ; "minimize"; "MCMC"; "range"

    if optimize == "leastsq":
        sen_para  = "hyds"
        operator  = "="
        sen_value = np.array([0.09])  # initial_guess
        (popt, pcov, info, mesg, success) = optimize.leastsq(residuals, sen_value, \
                                           args=( met_case, met_dir, met_subset,  \
                                           sen_para, operator, obs),full_output=True,\
                                           ftol=100000., xtol=100000.,maxfev = 10000, epsfcn = 0.01)

    elif optimize == "minimize":
        sen_para  = "hyds"
        operator  = "="
        sen_value = np.array([0.09])  # initial_guess
        res = optimize.minimize(residuals, sen_value,
                                args=( met_case, met_dir, met_subset,
                                       sen_para, operator, obs),
                                method='nelder-mead')
