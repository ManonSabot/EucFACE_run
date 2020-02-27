#!/usr/bin/env python

__author__ = "MU Mengyuan"
__version__ = "2019-10-06"
__changefrom__ = 'plot_eucface_swc_cable_vs_obs_obsved_dates-13-layer.py'

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import griddata
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from plot_eucface_waterbal import *
from plot_eucface_swc_tdr import *
from plot_eucface_swc_profile import *

if __name__ == "__main__":

    ring  = "amb"

    contour = False

    case_def = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_only_6_"
    fcbl_def ="%s/EucFACE_%s_out.nc" % (case_def, ring)

    #case_best = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_GW-wb_SM-fix_or_fix_fw-hie-exp"
    case_best = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-hie-exp"
    fcbl_best ="%s/EucFACE_%s_out.nc" % (case_best, ring)

    #case_fw_std = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_GW-wb_SM-fix_or_fix"
    case_fw_std = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter"
    fcbl_fw_std ="%s/EucFACE_%s_out.nc" % (case_fw_std, ring)
    '''
    #figure 1
    plot_profile(fcbl_def, case_def, ring, contour, "6")
    #figure 2
    plot_ET(fcbl_def, case_def, ring)
    #plot_tdr(fcbl_def, case_def, ring, "6")

    #figure 3
    plot_profile(fcbl_best, case_best, ring, contour, "31uni")
    #figure 4
    plot_ET(fcbl_best, case_best, ring)
    #plot_tdr(fcbl_best, case_best, ring, "31uni")

    #figure 5
    plot_Rain_Fwsoil(fcbl_def, fcbl_fw_std, fcbl_best, ring)
    #figure 6
    calc_waterbal(fcbl_def, fcbl_best, "6", "31uni")
    '''
    fwatbal_def = "./EucFACE_def_met_only_6_.csv"
    #"/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/doc/scripts/EucFACE_def_%s.csv" %(fcbl_def.split("/")[-1])#[-2])
    fwatbal_best= "./EucFACE_best_met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-hie-exp.csv"
    #fwatbal_best= "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/doc/scripts/EucFACE_best_%s.csv" %(fcbl_best.split("/")[-1])#[-2])
    plot_waterbal(fwatbal_def,fwatbal_best)
