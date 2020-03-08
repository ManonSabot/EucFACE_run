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
from plot_eucface_swc_tdr_ctl_litter_best import *
from plot_eucface_swc_profile_obs_ctl_best import *
from plot_eucface_waterbal import *
#from plot_eucface_swc_tdr import *
#from plot_eucface_swc_profile import *

if __name__ == "__main__":

    ring  = "amb"

    contour = False

    case_1 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6"
    fcbl_1 ="%s/EucFACE_%s_out.nc" % (case_1, ring)

    case_2 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6_litter"
    fcbl_2 ="%s/EucFACE_%s_out.nc" % (case_2, ring)

    case_3 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_SM_6_litter"
    fcbl_3 ="%s/EucFACE_%s_out.nc" % (case_3, ring)

    case_4 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_31uni_litter"
    fcbl_4 ="%s/EucFACE_%s_out.nc" % (case_4, ring)

    case_5 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter"
    fcbl_5 ="%s/EucFACE_%s_out.nc" % (case_5, ring)

    case_6 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter"
    fcbl_6 ="%s/EucFACE_%s_out.nc" % (case_6, ring)

    case_7 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_hie-exp"
    fcbl_7 ="%s/EucFACE_%s_out.nc" % (case_7, ring)

    case_8 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_Hvrd"
    fcbl_8 ="%s/EucFACE_%s_out.nc" % (case_8, ring)

    case_9 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_hie-watpot"
    fcbl_9 ="%s/EucFACE_%s_out.nc" % (case_9, ring)

    #plot_profile(fcbl_1, fcbl_7, ring, contour)
    #plot_ET(fcbl_1, fcbl_2, fcbl_7, ring)
    #plot_Rain_Fwsoil_Trans(fcbl_1, fcbl_6, fcbl_7, ring)
    #plot_Rain_Fwsoil_Trans(fcbl_6, fcbl_8, fcbl_7, fcbl_9, ring)
    plot_EF_SM(fcbl_6, fcbl_8, fcbl_7, fcbl_9, ring, "31uni")
    #plot_EF_SM_HW(fcbl_6, fcbl_8, fcbl_7, fcbl_9, ring, "31uni")
    fwatbal_1 = "./csv/EucFACE_met_LAI_6.csv"
    fwatbal_6 = "./csv/EucFACE_met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter.csv"
    fwatbal_7 = "./csv/EucFACE_met_LAI_vrt_swilt-watr-ssat_SM_hydsx10_31uni_litter_hie-exp.csv"

    #plot_waterbal(fwatbal_1,fwatbal_6,fwatbal_7)

    '''
    plot_profile(fcbl_1, case_1, ring, contour, "6")
    plot_ET(fcbl_1, case_1, ring)
    plot_tdr(fcbl_1, case_1, ring, "6")

    plot_profile(fcbl_2, case_2, ring, contour, "6")
    plot_ET(fcbl_2, case_2, ring)
    plot_tdr(fcbl_2, case_2, ring, "6")

    plot_profile(fcbl_3, case_3, ring, contour, "6")
    plot_ET(fcbl_3, case_3, ring)
    plot_tdr(fcbl_3, case_3, ring, "6")

    plot_profile(fcbl_4, case_4, ring, contour, "31uni")
    plot_ET(fcbl_4, case_4, ring)
    plot_tdr(fcbl_4, case_4, ring, "31uni")

    plot_profile(fcbl_5, case_5, ring, contour, "31uni")
    plot_ET(fcbl_5, case_5, ring)
    plot_tdr(fcbl_5, case_5, ring, "31uni")

    plot_profile(fcbl_6, case_6, ring, contour, "31uni")
    plot_ET(fcbl_6, case_6, ring)
    plot_tdr(fcbl_6, case_6, ring, "31uni")

    plot_profile(fcbl_7, case_7, ring, contour, "31uni")
    plot_ET(fcbl_7, case_7, ring)
    plot_tdr(fcbl_7, case_7, ring, "31uni")

    plot_profile(fcbl_8, case_8, ring, contour, "31uni")
    plot_ET(fcbl_8, case_8, ring)
    plot_tdr(fcbl_8, case_8, ring, "31uni")


    plot_Rain_Fwsoil(fcbl_1, fcbl_2, fcbl_3, ring)
    plot_Rain_Fwsoil(fcbl_4, fcbl_5, fcbl_6, ring)
    plot_Rain_Fwsoil(fcbl_6, fcbl_7, fcbl_8, ring)

    calc_waterbal(fcbl_1, fcbl_2, "6", "6")
    calc_waterbal(fcbl_1, fcbl_3, "6", "6")

    calc_waterbal(fcbl_4, fcbl_5, "31uni", "31uni")
    calc_waterbal(fcbl_6, fcbl_7, "31uni", "31uni")
    calc_waterbal(fcbl_7, fcbl_8, "31uni", "31uni")

    '''
