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
from plot_eucface_metrics import *
from plot_eucface_swc_tdr import *
from plot_eucface_swc_profile import *
from plot_eucface_waterbal import *
from plot_eucface_drought import *
from plot_eucface_heatwave import *


if __name__ == "__main__":

    ring  = 'amb'
    # 'amb':'R2','R3','R6'
    # 'ele':'R1','R4','R5'

    contour = True

    # "Ctl"
    case_name1 = "met_LAI-08_6"
    # "Teuc"
    case_name2 = "met_LAI-08_6_teuc"
    # "Sres"
    case_name3 = "met_LAI-08_6_teuc_sres"
    # "Watr"
    case_name4 = "met_LAI-08_6_teuc_sres_watr"
    # "Hi-Res-1"
    case_name5 = "met_LAI-08_31uni_teuc_sres_watr"
    # "Hi-Res-2"
    case_name6 = "met_LAI-08_vrt_31uni_teuc_sres_watr"
    # "Opt"
    case_name7 = "met_LAI-08_vrt_swilt-watr-ssat_31uni_teuc_sres_watr"
    # "β-hvrd"
    case_name8 = "met_LAI-08_vrt_swilt-watr-ssat_31uni_teuc_sres_watr_beta-hvrd"
    # "β-exp"
    case_name9 = "met_LAI-08_vrt_swilt-watr-ssat_31uni_teuc_sres_watr_beta-exp"
    # "Opt-sub"
    case_name10 = "met_LAI-08_vrt_swilt-watr-ssat-all_31uni_teuc_sres_watr"

    # case_name11 = "met_LAI-08_6"
    # case_name12 = "met_LAI-08_6"
    # case_name13 = "met_LAI-08_6"

    case_1 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name1
    fcbl_1 ="%s/EucFACE_%s_out.nc" % (case_1, ring)

    case_2 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name2
    fcbl_2 ="%s/EucFACE_%s_out.nc" % (case_2, ring)

    case_3 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name3
    fcbl_3 ="%s/EucFACE_%s_out.nc" % (case_3, ring)

    case_4 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name4
    fcbl_4 ="%s/EucFACE_%s_out.nc" % (case_4, ring)

    case_5 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name5
    fcbl_5 ="%s/EucFACE_%s_out.nc" % (case_5, ring)

    case_6 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name6
    fcbl_6 ="%s/EucFACE_%s_out.nc" % (case_6, ring)

    case_7 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name7
    fcbl_7 ="%s/EucFACE_%s_out.nc" % (case_7, ring)

    case_8 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name8
    fcbl_8 = "%s/EucFACE_%s_out.nc" % (case_8, ring)

    case_9 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name9
    fcbl_9 = "%s/EucFACE_%s_out.nc" % (case_9, ring)

    case_10 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name10
    fcbl_10 = "%s/EucFACE_%s_out.nc" % (case_10, ring)

    # case_11 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name11
    # fcbl_11 = "%s/EucFACE_%s_out.nc" % (case_11, ring)
    #
    # case_12 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name12
    # fcbl_12 ="%s/EucFACE_%s_out.nc" % (case_12, ring)
    #
    # case_13 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name13
    # fcbl_13 = "%s/EucFACE_%s_out.nc" % (case_13, ring)

    fcables     = [fcbl_1,  fcbl_2,   fcbl_3,    fcbl_4,     fcbl_5,      fcbl_6,
                   fcbl_7,  fcbl_8,   fcbl_9,    fcbl_10 ]
    case_labels = ["Ctl",   "Teuc",   "Sres",    "Watr",  "Hi-Res-1", "Hi-Res-2",
                   "Opt", "β-hvrd",  "β-exp",  "Opt-sub" ]
    layers      = [  "6",      "6",      "6",       "6",     "31uni",    "31uni",
                   "31uni", "31uni", "31uni",    "31uni" ]
    time_scale  = "hourly"
    vars        = ['Esoil', 'Trans', 'VWC', 'SM_25cm', 'SM_15m', 'SM_bot']
    CTL         = case_name1
    '''
    statistics of ET observation
    '''
    # stat_obs(fcables, case_labels, ring)

    '''
    annual values
    '''
    # annual_values(fcables, case_labels, layers, ring)

    '''
    metrics
    '''
    # calc_metrics(fcables, case_labels, layers, vars, ring)

    '''
    Check ET
    '''
    # plot_check_ET(fcables, case_labels, ring)

    '''
    Profile
    '''

    fpath1 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name1
    fpath2 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name2
    fpath3 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name3
    fpath4 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name4
    fpath5 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name5
    fpath6 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name6
    fpath7 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name7
    fpath8 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name8
    fpath9 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name9
    fpath10 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name10
    # fpath11 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name11
    # fpath12 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name12
    # fpath13 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s" % case_name13

    plot_profile_tdr_ET_error_rain(CTL, fpath1, case_name1, ring, contour, '6')
    plot_profile_tdr_ET_error_rain(CTL, fpath2, case_name2, ring, contour, '6')
    plot_profile_tdr_ET_error_rain(CTL, fpath3, case_name3, ring, contour, '6')
    plot_profile_tdr_ET_error_rain(CTL, fpath4, case_name4, ring, contour, '6')
    plot_profile_tdr_ET_error_rain(CTL, fpath5, case_name5, ring, contour, '31uni')
    plot_profile_tdr_ET_error_rain(CTL, fpath6, case_name6, ring, contour, '31uni')
    plot_profile_tdr_ET_error_rain(CTL, fpath7, case_name7, ring, contour, '31uni')
    plot_profile_tdr_ET_error_rain(CTL, fpath8, case_name8, ring, contour, '31uni')
    plot_profile_tdr_ET_error_rain(CTL, fpath9, case_name9, ring, contour, '31uni')
    plot_profile_tdr_ET_error_rain(CTL, fpath10, case_name10, ring, contour, '31uni')
    # plot_profile_tdr_ET_error_rain(fpath11, case_name11, ring, contour, '6')
    # plot_profile_tdr_ET_error_rain(fpath11, case_name11, ring, contour, '31uni')
    # plot_profile_tdr_ET_error_rain(fpath12, case_name12, ring, contour, '31uni')
    # plot_profile_tdr_ET_error_rain(fpath13, case_name13, ring, contour, '31uni')
    #
    # plot_profile_ET_error_rain(fpath1, case_name1, ring, contour, '6')
    # plot_profile_ET_error_rain(fpath2, case_name2, ring, contour, '6')
    # plot_profile_ET_error_rain(fpath3, case_name3, ring, contour, '6')
    # plot_profile_ET_error_rain(fpath4, case_name4, ring, contour, '31uni')
    # plot_profile_ET_error_rain(fpath5, case_name5, ring, contour, '31uni')
    # plot_profile_ET_error_rain(fpath6, case_name6, ring, contour, '31uni')
    # plot_profile_ET_error_rain(fpath11, case_name11, ring, contour, '31uni')
    # plot_profile_ET_error_rain(fpath12, case_name12, ring, contour, '31uni')
    # plot_profile_ET_error_rain(fpath13, case_name13, ring, contour, '31uni')


    '''
    Fwsoil plot
    '''
    plot_fwsoil_boxplot_SM_days_bar( fcables, case_labels, layers, ring)

    '''
    Drought plot
    '''
    plot_Rain_Fwsoil_Trans_Esoil_SH_SM( fcables, case_labels, layers, ring)

    '''
    Heatwave plots
    '''
    plot_EF_SM_HW(fcables, case_labels, layers, ring, time_scale)

    '''
    Heatwave same preceding soil moisture plots
    '''

    fcables2      = [fcbl_4, fcbl_12,  fcbl_5,    fcbl_6]
    fcables_re    = [fcbl_7,  fcbl_8,  fcbl_9,    fcbl_10]
    case_labels2  = ["Hi-Res-2", "Opt", "β-Hvrd", "β-exp"]
    layers2       = ["31uni",  "31uni",  "31uni", "31uni"]

    #plot_case_study_HW_event(fcables2, fcables_re, case_labels2, ring, layers2)

    # '''
    # Water Balance
    # '''
    # # calc_waterbal(fcbl_1, '6', ring)
    # # calc_waterbal(fcbl_2, '6', ring)
    # # calc_waterbal(fcbl_3, '6', ring)
    # # calc_waterbal(fcbl_4, '31uni', ring)
    # # calc_waterbal(fcbl_5, '31uni', ring)
    # # calc_waterbal(fcbl_6, '31uni', ring)
    # # calc_waterbal(fcbl_7, '31uni', ring)

    path = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/doc/scripts/csv/"
    case_names = [ "met_LAI_6",
                   "met_LAI_6_litter",
                   "met_LAI_6_litter_gw-ssat-bom",
                   "met_LAI_31uni_litter_gw-ssat-bom",
                   "met_LAI_vrt_31uni_litter_gw-ssat-bom",
                   "met_LAI_vrt_swilt-watr-ssat_31uni_litter_gw-ssat-bom",
                   "met_LAI_vrt_swilt-watr-ssat_31uni_litter_gw-ssat-bom_beta-hvrd",
                   "met_LAI_vrt_swilt-watr-ssat_31uni_litter_gw-ssat-bom_beta-exp"]

    # plot_waterbal_no_total_Evap_imbalance(path, case_names, case_labels)
