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
from plot_eucface_swc_tdr import *
from plot_eucface_swc_profile import *
from plot_eucface_waterbal import *
from plot_eucface_drought import *
from plot_eucface_heatwave import *


if __name__ == "__main__":

    ring  = "amb"

    contour = True

    case_1 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6"
    fcbl_1 ="%s/EucFACE_%s_out.nc" % (case_1, ring)

    case_2 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6_litter"
    fcbl_2 ="%s/EucFACE_%s_out.nc" % (case_2, ring)

    case_3 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter"
    fcbl_3 ="%s/EucFACE_%s_out.nc" % (case_3, ring)

    case_4 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter"
    fcbl_4 ="%s/EucFACE_%s_out.nc" % (case_4, ring)

    case_5 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_Hvrd"
    fcbl_5 ="%s/EucFACE_%s_out.nc" % (case_5, ring)

    case_6 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_hie-exp"
    fcbl_6 ="%s/EucFACE_%s_out.nc" % (case_6, ring)

    case_7 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_LAIx08_31uni_litter"
    fcbl_7 ="%s/EucFACE_%s_out.nc" % (case_7, ring)

    case_8 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_LAI-08_31uni_litter"
    fcbl_8 ="%s/EucFACE_%s_out.nc" % (case_8, ring)


    #case_10 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-watpot"
    #fcbl_10 ="%s/EucFACE_%s_out.nc" % (case_10, ring)

    fcables     = [fcbl_1,  fcbl_2,   fcbl_3,  fcbl_4,   fcbl_5,  fcbl_6] # , fcbl_7,    fcbl_8
    case_labels = ["Ctl",    "Lit", "Hi-Res",    "Hc", "β-hvrd", "β-exp"] # , "LAIx0.8", "LAI-0.8"
    layers      = [  "6",      "6",  "31uni",  "31uni", "31uni", "31uni"] # , "31uni",   "31uni"
    time_scale  = "hourly"
    ring        = "amb"

    '''
    Profile
    '''
    # plot_profile_tdr_ET(fcbl_1, ring, contour,    "6")
    # plot_profile_tdr_ET(fcbl_2, ring, contour,    "6")
    # plot_profile_tdr_ET(fcbl_3, ring, contour,"31uni")
    # plot_profile_tdr_ET(fcbl_4, ring, contour,"31uni")
    # plot_profile_tdr_ET(fcbl_5, ring, contour,"31uni")
    # plot_profile_tdr_ET(fcbl_6, ring, contour,"31uni")
    # plot_profile_tdr_ET(fcbl_7, ring, contour,"31uni")
    # plot_profile_tdr_ET(fcbl_8, ring, contour,"31uni")


    # fpath1 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6"
    # fpath2 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6_litter"
    # fpath3 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter"
    # fpath4 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter"
    # fpath5 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_Hvrd"
    # fpath6 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_hie-exp"
    #
    # case_name1 = "met_LAI_6"
    # case_name2 = "met_LAI_6_litter"
    # case_name3 = "met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter"
    # case_name4 = "met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter"
    # case_name5 = "met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_Hvrd"
    # case_name6 = "met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_hie-exp"
    #
    # plot_profile_tdr_ET_error(fpath1, case_name1, ring, contour, '6')
    # plot_profile_tdr_ET_error(fpath2, case_name2, ring, contour, '6')
    # plot_profile_tdr_ET_error(fpath3, case_name3, ring, contour, '31uni')
    # plot_profile_tdr_ET_error(fpath4, case_name4, ring, contour, '31uni')
    # plot_profile_tdr_ET_error(fpath5, case_name5, ring, contour, '31uni')
    # plot_profile_tdr_ET_error(fpath6, case_name6, ring, contour, '31uni')

    '''
    Fwsoil plot
    '''
    #plot_fwsoil_boxplot_SM( fcables, case_labels, layers, ring)
    #plot_Fwsoil_days_bar(fcables, case_labels)

    '''
    drought plot
    '''
    #plot_Rain_Fwsoil_Trans_Esoil_EF_SM( fcables, case_labels, layers, ring)

    '''
    Heatwave plots
    '''
    group_boxplot_Qle_Qh_EF_HW(fcables, case_labels)
    #plot_EF_SM_HW(fcables, case_labels, layers, ring, time_scale)

    '''
    Water Balance
    '''
    #calc_waterbal(fcbl_1, '6')

    # fcsv = ["/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/doc/scripts/csv/EucFACE_amb_met_LAI_6.csv",
    #         "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/doc/scripts/csv/EucFACE_amb_met_LAI_6_litter.csv",
    #         "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/doc/scripts/csv/EucFACE_amb_met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter.csv",
    #         "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/doc/scripts/csv/EucFACE_amb_met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter.csv",
    #         "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/doc/scripts/csv/EucFACE_amb_met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_Hvrd.csv",
    #         "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/doc/scripts/csv/EucFACE_amb_met_LAI_vrt_swilt-watr-ssat_SM_hydsx1_x10_31uni_litter_hie-exp.csv"]
    #
    # plot_waterbal_no_total_Evap(fcsv, case_labels)
