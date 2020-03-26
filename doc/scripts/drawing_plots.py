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

    case_6 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x1-x1_litter"
    fcbl_6 ="%s/EucFACE_%s_out.nc" % (case_6, ring)

    case_7 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter"
    fcbl_7 ="%s/EucFACE_%s_out.nc" % (case_7, ring)

    case_8 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_Hvrd"
    fcbl_8 ="%s/EucFACE_%s_out.nc" % (case_8, ring)

    case_9 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-exp"
    fcbl_9 ="%s/EucFACE_%s_out.nc" % (case_9, ring)

    case_10 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-watpot"
    fcbl_10 ="%s/EucFACE_%s_out.nc" % (case_10, ring)

    case_11 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_LAIx08_31uni_litter_LAIx08"
    fcbl_11 ="%s/EucFACE_%s_out.nc" % (case_11, ring)

    case_12 = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_LAIx12_31uni_litter_LAIx12"
    fcbl_12 ="%s/EucFACE_%s_out.nc" % (case_12, ring)

    fcables     = [fcbl_1, fcbl_2, fcbl_5,   fcbl_6,    fcbl_7, fcbl_8,  fcbl_9]#,  fcbl_11, fcbl_12]
    case_labels = ["Ctl",  "Lit",  "Hi-Res", "Opt-top", "Opt",  "β-hvrd","β-exp"]#, "Hi-Res-LAI-20", "Hi-Res-LAI+20"]
    layers      = ["6",    "6",    "31uni",  "31uni",   "31uni","31uni", "31uni"]#, "31uni", "31uni"]
    time_scale  = "hourly"
    ring        = "amb"
    plot_Fwsoil_days_bar(fcables, case_labels)
    plot_Fwsoil_boxplot(fcables, case_labels)
    #plot_GPP(fcables, ring, case_labels)


    '''
    fcables     = [fcbl_1, fcbl_5,  fcbl_11,  fcbl_12 ]
    case_labels = ["Ctl", "Hi-Res", "Hi-Res-LAI-20","Hi-Res-LAI+20"]
    layers      = ["6",   "31uni", "31uni", "31uni"]
    plot_Fwsoil_Trans(fcables, ring, case_labels)
    '''

    #sen_values = np.linspace(-10.,5.,31)
    #for i,sen_value in enumerate(sen_values):
    #    fcable = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_opt_31uni_hyds-30cm/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hyds^%s_litter/EucFACE_amb_out.nc" % str(sen_value).replace('.', '')
    #    plot_profile_tdr_ET(fcable, ring, contour,    "31uni")

    # plot_profile_tdr_ET(fcbl_1, ring, contour,    "6")
    # plot_profile_tdr_ET(fcbl_2, ring, contour,    "6")
    # plot_profile_tdr_ET(fcbl_5, ring, contour,"31uni")
    # plot_profile_tdr_ET(fcbl_6, ring, contour,"31uni")
    # plot_profile_tdr_ET(fcbl_7, ring, contour,"31uni")
    # plot_profile_tdr_ET(fcbl_8, ring, contour,"31uni")
    # plot_profile_tdr_ET(fcbl_9, ring, contour,"31uni")
    '''
    fcables     = [fcbl_1 ]
    case_labels = ["Ctl"  ]
    layers      = ["6"    ]
    '''
    #plot_Rain_Fwsoil_Trans_Esoil_EF_SM(fcables, ring, layers, case_labels)

    fcables = [
                "./csv/EucFACE_amb_met_LAI_6.csv",
                "./csv/EucFACE_amb_met_LAI_6_litter.csv",
                "./csv/EucFACE_amb_met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter.csv",
                "./csv/EucFACE_amb_met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x1-x1_litter.csv",
                "./csv/EucFACE_amb_met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter.csv",
                "./csv/EucFACE_amb_met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_Hvrd.csv",
                "./csv/EucFACE_amb_met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-exp.csv"
                ]

    #plot_waterbal(fcables,case_labels)
