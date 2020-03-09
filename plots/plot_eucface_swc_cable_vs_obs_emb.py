#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

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
from plot_eucface_swc_cable_vs_obs_neo import *
from plot_eucface_swc_cable_vs_obs_tdr import *
from plot_eucface_swc_cable_vs_obs_profile import *

if __name__ == "__main__":

    layer = "6" #"6" #"31uni"


    cases = [
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_6_litter",\
             "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_SM_6_litter"
              ] 
    # cases = [
    #           "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter",\
    #           "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x1-x1_litter",\
    #           "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter",\
    #           "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-exp",\
    #           "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_Hvrd",\
    #           "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx10-x100-x100_litter_hie-watpot"\
    #           ]
    #cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_opt_31uni_hyds-30cm/outputs",\
    #                  "met_LAI_vrt_swilt-watr-ssat_SM_31uni_hyds^*_litter"))

    contour = False

    rings = ["amb"]#"R1","R2","R3","R4","R5","R6",,"ele"

    for case_name in cases:
        for ring in rings:
            fcable ="%s/EucFACE_%s_out.nc" % (case_name, ring)
            plot_profile_ET(fcable, case_name, ring, contour, layer)
            # plot_profile(fcable, case_name, ring, contour, layer)
            # #plot_neo(fcable, case_name, ring, layer)
            # #plot_tdr(fcable, case_name, ring, layer)
            # plot_tdr_ET(fcable, case_name, ring, layer)
            # #plot_dry_down(fcable, case_name, ring, layer)


'''
cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_2bch-mid-bot/outputs",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=4-2_fw-hie-exp_fix"))
# bch-mid-bot 4. 2.
'''
'''
cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_5bch-top50-mid-bot/outputs",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=8-3-4_bch=6-3_fw-hie-exp_fix"))
# 5bch-top50-mid-bot 8 3 4 6 3
'''
'''
cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_3bch-top50/outputs",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=8-2-5_fw-hie-exp_fix"))
# 3bch-top50 8.0, 2.0 and 5.0
'''
'''
cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_opt_31uni_bch-50cm/outputs",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=130_fw-hie-exp"))
# bch-top50 13.
'''
'''
cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_opt_31uni_bch-bot/outputs",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=15_fw-hie-exp"))
# bch-bot 1.5
'''
'''
cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_opt_31uni_bch-mid/outputs",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=15_fw-hie-exp"))
# bch-mid 1.5
'''
'''
txt_info is  when hyds are -3.0, -7.0 and -7.0, the min rmse is 0.02408951555725588
# 3hyds-50cm
txt_info is  when bch are 13.0, 10.0 and 2.0, the min rmse is 0.008947011456542784
# 3bch
'''
