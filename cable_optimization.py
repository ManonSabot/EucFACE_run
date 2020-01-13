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
import subprocess
import multiprocessing as mp
import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime
from cable_run_optimization import RunCable

class Optimization(object):

    def __init__(self, met_dir=None, log_dir=None, output_dir=None,
                 restart_dir=None, aux_dir=None, namelist_dir=None,
                 nml_fname="cable.nml",
                 veg_fname="def_veg_params_zr_clitt_albedo_fix.txt",
                 soil_fname="def_soil_params.txt",
                 grid_fname="gridinfo_mmy_MD_elev_orig_std_avg-sand_mask.nc",
                 #grid_fname="gridinfo_CSIRO_1x1.nc",
                 phen_fname="modis_phenology_csiro.txt",
                 cnpbiome_fname="pftlookup_csiro_v16_17tiles.csv",
                 #elev_fname="GSWP3_gwmodel_parameters.nc",
                 lai_dir=None, fixed_lai=None,
                 met_subset=[], cable_src=None, cable_exe="cable", mpi=True,
                 num_cores=None, verbose=True):

        the same path = /srv/ccrc/data25/z5218916/cable/EucFACE
        case_name     = EucFACE_run_sen_31uni_3bch-top50

        met_fname = os.path.basename(met_dir_new).split("/")[-1]
        case_name = "fw-hie-exp_fix"
        cable_exe = "cable_met_GW-wb_HDM_fw-hie_SM-fix_or_31uni"
        #------------- Change stuff ------------- #
        met_dir = met_dir_new
        log_dir = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_3bch-top50/logs/%s_%s" % ( met_fname, case_name )
        output_dir = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_3bch-top50/outputs/%s_%s" % ( met_fname, case_name )
        restart_dir = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_3bch-top50/restart_files/%s_%s" % ( met_fname, case_name )
        namelist_dir = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_3bch-top50/namelists/%s_%s" % ( met_fname, case_name )
        aux_dir = "/srv/ccrc/data25/z5218916/cable/src/CABLE-AUX/"
        cable_src = "/srv/ccrc/data25/z5218916/cable/src/Marks_latest_branch_with_fixes/Marks_latest_branch_with_fixes_gw_for_EucFace_fix_met_multi-layer/"
        mpi = False

    def test_range(self, sen_para, operator, opt_layer, sen_value):
        if (not os.path.exists(self.met_dir)):
            raise Exception("No met folder: %s" % self.met_dir)
        else:
            print("we are changing met file")
            met_dir_new = alter_met_parameter(sen_para, operator, opt_layer, sen_value)

        print("we are running cable")

        C = RunCable(met_dir=met_dir, log_dir=log_dir, output_dir=output_dir,
                     restart_dir=restart_dir, aux_dir=aux_dir,
                     namelist_dir=namelist_dir, met_subset=met_subset,
                     cable_src=cable_src,cable_exe=cable_exe, mpi=mpi, num_cores=num_cores,
                     met_fname=met_fname, case_name=case_name)
        output_file = C.main() # change C.main() to return output_file

        cable_var, obs_var = get_cable_value(ref_var, output_file, layer, ring)

        metric = get_metric(metric_type, cable_var, obs_var)

        return metric

    def test_range_mpi(self, sen_para, operator, opt_layer, value1, sen_values):

        for value2 in sen_values2:
            for value3 in sen_values3:
                sen_value = [value1,value2,value3]
                print("sen_value are ", sen_value)
                print("we are trying %s%s%s-%s-%s" %(sen_para[0], operator[0], sen_value[0], sen_value[1], sen_value[2]))

                if (not os.path.exists(self.met_dir)):
                    raise Exception("No met folder: %s" % self.met_dir)
                else:
                    print("we are changing met file")
                    met_dir_new = alter_met_parameter(sen_para, operator, opt_layer, sen_value)

                print("we are running cable")

                C = RunCable(self, met_dir=met_dir, log_dir=log_dir, output_dir=output_dir,
                                 restart_dir=restart_dir, aux_dir=aux_dir,
                                 namelist_dir=namelist_dir, met_subset=met_subset,
                                 cable_src=cable_src,cable_exe=cable_exe, mpi=mpi, num_cores=num_cores,
                                 met_fname=met_fname, case_name=case_name)
                output_file = C.main() # change C.main() to return output_file

                cable_var, obs_var = get_cable_value(ref_var, output_file, layer, ring)

                metric = get_metric(metric_type, cable_var, obs_var)

                return metric

    def residuals_min/leastsq(sen_values, *args):

        (met_case, met_dir, met_subset, sen_para, operator, obs) = args

        output_file = main(sen_para, sen_value, operator, met_dir, met_subset)

        cable_var, obs_var = get_var_value(ref_var, output_file, met_case)

        metric = get_metric(metric_type, cable_var, obs_var)

        def main(sen_para, sen_value, operator, met_dir, met_subset):
            print("____________________________________")
            print(sen_value)
            print("____________________________________")
            if (not os.path.exists(met_dir)):
                raise Exception("No met folder: %s" %met_dir)
            else:
                print("we are changing met file")
                met_dir_new = alter_met_parameter(sen_para, sen_value, operator, met_dir, met_subset)

            print("we are running cable")
            C = RunCable(met_dir=met_dir, log_dir=log_dir, output_dir=output_dir,
                         restart_dir=restart_dir, aux_dir=aux_dir,
                         namelist_dir=namelist_dir, met_subset=met_subset,
                         cable_src=cable_src,cable_exe=cable_exe, mpi=mpi, num_cores=num_cores,
                         met_fname=met_fname, case_name=case_name)
            C.main()

            output_file = os.path.join(output_dir, "EucFACE_%s_out.nc" % os.path.basename(met_subset).split(".")[0].split("_")[-1])

        return metric


    def get_metric(metric_type, cable_var, obs_var):
        if metric_type == "rmse":
            metric = np.sqrt(np.mean((obs_var - cable_var)**2))
        elif metric_type == "r":

        return metric




    def alter_met_parameter(sen_para, operator, opt_layer, sen_value):

        met_dir_new = "%s_%s%s%s-%s-%s" %(met_dir, sen_para[0], operator[0] ,sen_value[0].replace('.', ''),\
                                           sen_value[1].replace('.', ''),sen_value[2].replace('.', '') )

        if len(met_subset) == 0:
            if os.path.exists(met_dir_new):
                shutil. rmtree(met_dir_new)
            shutil.copytree(met_dir, met_dir_new)
            met_files = glob.glob(os.path.join(met_dir_new, "*.nc"))
        else:
            if (not os.path.exists(met_dir_new)):
                os.makedirs(met_dir_new)
            for i in met_subset:
                shutil.copy(os.path.join(met_dir, i) , met_dir_new)
            met_files = glob.glob(os.path.join(met_dir_new, "*.nc"))
        for met_file in met_files:
            f = nc.Dataset(met_file, 'r+', format='NETCDF4')
            f.parameter_sensitivity = 'alter %s as %s-%s-%s' \
                                    % (sen_para[0], sen_value[0], sen_value[1], sen_value[2])
            for i in np.arange(3):
                sen_para_vec = "%s_vec" %sen_para[i]
                if operator[i] == "x":
                    print(sen_para[i])
                    f.variables[sen_para_vec][i,0,0] = f.variables[sen_para_vec][i,0,0] * float(sen_value[i])
                elif operator[i] == "=":
                    f.variables[sen_para_vec][i,0,0] = float(sen_value[i])
                elif operator[i] == "+":
                    f.variables[sen_para_vec][i,0,0] = f.variables[sen_para_vec][i,0,0] + float(sen_value[i])
                elif operator[i] == "-":
                    f.variables[sen_para_vec][i,0,0] = f.variables[sen_para_vec][i,0,0] - float(sen_value[i])
                elif operator[i] == "^":
                    f.variables[sen_para_vec][i,0,0] = 10.**float(sen_value[i])
            sen_para_type = np.unique(sen_para)
            for para_type in sen_para_type:
                para_type_vec = "%s_vec" % para_type
                if para_type == 'hyds':
                    f.variables[para_type][:,0] = f.variables[para_type_vec][:,0,0].mean()/1000.
                else:
                    f.variables[para_type][:,0] = f.variables[para_type_vec][:,0,0].mean()
            f.close()
        return met_dir_new


def optimization():

    ring       = "amb"
    met_case   = "met_LAI_vrt_swilt-watr-ssat_SM_31uni"
    met_dir    = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_opt_fw-hie-exp_31uni/met/%s" % met_case
    met_subset = "EucFACE_met_%s.nc" % ring

    optimize    = "" # "leastsq" ; "minimize"; "MCMC"; "range"
    ref_var     = "" # "swc_50"; "swc_all"; "trans"; "esoil"; "esoil2trans"

    local_exe = "cable"
    cable_src = "/srv/ccrc/data25/z5218916/cable/src/Marks_latest_branch_with_fixes/Marks_latest_branch_with_fixes_gw_for_EucFace_fix_met_multi-layer/offline/cable_met_GW-wb_HDM_fw-hie_SM-fix_or_31uni"
    if os.path.isfile(local_exe):
        os.remove(local_exe)
    shutil.copy(cable_src, local_exe)

    sen_para  = ["hyds","bch"]
    operator  = ["^","="]
    opt_layer = ["top1","top"] # "top1","top2","top3","mid","bot"

    D = Optimization(met_dir=met_dir, log_dir=log_dir, output_dir=output_dir,
                    restart_dir=restart_dir, aux_dir=aux_dir,
                    namelist_dir=namelist_dir, met_subset=met_subset,
                    cable_src=cable_src, mpi=mpi, num_cores=num_cores)

    if len(sen_para) > 1:
        mpi = True

    if optimize == "leastsq":

        sen_value = np.array([0.09])  # initial_guess
        (popt, pcov, info, mesg, success) = optimize.leastsq(residuals, sen_value, \
                                           args=( met_case, met_dir, met_subset,  \
                                           sen_para, operator, obs),full_output=True,\
                                           ftol=100000., xtol=100000.,maxfev = 10000, epsfcn = 0.01)

    elif optimize == "minimize":

        sen_value = np.array([0.09])  # initial_guess
        res = optimize.minimize(residuals, sen_value,
                                args=( met_case, met_dir, met_subset,
                                       sen_para, operator, obs),
                                method='nelder-mead')
    elif optimize == "MCMC":
        swc_average_above_the_depth

    elif optimize == "range":
        sen_values = [ ["2.","3.","4.","5.","6.","7.","8."],\
                       ["2.","3.","4.","5.","6.","7.","8."]  ]

        if mpi:
            num_cores = mp.cpu_count()

            if int(len(sen_values[0,:])) <= num_cores:
                pool = mp.Pool(processes = int(len(sen_values[0,:])))
                processes = []

                for i in np.arange(len(sen_values[0,:])):
                    # setup a list of processes that we want to run

                    p = mp.Process(target=D.test_range_mpi,
                                   args=(sen_para, operator, opt_layer, sen_values[0,i],
                                         sen_values[1:,:])) # , met_dir, met_subset given by D
                    processes.append(p)
                # Run processes
                for p in processes:
                    p.start()
            else:
                print("int(len(sen_values1)) > num_cores, please divide jobs again")
        else:
            sen_values = ["2.","3.","4.","5.","6.","7.","8."]
            for sen_value in sen_values1:
                print("we are trying %s%s%s-%s-%s" %(sen_para[0], operator[0], sen_value[0], sen_value[1], sen_value[2]))
                D.test_range(sen_para = sen_para, operator = operator, opt_layer = opt_layer,
                             sen_value = sen_value)

    else:

        RMSE = residuals(met_case, met_dir, met_subset, sen_para, operator, obs)
        print("Please choose from 'leastsq', 'minimize', 'MCMC', 'range'")
