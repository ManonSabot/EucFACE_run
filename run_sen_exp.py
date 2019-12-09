#!/usr/bin/env python

__author__    = "Martin De Kauwe"
__developer__ = "MU Mengyuan"

import os
import sys
import glob
import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import cable_run_sen_exp

def main(sen_para, sen_value, operator, met_dir, met_subset):
    if (not os.path.exists(met_dir)):
        raise Exception("No met folder: %s" %met_dir)
    else:
        met_dir_new = alter_met_parameter(sen_para, sen_value, operator, met_dir, met_subset):
     run_cable(met_dir_new, met_subset)

def alter_met_parameter(sen_para, sen_value, operator, met_dir, met_subset):

    met_dir_new = os.path.join(met_dir, "_%s%s%s" %( sen_para, operator ,sen_value.replace('.', '') ))

    if len(met_subset) == 0:
        shutil.copytree(met_dir, met_dir_new)
        met_files = glob.glob(os.path.join(met_dir_new, "*.nc"))
    else:
        if (not os.path.exists(met_dir_new)):
            os.makedirs(met_dir_new)
        for i in met_subset:
            shutil.copy(os.path.join(met_dir, i) , met_dir_new)
        met_files = glob.glob(os.path.join(met_dir_new, "*.nc"))
    for met_file in met_files:
        f = nc.Dataset(met_file, 'w', format='NETCDF4')
        f.parameter_sensitivity = 'alter %s as %s%s%s' % (sen_para, sen_para, operator, sen_value)

        sen_para_vec = "%s_vec" %sen_para

        if operator == "x":
            f.variables[sen_para][:]         = f.variables[sen_para][:] * sen_value
            f.variables[sen_para_vec][:,0,0] = f.variables[sen_para_vec][:,0,0] * sen_value
        elif operator == "=":
            f.variables[sen_para][:]         = sen_value
            f.variables[sen_para_vec][:,0,0] = sen_value
        elif operator == "+":
            f.variables[sen_para][:]         = f.variables[sen_para][:,0,0] + sen_value
            f.variables[sen_para_vec][:,0,0] = f.variables[sen_para_vec][:,0,0] + sen_value
        elif operator == "-":
            f.variables[sen_para][:]         = f.variables[sen_para][:] - sen_value
            f.variables[sen_para_vec][:,0,0] = f.variables[sen_para_vec][:,0,0] - sen_value
        f.close()
    return met_dir_new

def run_cable(met_dir_new, met_subset):

    met_fname = os.path.basename(met_dir_new).split("/")[-1]
    case_name = ""
    cable_exe = "cable_met_HDM_31uni"
    #------------- Change stuff ------------- #
    met_dir = met_dir_new
    log_dir = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/logs/%s_%s" % ( met_fname, case_name )
    output_dir = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s_%s" % ( met_fname, case_name )
    restart_dir = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/restart_files/%s_%s" % ( met_fname, case_name )
    namelist_dir = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/namelists/%s_%s" % ( met_fname, case_name )
    aux_dir = "/srv/ccrc/data25/z5218916/cable/src/CABLE-AUX/"
    cable_src = "/srv/ccrc/data25/z5218916/cable/src/Marks_latest_branch_with_fixes/Marks_latest_branch_with_fixes_gw_for_EucFace_fix_met_multi-layer/"
    mpi = False
    num_cores = 4 # set to a number, if None it will use all cores...!

    C = RunCable(met_dir=met_dir, log_dir=log_dir, output_dir=output_dir,
                 restart_dir=restart_dir, aux_dir=aux_dir,
                 namelist_dir=namelist_dir, met_subset=met_subset,
                 cable_src=cable_src,cable_exe=cable_exe, mpi=mpi, num_cores=num_cores,
                 met_fname=met_fname, case_name=case_name)
    C.main()


if __name__ == "__main__":

    sen_para   = "hyds"
    operator   = "x"
    sen_values = ["0.01","0.1","10.","100."]

    met_case   = "met_LAI_vrt_SM_swilt-watr_31uni"
    met_dir    = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/met/%s" %met_case
    met_subset = []

    for sen_value in sen_values:
        main(sen_para = sen_para, sen_values = sen_value,\
             operator=operator, met_dir = met_dir, met_subset = met_subset )
