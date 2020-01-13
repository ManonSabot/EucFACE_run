#!/usr/bin/env python

__author__    = "Martin De Kauwe"
__developer__ = "MU Mengyuan"

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
from cable_run_sen_exp import RunCable

def main_mpi( sen_para, operator, value1 , sen_values2, sen_values3, \
              met_dir, met_subset ):

    for value2 in sen_values2:
        for value3 in sen_values3:
            sen_value = [value1,value2,value3]
            print("sen_value are ", sen_value)
            print("we are trying %s%s%s-%s-%s" %(sen_para[0], operator[0], sen_value[0], sen_value[1], sen_value[2]))
            main(sen_para = sen_para, operator = operator, sen_value = sen_value, \
                     met_dir = met_dir, met_subset = met_subset )

def main(sen_para, operator, sen_value, met_dir, met_subset):
    if (not os.path.exists(met_dir)):
        raise Exception("No met folder: %s" %met_dir)
    else:
        print("we are changing met file")
        met_dir_new = alter_met_parameter(sen_para, operator, sen_value)

    print("we are running cable")
    run_cable(met_dir_new, met_subset)

def alter_met_parameter(sen_para, operator, sen_value):

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

def run_cable(met_dir_new, met_subset):

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


    num_cores = 4 # set to a number, if None it will use all cores...!

    C = RunCable(met_dir=met_dir, log_dir=log_dir, output_dir=output_dir,
                 restart_dir=restart_dir, aux_dir=aux_dir,
                 namelist_dir=namelist_dir, met_subset=met_subset,
                 cable_src=cable_src,cable_exe=cable_exe, mpi=mpi, num_cores=num_cores,
                 met_fname=met_fname, case_name=case_name)
    C.main()


if __name__ == "__main__":

    sen_para   = ["bch","bch","bch"]
    operator   = ["=","=","="]
    sen_values1 = ["2.","3.","4.","5.","6.","7.","8."]
    sen_values2 = ["2.","3.","4.","5.","6.","7.","8."]
    sen_values3 = ["2.","3.","4.","5.","6.","7.","8."]

    met_case   = "met_LAI_vrt_swilt-watr-ssat_SM_31uni"
    met_dir    = "/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_31uni_3bch-top50/met/%s" %met_case
    met_subset = ["EucFACE_met_amb.nc"]
    print("we start!")
    # delete local executable, copy a local copy and use that
    local_exe = "cable"
    cable_src = "/srv/ccrc/data25/z5218916/cable/src/Marks_latest_branch_with_fixes/Marks_latest_branch_with_fixes_gw_for_EucFace_fix_met_multi-layer/offline/cable_met_GW-wb_HDM_fw-hie_SM-fix_or_31uni"
    if os.path.isfile(local_exe):
        os.remove(local_exe)
    shutil.copy(cable_src, local_exe)

    mpi = True

    if mpi:
        num_cores = mp.cpu_count()

        if int(len(sen_values1)) <= num_cores:
            pool = mp.Pool(processes = int(len(sen_values1)))
            processes = []

            for i in np.arange(len(sen_values1)): 
                # setup a list of processes that we want to run

                p = mp.Process(target=main_mpi,
                               args=(sen_para, operator, sen_values1[i], \
                                     sen_values2, sen_values3, \
                                     met_dir, met_subset))
                processes.append(p)
            # Run processes
            for p in processes:
                p.start()
        else:
            print("int(len(sen_values1)) > num_cores, please divide jobs again")
    else:
        for sen_value[0] in sen_values1:
            for sen_value[1] in sen_values2:
                for sen_value[2] in sen_values3:
                    print("we are trying %s%s%s-%s-%s" %(sen_para[0], operator[0], sen_value[0], sen_value[1], sen_value[2]))
                    main(sen_para = sen_para, operator = operator, sen_value = sen_value, \
                         met_dir = met_dir, met_subset = met_subset )
