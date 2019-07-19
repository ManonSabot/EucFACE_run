#!/usr/bin/env python

"""
Plot EucFACE soil moisture

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (24.10.2018)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import netCDF4 as nc
import glob

def main(fobs, fout):

    neo = pd.read_csv(fobs, usecols = ['Ring','Depth','Date','VWC'])
    # usecols : read specific columns from CSV

    # divide neo into groups
    subset_amb = neo[neo['Ring'].isin(['R2','R3','R6'])]
    subset_ele = neo[neo['Ring'].isin(['R1','R4','R5'])]
    subset_R1  = neo[neo['Ring'].isin(['R1'])]
    subset_R2  = neo[neo['Ring'].isin(['R2'])]
    subset_R3  = neo[neo['Ring'].isin(['R3'])]
    subset_R4  = neo[neo['Ring'].isin(['R4'])]
    subset_R5  = neo[neo['Ring'].isin(['R5'])]
    subset_R6  = neo[neo['Ring'].isin(['R6'])]

    # calculate the mean of every group and unstack
    neo_mean = neo.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    amb_mean = subset_amb.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    ele_mean = subset_ele.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R1_mean  = subset_R1.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R2_mean  = subset_R2.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R3_mean  = subset_R3.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R4_mean  = subset_R4.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R5_mean  = subset_R5.groupby(by=["Depth","Date"]).mean().unstack(level=0)
    R6_mean  = subset_R6.groupby(by=["Depth","Date"]).mean().unstack(level=0)

    # remove 'VWC'
    neo_mean = neo_mean.xs('VWC', axis=1, drop_level=True)
    amb_mean = amb_mean.xs('VWC', axis=1, drop_level=True)
    ele_mean = ele_mean.xs('VWC', axis=1, drop_level=True)
    R1_mean  = R1_mean.xs('VWC', axis=1, drop_level=True)
    R2_mean  = R2_mean.xs('VWC', axis=1, drop_level=True)
    R3_mean  = R3_mean.xs('VWC', axis=1, drop_level=True)
    R4_mean  = R4_mean.xs('VWC', axis=1, drop_level=True)
    R5_mean  = R5_mean.xs('VWC', axis=1, drop_level=True)
    R6_mean  = R6_mean.xs('VWC', axis=1, drop_level=True)
    # 'VWC' : key on which to get cross section
    # axis=1 : get cross section of column
    # drop_level=True : returns cross section without the multilevel index

    # Converting the index as date
    neo_mean.index = pd.to_datetime(neo_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    amb_mean.index = pd.to_datetime(amb_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    ele_mean.index = pd.to_datetime(ele_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R1_mean.index  = pd.to_datetime(R1_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R2_mean.index  = pd.to_datetime(R2_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R3_mean.index  = pd.to_datetime(R3_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R4_mean.index  = pd.to_datetime(R4_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R5_mean.index  = pd.to_datetime(R5_mean.index,format="%d/%m/%y",infer_datetime_format=False)
    R6_mean.index  = pd.to_datetime(R6_mean.index,format="%d/%m/%y",infer_datetime_format=False)

    # Sort by date
    neo_mean = neo_mean.sort_values(by=['Date'])
    amb_mean = amb_mean.sort_values(by=['Date'])
    ele_mean = ele_mean.sort_values(by=['Date'])
    R1_mean = R1_mean.sort_values(by=['Date'])
    R2_mean = R2_mean.sort_values(by=['Date'])
    R3_mean = R3_mean.sort_values(by=['Date'])
    R4_mean = R4_mean.sort_values(by=['Date'])
    R5_mean = R5_mean.sort_values(by=['Date'])
    R6_mean = R6_mean.sort_values(by=['Date'])

    depths = np.unique(neo.Depth)
    ndepths = len(depths)
    dates_org = np.unique(amb_mean.index).astype('datetime64[D]')
    ntimes_org = len(dates_org)

    neo_mean = np.transpose(neo_mean)
    amb_mean = np.transpose(amb_mean)
    ele_mean = np.transpose(ele_mean)
    R1_mean = np.transpose(R1_mean)
    R2_mean = np.transpose(R2_mean)
    R3_mean = np.transpose(R3_mean)
    R4_mean = np.transpose(R4_mean)
    R5_mean = np.transpose(R5_mean)
    R6_mean = np.transpose(R6_mean)
    print(np.transpose(neo_mean))

    f = nc.Dataset(fout, 'w', format='NETCDF4')
    f.description = 'EucFACE swc data (%s), created by MMY', fobs
    f.history = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date = "%s" % (dt.datetime.now())


    # set dimensions
    f.createDimension('time', None)
    f.createDimension('depth', ndepths)
    #f.Conventions = "CF-1.0"

    # create variables
    time = f.createVariable('time', 'f8', ('time',))
    time.units = "seconds since 2012-01-01 00:00:00"
    time.long_name = "time"
    time.calendar = "standard"

    depth = f.createVariable('depth', 'f8', ('depth',))
    depth.long_name = "depth"
    depth.long_name = "soil depth"

    swc_neo = f.createVariable('swc_neo', 'f8', ( 'depth', 'time',))
    swc_neo.units = "%"
    swc_neo.missing_value = -9999.
    swc_neo.long_name = "Volumetric soil water content"
    swc_neo.CF_name = "Volumetric_soil_water_content_by_neutron_probe"

    swc_amb = f.createVariable('swc_amb', 'f8', ( 'depth', 'time',))
    swc_amb.units = "%"
    swc_amb.missing_value = -9999.
    swc_amb.long_name = "Volumetric soil water content"
    swc_amb.CF_name = "Volumetric_soil_water_content_by_neutron_probe"

    swc_ele = f.createVariable('swc_ele', 'f8', ( 'depth', 'time',))
    swc_ele.units = "%"
    swc_ele.missing_value = -9999.
    swc_ele.long_name = "Volumetric soil water content"
    swc_ele.CF_name = "Volumetric_soil_water_content_by_neutron_probe"

    swc_R1 = f.createVariable('swc_R1', 'f8', ( 'depth', 'time',))
    swc_R1.units = "%"
    swc_R1.missing_value = -9999.
    swc_R1.long_name = "Volumetric soil water content"
    swc_R1.CF_name = "Volumetric_soil_water_content_by_neutron_probe"

    swc_R2 = f.createVariable('swc_R2', 'f8', ( 'depth', 'time',))
    swc_R2.units = "%"
    swc_R2.missing_value = -9999.
    swc_R2.long_name = "Volumetric soil water content"
    swc_R2.CF_name = "Volumetric_soil_water_content_by_neutron_probe"

    swc_R3 = f.createVariable('swc_R3', 'f8', ( 'depth', 'time',))
    swc_R3.units = "%"
    swc_R3.missing_value = -9999.
    swc_R3.long_name = "Volumetric soil water content"
    swc_R3.CF_name = "Volumetric_soil_water_content_by_neutron_probe"

    swc_R4 = f.createVariable('swc_R4', 'f8', ( 'depth', 'time',))
    swc_R4.units = "%"
    swc_R4.missing_value = -9999.
    swc_R4.long_name = "Volumetric soil water content"
    swc_R4.CF_name = "Volumetric_soil_water_content_by_neutron_probe"

    swc_R5 = f.createVariable('swc_R5', 'f8', ( 'depth', 'time',))
    swc_R5.units = "%"
    swc_R5.missing_value = -9999.
    swc_R5.long_name = "Volumetric soil water content"
    swc_R5.CF_name = "Volumetric_soil_water_content_by_neutron_probe"

    swc_R6 = f.createVariable('swc_R6', 'f8', ( 'depth', 'time',))
    swc_R6.units = "%"
    swc_R6.missing_value = -9999.
    swc_R6.long_name = "Volumetric soil water content"
    swc_R6.CF_name = "Volumetric_soil_water_content_by_neutron_probe"

    neo_mean = np.where(np.isnan(neo_mean), -9999., neo_mean)
    amb_mean = np.where(np.isnan(amb_mean), -9999., amb_mean)
    ele_mean = np.where(np.isnan(ele_mean), -9999., ele_mean)
    R1_mean  = np.where(np.isnan(R1_mean), -9999., R1_mean)
    R2_mean  = np.where(np.isnan(R2_mean), -9999., R2_mean)
    R3_mean  = np.where(np.isnan(R3_mean), -9999., R3_mean)
    R4_mean  = np.where(np.isnan(R4_mean), -9999., R4_mean)
    R5_mean  = np.where(np.isnan(R5_mean), -9999., R5_mean)
    R6_mean  = np.where(np.isnan(R6_mean), -9999., R6_mean)

    print(neo_mean)
    # write data to file
    depth[:] = depths
    time[:]  = dates_org

    print("start")
    print(neo_mean.values.reshape(ndepths, ntimes_org))
    swc_neo = neo_mean.values.reshape(ndepths, ntimes_org)
    swc_amb = amb_mean.values.reshape(ndepths, ntimes_org)
    swc_ele = ele_mean.values.reshape(ndepths, ntimes_org)
    swc_R1  = R1_mean.values.reshape(ndepths, ntimes_org)
    swc_R2  = R2_mean.values.reshape(ndepths, ntimes_org)
#    swc_R3  = R3_mean.values.reshape(ndepths, ntimes_org)
#    swc_R4  = R4_mean.values.reshape(ndepths, ntimes_org)
#    swc_R5  = R5_mean.values.reshape(ndepths, ntimes_org)
#    swc_R6  = R6_mean.values.reshape(ndepths, ntimes_org)
#    print(swc_R6)
    f.close()



if __name__ == "__main__":

    fobs = "/short/w35/mm3972/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    fout = "/short/w35/mm3972/cable/runs/EucFACE/EucFACE_jim_dushan/EucFACE_swc.nc"
    main(fobs, fout)
