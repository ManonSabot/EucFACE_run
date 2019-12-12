#!/usr/bin/env python

"""
Run CABLE for the EucFACE site and gather some specific outputs about direct
and diffuse fracs and direct and diffuse GPP for Jim and Dushan

That's all folks.
"""

'''
modifications:
1. add bulk density to calculate cnsd
2. divide soil/buld density for amb and ele
3. adding Cosby uni Cosby multi HC_SWC
4. add option for single ring
5. add soil fraction interpolation methods
6. add option for soil layers - done
'''

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

def main(met_fname, lai_fname, swc_fname, tdr_fname, stx_fname, out_fname,\
         PTF, soil_frac, layer_num, neo_constrain, tdr_constrain, ring):

    DEG_2_KELVIN = 273.15
    SW_2_PAR = 2.3
    PAR_2_SW = 1.0 / SW_2_PAR
    HLFHR_2_SEC = 1.0 / 1800.

    df = pd.read_csv(met_fname)

    if layer_num == "6":
        nsoil = 6
    elif layer_num == "13":
        nsoil = 13
    elif layer_num in ["31uni","31exp","31para"]:
        nsoil = 31

    ndim = 1
    nsoil= nsoil
    n_timesteps = len(df)
    times = []
    secs = 0.0
    for i in range(n_timesteps):
        times.append(secs)
        secs += 1800.

    # create file and write global attributes
    f = nc.Dataset(out_fname, 'w', format='NETCDF4')
    f.description = 'EucFACE met data, created by MU Mengyuan'
    f.history = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date = "%s" % (datetime.datetime.now())

    # set dimensions
    f.createDimension('time', None)
    f.createDimension('z', ndim)
    f.createDimension('y', ndim)
    f.createDimension('x', ndim)
    f.createDimension('soil_depth', nsoil)
    f.Conventions = "CF-1.0"

    # create variables
    time = f.createVariable('time', 'f8', ('time',))
    time.units = "seconds since 2013-01-01 00:00:00"
    time.long_name = "time"
    time.calendar = "standard"

    z = f.createVariable('z', 'f8', ('z',))
    z.long_name = "z"
    z.long_name = "z dimension"

    y = f.createVariable('y', 'f8', ('y',))
    y.long_name = "y"
    y.long_name = "y dimension"

    x = f.createVariable('x', 'f8', ('x',))
    x.long_name = "x"
    x.long_name = "x dimension"

    soil_depth = f.createVariable('soil_depth', 'f4', ('soil_depth',))
    soil_depth.long_name = "soil_depth"

    latitude = f.createVariable('latitude', 'f4', ('y', 'x',))
    latitude.units = "degrees_north"
    latitude.missing_value = -9999.
    latitude.long_name = "Latitude"

    longitude = f.createVariable('longitude', 'f4', ('y', 'x',))
    longitude.units = "degrees_east"
    longitude.missing_value = -9999.
    longitude.long_name = "Longitude"

    SWdown = f.createVariable('SWdown', 'f4', ('time', 'y', 'x',))
    SWdown.units = "W/m^2"
    SWdown.missing_value = -9999.
    SWdown.long_name = "Surface incident shortwave radiation"
    SWdown.CF_name = "surface_downwelling_shortwave_flux_in_air"

    Tair = f.createVariable('Tair', 'f4', ('time', 'z', 'y', 'x',))
    Tair.units = "K"
    Tair.missing_value = -9999.
    Tair.long_name = "Near surface air temperature"
    Tair.CF_name = "surface_temperature"

    Rainf = f.createVariable('Rainf', 'f4', ('time', 'y', 'x',))
    Rainf.units = "mm/s"
    Rainf.missing_value = -9999.
    Rainf.long_name = "Rainfall rate"
    Rainf.CF_name = "precipitation_flux"

    Qair = f.createVariable('Qair', 'f4', ('time', 'z', 'y', 'x',))
    Qair.units = "kg/kg"
    Qair.missing_value = -9999.
    Qair.long_name = "Near surface specific humidity"
    Qair.CF_name = "surface_specific_humidity"

    Wind = f.createVariable('Wind', 'f4', ('time', 'z', 'y', 'x',))
    Wind.units = "m/s"
    Wind.missing_value = -9999.
    Wind.long_name = "Scalar windspeed" ;
    Wind.CF_name = "wind_speed"

    PSurf = f.createVariable('PSurf', 'f4', ('time', 'y', 'x',))
    PSurf.units = "Pa"
    PSurf.missing_value = -9999.
    PSurf.long_name = "Surface air pressure"
    PSurf.CF_name = "surface_air_pressure"

    LWdown = f.createVariable('LWdown', 'f4', ('time', 'y', 'x',))
    LWdown.units = "W/m^2"
    LWdown.missing_value = -9999.
    LWdown.long_name = "Surface incident longwave radiation"
    LWdown.CF_name = "surface_downwelling_longwave_flux_in_air"

    CO2 = f.createVariable('CO2air', 'f4', ('time', 'z', 'y', 'x',))
    CO2.units = "ppm"
    CO2.missing_value = -9999.
    CO2.long_name = ""
    CO2.CF_name = ""

    LAI = f.createVariable('LAI', 'f4', ('time', 'y', 'x'))
    LAI.setncatts({'long_name': u"Leaf Area Index",})

    vcmax = f.createVariable('vcmax', 'f4', ('y', 'x'))
    ejmax = f.createVariable('ejmax', 'f4', ('y', 'x'))
    g1 = f.createVariable('g1', 'f4', ('y', 'x'))
    hc = f.createVariable('hc', 'f4', ('y', 'x'))

    elevation = f.createVariable('elevation', 'f4', ('y', 'x',))
    elevation.units = "m" ;
    elevation.missing_value = -9999.
    elevation.long_name = "Site elevation above sea level" ;

    za = f.createVariable('za', 'f4', ('y', 'x',))
    za.units = "m"
    za.missing_value = -9999.
    za.long_name = "level of lowest atmospheric model layer"

########## add hydrological parameters ###########
    # 2-Dimension variables
    # slope = 0.004(maximum)
    # sfc   = 0.265 (m3/m3)
    # swilt = 0.115 (m3/m3)
    iveg = f.createVariable('iveg', 'f4', ('y', 'x',))
    iveg.long_name = "vegetation type"
    iveg.units = "-"
    iveg.missing_value = -9999.0

    sand = f.createVariable('sand', 'f4', ('y', 'x',))
    sand.units = "-"
    sand.missing_value = -1.0

    clay = f.createVariable('clay', 'f4', ('y', 'x',))
    clay.units = "-"
    clay.missing_value = -1.0

    silt = f.createVariable('silt', 'f4', ('y', 'x',))
    silt.units = "-"
    silt.missing_value = -1.0

    rhosoil = f.createVariable('rhosoil', 'f4', ('y', 'x',))
    rhosoil.units = "kg m-3"
    rhosoil.long_name = "soil density"
    rhosoil.missing_value = -9999.0

    bch  = f.createVariable('bch', 'f4', ('y', 'x',))
    bch.units = "-"
    bch.long_name = "C and H B"
    bch.missing_value = -9999.0

    hyds = f.createVariable('hyds', 'f4', ('y', 'x',))
    hyds.units = "m s-1"
    hyds.long_name = "hydraulic conductivity at saturation"
    hyds.missing_value = -9999.0

    sucs = f.createVariable('sucs', 'f4', ('y', 'x',))
    sucs.units = "m"
    sucs.long_name = "matric potential at saturation"
    sucs.missing_value = -9999.0

    ssat = f.createVariable('ssat', 'f4', ('y', 'x',))
    ssat.units = "m3 m-3"
    ssat.long_name = "volumetric water content at saturation"
    ssat.missing_value = -9999.0

    swilt= f.createVariable('swilt', 'f4', ('y', 'x',))
    swilt.units = "m3 m-3"
    swilt.long_name = "wilting point"
    swilt.missing_value = -9999.0

    sfc  = f.createVariable('sfc', 'f4', ('y', 'x',))
    sfc.units = "m3 m-3"
    sfc.long_name = "field capcacity"
    sfc.missing_value = -9999.0

    css  = f.createVariable('css', 'f4', ('y', 'x',))
    css.units = "kJ kg-1 K-1"
    css.long_name = "soil specific heat capacity"
    css.missing_value = -9999.0

    cnsd = f.createVariable('cnsd', 'f4', ('y', 'x',))
    cnsd.units = "W m-1 K-1"
    cnsd.long_name = "thermal conductivity of dry soil"
    cnsd.missing_value = -9999.0

    # 3-Dimension variables
    sand_vec = f.createVariable('sand_vec', 'f4', ('soil_depth', 'y', 'x',))
    sand_vec.units = "-"
    sand_vec.missing_value = -1.0

    clay_vec = f.createVariable('clay_vec', 'f4', ('soil_depth', 'y', 'x',))
    clay_vec.units = "-"
    clay_vec.missing_value = -1.0

    silt_vec = f.createVariable('silt_vec', 'f4', ('soil_depth', 'y', 'x',))
    silt_vec.units = "-"
    silt_vec.missing_value = -1.0

    org_vec  = f.createVariable('org_vec', 'f4', ('soil_depth', 'y', 'x',))
    org_vec.units = "-"
    org_vec.missing_value = -1.0

    rhosoil_vec = f.createVariable('rhosoil_vec', 'f4', ('soil_depth', 'y', 'x',))
    rhosoil_vec.units = "kg m-3"
    rhosoil_vec.long_name = "soil density"
    rhosoil_vec.missing_value = -9999.0

    bch_vec  = f.createVariable('bch_vec', 'f4', ('soil_depth', 'y', 'x',))
    bch_vec.units = "-"
    bch_vec.long_name = "C and H B"
    bch_vec.missing_value = -9999.0

    hyds_vec = f.createVariable('hyds_vec', 'f4', ('soil_depth', 'y', 'x',))
    hyds_vec.units = "mm s-1"
    hyds_vec.long_name = "hydraulic conductivity at saturation"
    hyds_vec.missing_value = -9999.0

    sucs_vec = f.createVariable('sucs_vec', 'f4', ('soil_depth', 'y', 'x',))
    sucs_vec.units = "m"
    sucs_vec.long_name = "matric potential at saturation"
    sucs_vec.missing_value = -9999.0

    ssat_vec = f.createVariable('ssat_vec', 'f4', ('soil_depth', 'y', 'x',))
    ssat_vec.units = "m3 m-3"
    ssat_vec.long_name = "volumetric water content at saturation"
    ssat_vec.missing_value = -9999.0

    swilt_vec= f.createVariable('swilt_vec', 'f4', ('soil_depth', 'y', 'x',))
    swilt_vec.units = "m3 m-3"
    swilt_vec.long_name = "wilting point"
    swilt_vec.missing_value = -9999.0

    sfc_vec  = f.createVariable('sfc_vec', 'f4', ('soil_depth', 'y', 'x',))
    sfc_vec.units = "m3 m-3"
    sfc_vec.long_name = "field capcacity"
    sfc_vec.missing_value = -9999.0

    css_vec  = f.createVariable('css_vec', 'f4', ('soil_depth', 'y', 'x',))
    css_vec.units = "kJ kg-1 K-1"
    css_vec.long_name = "soil specific heat capacity"
    css_vec.missing_value = -9999.0

    cnsd_vec = f.createVariable('cnsd_vec', 'f4', ('soil_depth', 'y', 'x',))
    cnsd_vec.units = "W m-1 K-1"
    cnsd_vec.long_name = "thermal conductivity of dry soil"
    cnsd_vec.missing_value = -9999.0

    watr = f.createVariable('watr', 'f4', ('soil_depth', 'y', 'x',))
    watr.units = "m3 m-3"
    watr.long_name = "residual water content of the soil"
    watr.missing_value = -9999.0

    SoilMoist = f.createVariable('SoilMoist', 'f4', ('soil_depth', 'y', 'x',))
    SoilMoist.units = "m3 m-3"
    SoilMoist.long_name = "soil moisture (water+ice)"
    SoilMoist.missing_value = -9999.0

    """
    Parameters based on param_vec:

            2D: sand, silt, clay, organic, sucs, ssat, rhosoil, bch, hyds,
                cnsd (thermal conductivity of dry soil, W/m/K), css (soil specific heat capacity, kJ/kg/K)

    Aquifer Parameters copied from bottom soil layer:
            "Sy", "permeability"

    Parameters read from gridinfo file:

           2D:  "slope_std", "Albedo", "albedo2", "elevation_std", "drainage_density",
                "drainage_dist",  "permeability_std",
                "dtb" (is there an observation)
                "soil_color" (relate to soil type?)
           3D:  "SnowDepth", "patchfrac",
           4D:  "SoilTemp"
    """

    # write data to file
    x[:] = ndim
    y[:] = ndim
    z[:] = ndim

    soil_depth[:] = np.arange(0,nsoil,1)
    time[:] = times
    latitude[:]  = -33.617778 # Ellsworth 2017, NCC
    longitude[:] = 150.740278 # Ellsworth 2017, NCC

    SWdown[:,0,0] = (df.PAR.values * PAR_2_SW).reshape(n_timesteps, ndim, ndim)
    Tair[:,0,0,0] = (df.TAIR.values + DEG_2_KELVIN).reshape(n_timesteps,
                                                            ndim, ndim, ndim)
    df.PPT *= HLFHR_2_SEC
    Rainf[:,0,0] = df.PPT.values.reshape(n_timesteps, ndim, ndim)
    qa_vals = convert_rh_to_qair(df.RH.values, df.TAIR.values, df.PRESS.values)
    Qair[:,0,0,0] = qa_vals.reshape(n_timesteps, ndim, ndim, ndim)
    Wind[:,0,0,0] = df.WIND.values.reshape(n_timesteps, ndim, ndim, ndim)
    PSurf[:,0,0] = df.PRESS.values.reshape(n_timesteps, ndim, ndim)
    lw = estimate_lwdown(df.TAIR.values + DEG_2_KELVIN, df.RH.values)
    LWdown[:,0,0] = lw.reshape(n_timesteps, ndim, ndim)
    if ring in ["amb","R2","R3","R6"]:
        CO2[:,0,0] = df["Ca.A"].values.reshape(n_timesteps, ndim, ndim, ndim)
        vcmax[:] = 86.1425919e-6
        ejmax[:] = 138.4595736e-6
    elif ring in ["ele","R1","R4","R5"]:
        CO2[:,0,0] = df["Ca.E"].values.reshape(n_timesteps, ndim, ndim, ndim)
        vcmax[:] = 81.70591263e-6
        ejmax[:] = 135.8062907e-6
    elevation[:] = 23.0 # Ellsworth 2017, NCC
    LAI[:,0,0] = interpolate_lai(lai_fname, ring)

    #df.lai.values.reshape(n_timesteps, ndim, ndim)
    g1[:] = 3.8
    hc[:] = 20.
    za[:] = 20.0 + 2.0 # ???

# read hydraulic parameters:
    iveg[:]       = 2
    if layer_num == "6":
        soil_depth[:] = [0.011, 0.051, 0.157, 0.4385, 1.1855, 3.164]
        zse_vec       = [0.022, 0.058, 0.154, 0.409, 1.085, 2.872]
        boundary      = [0, 2.2, 8., 23.4, 64.3, 172.8, 460.]
        org_vec[:,0,0]= [0.0102,0.0102,0.0025,0.0025, 0.0025,0.0025]
    elif layer_num == "13":
        soil_depth[:] = [0.01,0.045,0.10,0.195,0.41,0.71,1.01,1.31,1.61,1.91,2.21,2.735,3.86]
        zse_vec       = [0.02, 0.05, 0.06, 0.13, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.75, 1.50]
        boundary      = [0, 2., 7., 13., 26., 56., 86., 116., 146., 176., 206., 236., 311., 461.]
        org_vec[:,0,0]= [0.0102,0.0102,0.0102,0.0025,0.0025,0.0025,0.0025,0.0025,\
                         0.0025,0.0025,0.0025,0.0025,0.0025]
    elif layer_num == "31uni":
        soil_depth[:] = [ 0.075, 0.225 , 0.375 , 0.525 , 0.675 , 0.825 , 0.975 , \
                          1.125, 1.275, 1.425, 1.575, 1.725, 1.875, 2.025, \
                          2.175, 2.325, 2.475, 2.625, 2.775, 2.925, 3.075, \
                          3.225, 3.375, 3.525, 3.675, 3.825, 3.975, 4.125, \
                          4.275, 4.425, 4.575 ]
        zse_vec       = [ 0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                          0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                          0.15,  0.15,  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  \
                          0.15 ]
        boundary      = [   0.,  15.,  30.,  45.,  60.,  75.,  90., 105., 120., 135., 150., \
                          165., 180., 195., 210., 225., 240., 255., 270., 285., 300., 315., \
                          330., 345., 360., 375., 390., 405., 420., 435., 450., 465.]
        org_vec[:,0,0]= [ 0.0102, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025]
    elif layer_num == "31exp":
        soil_depth[:] = [ 0.01021985, 0.02131912, 0.02417723, 0.02967358, 0.03868759, 0.05209868,\
                          0.07078627, 0.09562978, 0.1275086, 0.1673022, 0.2158899, 0.2741512,\
                          0.3429655, 0.4232122, 0.5157708, 0.6215205, 0.741341 , 0.8761115,\
                          1.026711, 1.19402 , 1.378918, 1.582283, 1.804995, 2.047933,\
                          2.311978, 2.598008, 2.906903, 3.239542, 3.596805, 3.979571,\
                          4.388719 ]
        zse_vec       = [ 0.020440, 0.001759, 0.003957, 0.007035, 0.010993, 0.015829,\
                        0.021546, 0.028141, 0.035616, 0.043971, 0.053205, 0.063318,\
                        0.074311, 0.086183, 0.098934, 0.112565, 0.127076, 0.142465,\
                        0.158735, 0.175883, 0.193911, 0.212819, 0.232606, 0.253272,\
                        0.274818, 0.297243, 0.320547, 0.344731, 0.369794, 0.395737,\
                        0.422559 ]
        boundary      = [      0.,    2.044,   2.2199,   2.6156,   3.3191,   4.4184,   6.0013,   8.1559,\
                           10.97,  14.5316,  18.9287,  24.2492,   30.581,  38.0121,  46.6304,  56.5238,\
                         67.7803,  80.4879,  94.7344, 110.6079, 128.1962, 147.5873, 168.8692, 192.1298,\
                         217.457, 244.9388, 274.6631, 306.7178, 341.1909, 378.1703, 417.744 , 460.    ]
        org_vec[:,0,0]= [ 0.0102, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025]
    elif layer_num == "31para":
        soil_depth[:] = [ 0.01000014,  0.0347101, 0.07782496, 0.1473158, 0.2411537, 0.3573098, \
                          0.4937551, 0.6484607, 0.8193976, 1.004537, 1.20185 , 1.409308, \
                          1.624881, 1.846541, 2.072259, 2.30    , 2.527742, 2.75346 , \
                          2.97512 , 3.190693, 3.398151, 3.595464, 3.780603, 3.95154 , \
                          4.106246, 4.242691, 4.358847, 4.452685, 4.522176, 4.565291, \
                          4.590001 ]
        zse_vec       = [ 0.020000, 0.029420, 0.056810, 0.082172, 0.105504, 0.126808,\
                        0.146083, 0.163328, 0.178545, 0.191733, 0.202892, 0.212023,\
                        0.219124, 0.224196, 0.227240, 0.228244, 0.227240, 0.224196,\
                        0.219124, 0.212023, 0.202892, 0.191733, 0.178545, 0.163328,\
                        0.146083, 0.126808, 0.105504, 0.082172, 0.056810, 0.029420,\
                        0.020000 ]
        boundary      =[     0.,      2.,     4.942,    10.623,   18.8402,   29.3906,  42.0714,\
                        56.6797,  73.0125,   90.867,  110.0403,  130.3295,  151.5318, 173.4442,\
                        195.8638, 218.5878, 241.4122,  264.1362,  286.5558,  308.4682, 329.6705,\
                        349.9597,  369.133, 386.9875,  403.3203,  417.9286,  430.6094, 441.1598,\
                        449.377,   455.058,     458.,      460. ]
        org_vec[:,0,0]= [ 0.0102, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025,\
                          0.0025]

    sand_vec[:,0,0]   = np.zeros(nsoil)
    silt_vec[:,0,0]   = np.zeros(nsoil)
    clay_vec[:,0,0]   = np.zeros(nsoil)
    rhosoil_vec[:,0,0]= np.zeros(nsoil)
    hyds_vec[:,0,0]   = np.zeros(nsoil)
    bch_vec[:,0,0]    = np.zeros(nsoil)
    sucs_vec[:,0,0]   = np.zeros(nsoil)
    ssat_vec[:,0,0]   = np.zeros(nsoil)
    watr[:,0,0]       = np.zeros(nsoil)
    swilt_vec[:,0,0]  = np.zeros(nsoil)
    sfc_vec[:,0,0]    = np.zeros(nsoil)
    css_vec[:,0,0]    = np.zeros(nsoil)
    cnsd_vec[:,0,0]   = np.zeros(nsoil)
    SoilMoist[:,0,0]  = np.zeros(nsoil)

    bulk_density      = np.zeros(nsoil)
    sand_vec[:,0,0],silt_vec[:,0,0],clay_vec[:,0,0] = calc_soil_frac(stx_fname, ring, nsoil, boundary, soil_frac)
    rhosoil_vec[:,0,0],bulk_density = estimate_rhosoil_vec(swc_fname, nsoil, ring, soil_frac, boundary)
    SoilMoist[:,0,0]  = init_soil_moisture(swc_fname, nsoil, ring, soil_frac, boundary)


    psi_tmp  = 2550000.0 # water potential at wilting point for iveg=2 in CABLE
    print("good")
    if PTF == 'Campbell_Cosby_multi_Python':
        for i in np.arange(0,nsoil,1):
            css_vec[i]  = (1.0-org_vec[i]) * ( 850*(1.0 - sand_vec[i] - clay_vec[i]) + \
                            865.0*clay_vec[i] + 750.0*sand_vec[i] ) + org_vec[i]*950.0
            hyds_vec[i] = (1.0-org_vec[i]) * 0.00706 * ( 10.0 ** (-0.60 + 1.26*sand_vec[i] - 0.64*clay_vec[i]) )\
                          + org_vec[i]*10**(-4)
            bch_vec[i]  = (1.0-org_vec[i]) * ( 3.1 + 15.7*clay_vec[i] - 0.3*sand_vec[i]) + org_vec[i]*3.0
            watr[i]     = (1.0-org_vec[i]) * ( 0.02 + 0.018*clay_vec[i] ) + org_vec[i]*0.15
            ssat_vec[i] = (1.0-org_vec[i]) * ( 0.505 - 0.142*sand_vec[i] - 0.037*clay_vec[i]) \
                          + org_vec[i]*0.6
            sfc_vec[i]   = (ssat_vec[i] - watr[i]) * ( 1.157407 * 10**(-6) / hyds_vec[i])** \
                          (1.0 / (2.0*bch_vec[i] + 3.0) ) + watr[i]
            sst_tmp  = 1.0 - max(min(ssat_vec[i], 0.85), 0.15)
            cnsd_vec[i]  = (1.0-org_vec[i]) * ( 0.135*sst_tmp + 0.0239/sst_tmp )  /  \
                            (1.0 - 0.947*sst_tmp) + org_vec[i]*0.05
            sucs_vec[i] = (1.0-org_vec[i]) * 10.0 * 10.0**( 1.54 - 0.95*sand_vec[i] + 0.63*silt_vec[i] ) \
                            + org_vec[i]*10.3
            swilt_vec[i] = (ssat_vec[i] - watr[i]) * ( (psi_tmp/sucs_vec[i]) ** (-1.0/bch_vec[i]) ) \
                           + watr[i]
            sucs_vec[i] = sucs_vec[i]/1000. # cannot put it before swilt_vec calculation, it will cause error
                                            # comment out *(-1.0) then waterbal can be closed, because CABLE expect the positive sucs_vec input,
                                            # for global sucs_vec in cable is soil%sucs_vec  = 1000._r_2 * ( abs('sucs_vec') / abs(insucs) ),
                                            # thus the negetive value in surface forcing data is transfer into positve value...
    else:
        if PTF == 'Campbell_Cosby_univariate':
            for i in np.arange(0,nsoil,1):
                hyds_vec[i] = (1.0-org_vec[i]) * ( 0.0070556 * 10.0**( -0.884 + 1.53*sand_vec[i] )) \
                              + org_vec[i]*10**(-4)
                sucs_vec[i] = (1.0-org_vec[i]) * ( 10.0 * 10.0**( 1.88 -1.31*sand_vec[i] ))\
                              + org_vec[i]*10.3
                bch_vec[i]  = (1.0-org_vec[i]) * ( 2.91 + 15.9*clay_vec[i] ) \
                              + org_vec[i]*2.91
                ssat_vec[i] = (1.0-org_vec[i]) * ( min( 0.489, max( 0.1, 0.489 - 0.126*sand_vec[i] ))) \
                              + org_vec[i]*0.9
                watr[i]     = (1.0-org_vec[i]) * ( 0.02 + 0.018*clay_vec[i] ) \
                              + org_vec[i]*0.1
        elif PTF == 'Campbell_Cosby_multivariate':
            for i in np.arange(0,nsoil,1):
                hyds_vec[i] = (1.0-org_vec[i]) * ( 0.00706*(10.0**(-0.60 + 1.26*sand_vec[i] - 0.64*clay_vec[i] ))) \
                              + org_vec[i]*10**(-4)
                sucs_vec[i] = (1.0-org_vec[i]) * ( 10.0 * 10.0**(1.54 - 0.95*sand_vec[i] + 0.63*silt_vec[i] )) \
                              + org_vec[i]*10.3
                bch_vec[i]  = (1.0-org_vec[i]) * ( 3.1 + 15.7 *clay_vec[i] - 0.3*sand_vec[i] ) \
                              + org_vec[i]*2.91
                # 15.4 -> 15.7, because in Cosby 1980 is 15.7
                ssat_vec[i] = (1.0-org_vec[i]) * ( 0.505 - 0.142*sand_vec[i] - 0.037*clay_vec[i] ) \
                              + org_vec[i]*0.9
                watr[i]     = (1.0-org_vec[i]) * ( 0.02 + 0.018*clay_vec[i] ) \
                              + org_vec[i]*0.1
        elif PTF == 'Campbell_HC_SWC':
            for i in np.arange(0,nsoil,1):
                sucs_vec[i] = 10.0*10.0**( -4.9840 + 5.0923*sand_vec[i] + 15.752*silt_vec[i]\
                              + 0.12409*bulk_density[i] - 16.400*org_vec[i] - 21.767*(silt_vec[i]**2.0)\
                              + 14.382*(silt_vec[i]**3.0) + 8.0407*(clay_vec[i]**2.0) + 44.067*(org_vec[i]**2.0) )
                print("we did sucs")
                bch_vec[i]  = 10.0**(0.84669 + 0.46806*sand_vec[i] - 0.92464*silt_vec[i] \
                              + 0.45428*bulk_density[i] +4.9792*org_vec[i] - 3.2947*(sand_vec[i]**2.0)\
                              + 1.6891*(sand_vec[i]**3.0) - 11.225*(org_vec[i]**3.0) )
                print("we did bch")
                ssat_vec[i] = 0.23460 + 0.46614*sand_vec[i] + 0.88163*silt_vec[i] \
                              + 0.64339*clay_vec[i] - 0.30282*bulk_density[i] \
                              + 0.17976*(sand_vec[i]**2.0) - 0.31346*(silt_vec[i]**2.0)
                print("we did ssat")
                hyds_vec[i] = (1.0-org_vec[i]) * ( 0.00706*(10.0**(-0.60 + 1.26*sand_vec[i] - 0.64*clay_vec[i] ))) \
                              + org_vec[i]*10**(-4)
                print("we did hyds")
                # In CABLE for HC_SWC is hyds_vec[i] = 0.00706*(10.0**(-0.60 + 1.26*sand_vec[i]+ -0.64*clay_vec[i] ))
                watr[i]     = 0.0
        print("well done")
        for i in np.arange(0,nsoil,1):
            swilt_vec[i] = (ssat_vec[i]-watr[i]) * (psi_tmp/sucs_vec[i])**(-1.0/bch_vec[i])+watr[i]
            sfc_vec[i]   = (1.157407 * 10**(-6) / hyds_vec[i])**(1.0/(2.0*bch_vec[i]+3.0)) \
                           *(ssat_vec[i]-watr[i]) + watr[i]
            swilt_vec[i] = min(0.95*sfc_vec[i],swilt_vec[i])
            ssat_bounded = min(0.8, max(0.1, ssat_vec[i] ))
            cnsd_vec[i]  = (1.0-org_vec[i]) * (( 0.135*(1.0-ssat_bounded)) + (64.7/rhosoil_vec[i]))\
                           / (1.0 - 0.947*(1.0-ssat_bounded)) + org_vec[i]*0.1
            css_vec[i]   = (1.0-org_vec[i]) * max( 910.6479*silt_vec[i] + 916.4438 * clay_vec[i] + 740.7491*sand_vec[i], 800.0)\
                           + org_vec[i]*4000.0
            sucs_vec[i] = sucs_vec[i]/1000.
    print("all are good")

    if neo_constrain:
        swilt_vec[:,0,0],watr[:],ssat_vec[:,0,0]  = \
            neo_swilt_ssat(swc_fname, nsoil, ring, layer_num, swilt_vec[:,0,0], watr[:], ssat_vec[:,0,0], soil_frac, boundary)
    if tdr_constrain:
        swilt_vec[:,0,0],ssat_vec[:,0,0] = tdr_constrain_top_50cm(tdr_fname, ring, layer_num, swilt_vec[:,0,0],ssat_vec[:0,0])

    if swilt_neo or tdr_constrain:
        if PTF == 'Campbell_Cosby_multi_Python':
            for i in np.arange(0,nsoil,1):
                sfc_vec[i]   = (ssat_vec[i] - watr[i]) * ( 1.157407 * 10**(-6) / hyds_vec[i])** \
                              (1.0 / (2.0*bch_vec[i] + 3.0) ) + watr[i]
                sst_tmp  = 1.0 - max(min(ssat_vec[i], 0.85), 0.15)
                cnsd_vec[i]  = (1.0-org_vec[i]) * ( 0.135*sst_tmp + 0.0239/sst_tmp )  /  \
                                (1.0 - 0.947*sst_tmp) + org_vec[i]*0.05
        else:
            for i in np.arange(0,nsoil,1):
                sfc_vec[i]   = (1.157407 * 10**(-6) / hyds_vec[i])**(1.0/(2.0*bch_vec[i]+3.0)) \
                               *(ssat_vec[i]-watr[i]) + watr[i]
                ssat_bounded = min(0.8, max(0.1, ssat_vec[i] ))
                cnsd_vec[i]  = (1.0-org_vec[i]) * (( 0.135*(1.0-ssat_bounded)) + (64.7/rhosoil_vec[i]))\
                               / (1.0 - 0.947*(1.0-ssat_bounded)) + org_vec[i]*0.1

    sand[:,0] = thickness_weighted_average(sand_vec, nsoil, zse_vec)
    silt[:,0] = thickness_weighted_average(silt_vec, nsoil, zse_vec)
    clay[:,0] = thickness_weighted_average(clay_vec, nsoil, zse_vec)
    rhosoil[:,0] = thickness_weighted_average(rhosoil_vec, nsoil, zse_vec)
    css[:,0] = thickness_weighted_average(css_vec, nsoil, zse_vec)
    hyds[:,0] = thickness_weighted_average(hyds_vec, nsoil, zse_vec)
    hyds[:,0] = hyds[:,0]/1000.
    bch[:,0]  = thickness_weighted_average(bch_vec, nsoil, zse_vec)
    ssat[:,0] = thickness_weighted_average(ssat_vec, nsoil, zse_vec)
    sfc[:,0] = thickness_weighted_average(sfc_vec, nsoil, zse_vec)
    cnsd[:,0] = thickness_weighted_average(cnsd_vec, nsoil, zse_vec)
    sucs[:,0] = thickness_weighted_average(sucs_vec, nsoil, zse_vec)
    swilt[:,0] = thickness_weighted_average(swilt_vec, nsoil, zse_vec)

    f.close()

def calc_soil_frac(stx_fname, ring, nsoil, boundary, soil_frac):
    """
    interpolate the observated sand,silt,clay to pointed depth
    """
    soil_texture = pd.read_csv(stx_fname, usecols = ['Ring','Depth_interval_cm','Sand_%','Silt_%','Clay_%'])
    soil_texture['Depth'] = np.zeros(len(soil_texture))
    soil_texture['Depth_top'] = np.zeros(len(soil_texture))
    soil_texture['Depth_bot'] = np.zeros(len(soil_texture))

    for i in np.arange(0,len(soil_texture),1):
        index = soil_texture['Depth_interval_cm'].values[i].index('-')
        soil_texture['Depth'].iloc[i] = (float(soil_texture['Depth_interval_cm'].values[i][:index]) \
                                        + float(soil_texture['Depth_interval_cm'].values[i][index+1:]))/2.
        soil_texture['Depth_top'].iloc[i] = float(soil_texture['Depth_interval_cm'].values[i][:index])
        soil_texture['Depth_bot'].iloc[i] = float(soil_texture['Depth_interval_cm'].values[i][index+1:])

    if ring == 'amb':
        subset = soil_texture[soil_texture['Ring'].isin(['R2','R3','R6'])]
    elif ring == 'ele':
        subset = soil_texture[soil_texture['Ring'].isin(['R1','R4','R5'])]
    else:
        subset = soil_texture[soil_texture['Ring'].isin([ring])]
    print(subset)
    grid_value = np.arange(0.5,465,1)
    subset = subset.groupby(by=["Depth"]).mean()
    Sand_calu = interp1d(subset.index, subset['Sand_%'].values, kind = soil_frac, \
                fill_value=(subset['Sand_%'].values[0],subset['Sand_%'].values[-1]), \
                bounds_error=False)
    sand_grid = Sand_calu(grid_value)/100.

    Silt_calu = interp1d(subset.index, subset['Silt_%'].values, kind = soil_frac, \
                fill_value=(subset['Silt_%'].values[0],subset['Silt_%'].values[-1]), \
                bounds_error=False)
    silt_grid = Silt_calu(grid_value)/100.

    Clay_calu = interp1d(subset.index, subset['Clay_%'].values, kind = soil_frac, \
                fill_value=(subset['Clay_%'].values[0],subset['Clay_%'].values[-1]), \
                bounds_error=False)
    clay_grid = Clay_calu(grid_value)/100.

    # check for the data range
    if ring in ['R1','R2','R3','R4','R5']:
        for j in np.arange(0,len(subset),1):
            for i in np.arange(0,len(grid_value),1):
                if ((grid_value[i] >= subset['Depth_top'].values[j]) and (grid_value[i] <= subset['Depth_bot'].values[j])):
                    sand_grid[i] = subset['Sand_%'].values[j]/100.
                    silt_grid[i] = subset['Silt_%'].values[j]/100.
                    clay_grid[i] = subset['Clay_%'].values[j]/100.
    sand_calu = np.zeros(nsoil)
    silt_calu = np.zeros(nsoil)
    clay_calu = np.zeros(nsoil)

    # average
    for j in np.arange(0,nsoil,1):
        if (j == 0 and grid_value[0] > boundary[1]):
            sand_calu[0] = sand_grid[0]
            silt_calu[0] = silt_grid[0]
            clay_calu[0] = clay_grid[0]
        else:
            counter = 0.
            for i in np.arange(0,len(grid_value),1):
                if (grid_value[i] >= boundary[j]) and (grid_value[i] <= boundary[j+1]):
                    sand_calu[j] = sand_calu[j] + sand_grid[i]
                    silt_calu[j] = silt_calu[j] + silt_grid[i]
                    clay_calu[j] = clay_calu[j] + clay_grid[i]
                    counter += 1.
            sand_calu[j] = sand_calu[j]/counter
            silt_calu[j] = silt_calu[j]/counter
            clay_calu[j] = clay_calu[j]/counter
        if abs(sand_calu[j] + silt_calu[j] + clay_calu[j] -1.0) > 0.0001:
            print("(sand_calu[j] + silt_calu[j] + clay_calu[j] -1.0) > 0.0001")
            print(ring)
            print("j is")
            print(j)
            print(sand_calu[j])
            print(silt_calu[j])
            print(clay_calu[j])
            sand_calu[j] = 1. - silt_calu[j] - clay_calu[j]

    return sand_calu, silt_calu, clay_calu;

def estimate_rhosoil_vec(swc_fname, nsoil, ring, soil_frac, boundary):
    """
    get Bulk.den
    """
    neo = pd.read_csv(swc_fname, usecols = ['Ring','Depth','Date','Bulk.den'])
    if ring == 'amb':
        subset = neo[neo['Ring'].isin(['R2','R3','R6'])]
    elif ring == 'ele':
        subset = neo[neo['Ring'].isin(['R1','R4','R5'])]
    else:
        subset = neo[neo['Ring'].isin([ring])]
    bulk_den = subset.groupby(by=['Depth']).mean()['Bulk.den']
    f = interp1d(bulk_den.index, bulk_den.values, kind = soil_frac, \
             fill_value=(bulk_den.values[0],bulk_den.values[-1]), bounds_error=False) # fill_value='extrapolate'
    grid_value = np.arange(0.5,465,1)
    Bulk_grid  = f(grid_value)
    Bulk_den_g_cm = np.zeros(nsoil)

    for j in np.arange(0,nsoil,1):
        if (j == 0 and grid_value[0] > boundary[1]):
            Bulk_den_g_cm[0] = Bulk_grid[0]
        else:
            counter = 0.
            for i in np.arange(0,len(grid_value),1):
                if (grid_value[i] >= boundary[j]) and (grid_value[i] <= boundary[j+1]):
                    Bulk_den_g_cm[j] = Bulk_den_g_cm[j] + Bulk_grid[i]
                    counter += 1.
            Bulk_den_g_cm[j] = Bulk_den_g_cm[j]/counter # units: g cm-3
    print(Bulk_den_g_cm)
    Bulk_den_kg_m = Bulk_den_g_cm*1000. # units: kg m-3
    return Bulk_den_kg_m, Bulk_den_g_cm

def init_soil_moisture(swc_fname, nsoil, ring, soil_frac, boundary):

    neo = pd.read_csv(swc_fname, usecols = ['Ring','Depth','Date','VWC'])
    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)
    neo['Date'] = neo['Date'] - pd.datetime(2012,12,31)
    neo['Date'] = neo['Date'].dt.days
    neo = neo.sort_values(by=['Date','Depth'])

    if ring == 'amb':
        subset = neo[neo['Ring'].isin(['R2','R3','R6'])]
    elif ring == 'ele':
        subset = neo[neo['Ring'].isin(['R1','R4','R5'])]
    else:
        subset = neo[neo['Ring'].isin([ring])]

    neo_mean = subset.groupby(by=["Depth","Date"]).mean()
    neo_mean = neo_mean.xs('VWC', axis=1, drop_level=True)
    date_start = pd.datetime(2012,12,1) - pd.datetime(2012,12,31)
    date_end   = pd.datetime(2013,1,31) - pd.datetime(2012,12,31)
    date_start = date_start.days
    date_end   = date_end.days
    x     = np.concatenate((neo_mean[(25)].index.values,               \
                            neo_mean.index.get_level_values(1).values, \
                            neo_mean[(450)].index.values ))              # time
    y     = np.concatenate(([0]*len(neo_mean[(25)]),                   \
                            neo_mean.index.get_level_values(0).values, \
                           [460]*len(neo_mean[(25)])    ))
    value = np.concatenate((neo_mean[(25)].values, neo_mean.values, neo_mean[(450)].values))
    X     = np.arange(date_start,date_end,1) # 2012-4-30 to 2019-5-11
    Y     = np.arange(0.5,465,1)
    grid_X, grid_Y = np.meshgrid(X,Y)

    # interpolate
    grid_data = griddata((x, y) , value, (grid_X, grid_Y), method=soil_frac)
    SoilMoist_grid = grid_data[:,30]/100.

    SoilM     = np.zeros(nsoil)
    grid_value = np.arange(0.5,465,1)
    for j in np.arange(0,nsoil,1):
        if (j == 0 and grid_value[0] > boundary[1]):
            SoilM[0] = SoilMoist_grid[0]
        else:
            counter = 0.
            for i in np.arange(0,len(grid_value),1):
                if ((grid_value[i] >= boundary[j]) and (grid_value[i] <= boundary[j+1])):
                    SoilM[j] = SoilM[j] + SoilMoist_grid[i]
                    counter += 1.
            SoilM[j] = SoilM[j]/counter # units: g cm-3
    print(SoilM)
    return SoilM

def neo_swilt_ssat(swc_fname, nsoil, ring, layer_num, swilt_input, watr_input, ssat_input, soil_frac, boundary):

    neo = pd.read_csv(swc_fname, usecols = ['Ring','Depth','VWC'])
    neo = neo.sort_values(by=['Depth'])

    if ring == 'amb':
        subset = neo[neo['Ring'].isin(['R2','R3','R6'])]
    elif ring == 'ele':
        subset = neo[neo['Ring'].isin(['R1','R4','R5'])]
    else:
        subset = neo[neo['Ring'].isin([ring])]
    subset['VWC'] = subset['VWC'].clip(lower=0.)
    subset['VWC'] = subset['VWC'].replace(0., float('nan'))

    neo_min = np.zeros(12)
    neo_min[0] = subset[subset['Depth'] == 25]['VWC'].nsmallest(5).mean()/100.
    neo_min[1] = subset[subset['Depth'] == 50]['VWC'].nsmallest(5).mean()/100.
    neo_min[2] = subset[subset['Depth'] == 75]['VWC'].nsmallest(5).mean()/100.
    neo_min[3] = subset[subset['Depth'] == 100]['VWC'].nsmallest(5).mean()/100.
    neo_min[4] = subset[subset['Depth'] == 125]['VWC'].nsmallest(5).mean()/100.
    neo_min[5] = subset[subset['Depth'] == 150]['VWC'].nsmallest(5).mean()/100.
    neo_min[6] = subset[subset['Depth'] == 200]['VWC'].nsmallest(5).mean()/100.
    neo_min[7] = subset[subset['Depth'] == 250]['VWC'].nsmallest(5).mean()/100.
    neo_min[8] = subset[subset['Depth'] == 300]['VWC'].nsmallest(5).mean()/100.
    neo_min[9] = subset[subset['Depth'] == 350]['VWC'].nsmallest(5).mean()/100.
    neo_min[10] = subset[subset['Depth'] == 400]['VWC'].nsmallest(5).mean()/100.
    neo_min[11] = subset[subset['Depth'] == 450]['VWC'].nsmallest(5).mean()/100.
    print(neo_min)
    neo_index = [ 25.,  50.,  75., 100., 125., 150., 200., 250.,\
                     300., 350., 400., 450. ]
    f = interp1d(neo_index, neo_min, kind = soil_frac, \
             fill_value=(neo_min[0],neo_min[-1]), bounds_error=False) # fill_value='extrapolate'

    grid_value = np.arange(0.5,465,1)
    swilt_neo  = f(grid_value)

    swilt_output = np.zeros(nsoil)
    watr_output  = np.zeros(nsoil)

    for j in np.arange(0,nsoil,1):
        if (j == 0 and grid_value[0] > boundary[1]):
            swilt_output[0] = min(swilt_neo[0],swilt_input[0])
            if swilt_output[0] <= watr_input[0]:
                watr_output[0] = swilt_output[0]
                swilt_output[0]= watr_output[0]+0.0001
            else:
                watr_output[0] = watr_input[0]
        else:
            counter = 0.
            for i in np.arange(0,len(grid_value),1):
                if ((grid_value[i] >= boundary[j]) and (grid_value[i] <= boundary[j+1])):
                    swilt_output[j] = swilt_output[j] + swilt_neo[i]
                    counter += 1.
            swilt_output[j] = swilt_output[j]/counter
            swilt_output[j] = min(swilt_output[j],swilt_input[j])
            if swilt_output[j] <= watr_input[j]:
                watr_output[j] = swilt_output[j]
                swilt_output[j]= watr_output[j]+0.0001
            else:
                watr_output[j] = watr_input[j]

    neo_max = np.zeros(12)
    neo_max[0] = subset[subset['Depth'] == 25]['VWC'].nlargest(1)/100.
    neo_max[1] = subset[subset['Depth'] == 50]['VWC'].nlargest(1)/100.
    neo_max[2] = subset[subset['Depth'] == 75]['VWC'].nlargest(1)/100.
    neo_max[3] = subset[subset['Depth'] == 100]['VWC'].nlargest(1)/100.
    neo_max[4] = subset[subset['Depth'] == 125]['VWC'].nlargest(1)/100.
    neo_max[5] = subset[subset['Depth'] == 150]['VWC'].nlargest(1)/100.
    neo_max[6] = subset[subset['Depth'] == 200]['VWC'].nlargest(1)/100.
    neo_max[7] = subset[subset['Depth'] == 250]['VWC'].nlargest(1)/100.
    neo_max[8] = subset[subset['Depth'] == 300]['VWC'].nlargest(1)/100.
    neo_max[9] = subset[subset['Depth'] == 350]['VWC'].nlargest(1)/100.
    neo_max[10] = subset[subset['Depth'] == 400]['VWC'].nlargest(1)/100.
    neo_max[11] = subset[subset['Depth'] == 450]['VWC'].nlargest(1)/100.
    print(neo_max)

    g = interp1d(neo_index, neo_max, kind = soil_frac, \
             fill_value=(neo_max[0],neo_max[-1]), bounds_error=False) # fill_value='extrapolate'
    ssat_output = g(grid_value)

    # assumption: neo_max cannot capture ssat when depth > 1m
    if layer_num == "6":
        layer_num_1m = 4 # [0-64.3cm]
    elif layer_num == "13":
        layer_num_1m = 7 # [0-116cm]
    elif layer_num == "31uni":
        layer_num_1m = 3 # [0-105cm]
    elif layer_num == "31exp":
        layer_num_1m = 19 # [0-110.6079cm]
    elif layer_num == "31para":
        layer_num_1m = 10 # [0-110.0403cm]

    for i in np.arange(layer_num_1m, len(ssat_input)):
        if ssat_output[i] < ssat_input[i]:
            ssat_output[i] = ssat_input[i]

    for j in np.arange(0,nsoil,1):
        if (abs(ssat_input[j]-ssat_output[j]) > 0.03):
            print("*********************************")
            print("the difference between calculated and observated ssats in the %s layer is larger than 0.03" % str(j))
            print("the calculated is %s and the observated is %s", % ( str(ssat_input[j]), str(ssat_output[j])))
            print("*********************************")

    return swilt_output, watr_output, ssat_output;

def tdr_constrain_top_50cm(tdr_fname, ring, layer_num, swilt_input, ssat_input):

    tdr = pd.read_csv(tdr_fname, usecols = ['Ring','swc.tdr'])

    # divide neo into groups
    if ring == 'amb':
        subset = tdr[(tdr['Ring'].isin(['R2','R3','R6']))]
    elif ring == 'ele':
        subset = tdr[(tdr['Ring'].isin(['R1','R4','R5']))]
    else:
        subset = tdr[(tdr['Ring'].isin([ring]))]

    tdr_min = subset['swc.tdr'].nsmallest(1)/100.
    tdr_max = subset['swc.tdr'].nlargest(1)/100.


    # assumption: tdr variance can capture swilt and ssat better within 50 cm
    if layer_num == "6":
        layer_num_50cm = 4 # [0-64.3cm]
    elif layer_num == "13":
        layer_num_50cm = 5 # [0-56cm]
    elif layer_num == "31uni":
        layer_num_50cm = 3 # [0-45cm]
    elif layer_num == "31exp":
        layer_num_50cm = 14 # [0-46.63cm]
    elif layer_num == "31para":
        layer_num_50cm = 6 # [0-42.0714cm]

    swilt_output = swilt_input
    ssat_output  = ssat_input

    for i in np.arange(0,layer_num_50cm):
        swilt_output[i] = tdr_min
        ssat_output[i]  = tdr_max
    return swilt_output,ssat_output;

def convert_rh_to_qair(rh, tair, press):
    """
    Converts relative humidity to specific humidity (kg/kg)

    Params:
    -------
    tair : float
        deg C
    press : float
        pa
    rh : float
        %
    """

    # Sat vapour pressure in Pa
    esat = calc_esat(tair)

    # Specific humidity at saturation:
    ws = 0.622 * esat / (press - esat)

    # specific humidity
    qair = rh * ws

    return qair

def calc_esat(tair):
    """
    Calculates saturation vapour pressure

    Params:
    -------
    tair : float
        deg C

    Reference:
    ----------
    * Jones (1992) Plants and microclimate: A quantitative approach to
    environmental plant physiology, p110
    """

    esat = 613.75 * np.exp(17.502 * tair / (240.97 + tair))

    return esat


def estimate_lwdown(tairK, rh):
    """
    Synthesises downward longwave radiation based on Tair RH

    Reference:
    ----------
    * Abramowitz et al. (2012), Geophysical Research Letters, 39, L04808

    """
    zeroC = 273.15

    sat_vapress = 611.2 * np.exp(17.67 * ((tairK - zeroC) / (tairK - 29.65)))
    vapress = np.maximum(0.05, rh) * sat_vapress
    lw_down = 2.648 * tairK + 0.0346 * vapress - 474.0

    return lw_down

def interpolate_lai(lai_fname, ring):
    """
    """
    df_lai = pd.read_csv(lai_fname, usecols = ['ring','Date','LAIsmooth']) # daily data
    if ring == "amb":
        subset = df_lai[df_lai['ring'].isin(['2','3','6'])]
    elif ring == "ele":
        subset = df_lai[df_lai['ring'].isin(['1','4','5'])]
    else:
        subset = df_lai[df_lai['ring'].isin([ring[-1]])] #???
    subset = subset.groupby(by=["Date"])['LAIsmooth'].mean()

    tmp = pd.DataFrame(subset.values, columns=['LAI'])
    tmp['month'] = subset.values
    tmp['day']   = subset.values
    for i in np.arange(0,len(tmp['LAI']),1):
        tmp['month'][i] = subset.index[i][5:7]
        tmp['day'][i]   = subset.index[i][8:10]
    tmp = tmp.groupby(by=['month','day'])['LAI'].mean()

    rate = subset[-1]/tmp[(4)][(27)]

    day_len_1 = (pd.datetime(2019,7,1) - pd.datetime(2012,12,31)).days
    day_len_2 = (pd.datetime(2018,4,27) - pd.datetime(2012,12,31)).days
    day_len_3 = (pd.datetime(2013,1,1) - pd.datetime(2012,10,26)).days
    day_len_4 = (pd.datetime(2019,1,1) - pd.datetime(2013,1,1)).days
    day_len_5 = (pd.datetime(2019,2,28) - pd.datetime(2012,12,31)).days

    lai = pd.DataFrame(np.arange(np.datetime64('2013-01-01','D'),\
            np.datetime64('2019-07-02','D')), columns=['date'])
    lai['LAI'] = np.zeros(day_len_1)
    lai['LAI'][:day_len_2]          = subset[day_len_3:].values # 2013,1,1 - 2018,4,27
    lai['LAI'][day_len_2:day_len_4] = tmp.values[118:]*rate # 2018,4,28-2018,12,31
    lai['LAI'][day_len_4:day_len_5] = tmp.values[:59]*rate # 2019,1,1-2019,2,28
    lai['LAI'][day_len_5:]          = tmp.values[60:183]*rate # 2019,3,1-2019,7,1

    date = lai['date'] - np.datetime64('2013-01-01T00:00:00')
    lai['Date']  = np.zeros(len(lai))
    for i in np.arange(0,len(lai),1):
        lai['Date'][i] = date.iloc[i].total_seconds()
    grid_x = np.arange(0.,204940800.,1800)

    LAI_interp = np.interp(grid_x, lai['Date'].values, lai['LAI'].values)

    return LAI_interp

def thickness_weighted_average(var, nsoil, zse_vec):
    VAR     = 0.0
    for i in np.arange(0,nsoil,1):
        VAR += var[i]*zse_vec[i]
    VAR = VAR/sum(zse_vec)

    return VAR

if __name__ == "__main__":

    met_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/met_July2019/eucMet_gap_filled.csv"
    lai_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/met_July2019/eucLAI.csv"
    swc_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
    tdr_fname = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
    stx_fname = "/srv/ccrc/data25/z5218916/data/Eucface_data/soil_texture/FACE_P0018_RA_SOILTEXT_L2_20120501.csv"

    PTF = "Campbell_Cosby_multivariate"
    # "Campbell_Cosby_multi_Python"
    # "Campbell_Cosby_univariate"
    # "Campbell_Cosby_multivariate"
    # "Campbell_HC_SWC"
    soil_frac = "nearest" # "linear"
    layer_num = "6"
    neo_constrain = True
    tdr_constrain = True
    for ring in ["R1","R2","R3","R4","R5","R6","amb", "ele"]:
        out_fname = "EucFACE_met_%s.nc" % (ring)
        main(met_fname, lai_fname, swc_fname, tdr_fname, stx_fname, out_fname, PTF, soil_frac,\
             layer_num, neo_constrain, tdr_constrain, ring=ring)
