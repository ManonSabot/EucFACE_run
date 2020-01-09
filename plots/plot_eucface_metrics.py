#!/usr/bin/env python

"""
Plot EucFACE soil moisture at observated dates

That's all folks.
"""

__author__ = "MU Mengyuan"
__version__ = "2019-10-06"
__changefrom__ = 'plot_eucface_swc_cable_vs_obs.py'

import os
import sys
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

#def main(fobs_Esoil, fobs, fcable, case_name, hk, b, ring, layer):
def main(fobs_Esoil, fobs_Trans, fobs_tdr, fobs_neo, fcable, case_name, ring, layer):

    subs_Esoil = read_obs_esoil(fobs_Esoil, ring)
    print(subs_Esoil)
    subs_Trans = read_obs_trans(fobs_Trans, ring)
    print(subs_Trans)
    subs_tdr   = read_obs_tdr(fobs_tdr, ring)
    print(subs_tdr)
    subs_neo   = read_obs_neo(fobs_neo, ring)
    print(subs_neo)
    subs_cable = read_cable(fcable, ring)
    print(subs_cable)

    # unify dates
    Esoil_obs = subs_Esoil['wuTP'].loc[np.all([subs_Esoil.index.isin(subs_Trans.index),subs_Esoil.index.isin(subs_cable.index)],axis=0)]
    Trans_obs = subs_Trans['volRing'].loc[np.all([subs_Trans.index.isin(subs_Esoil.index),subs_Trans.index.isin(subs_cable.index)],axis=0)]
    Esoil_cable = subs_cable["ESoil"].loc[np.all([subs_cable.index.isin(subs_Esoil.index),subs_cable.index.isin(subs_Trans.index)],axis=0)]
    Trans_cable = subs_cable["TVeg"].loc[np.all([subs_cable.index.isin(subs_Esoil.index),subs_cable.index.isin(subs_Trans.index)],axis=0)]

    mask      = np.any([np.isnan(Esoil_obs), np.isnan(Trans_obs)],axis=0)
    Esoil_obs      = Esoil_obs[mask == False]
    Trans_obs      = Trans_obs[mask == False]
    Esoil_cable    = Esoil_cable[mask == False]
    Trans_cable    = Trans_cable[mask == False]

    SM_50cm_obs  = subs_tdr.loc[subs_tdr.index.isin(subs_cable.index)]
    SM_50cm_cable= subs_cable["SM_50cm"].loc[subs_cable.index.isin(subs_tdr.index)]

    mask           = np.isnan(SM_50cm_obs)
    SM_50cm_obs    = SM_50cm_obs[mask == False]
    SM_50cm_cable  = SM_50cm_cable[mask == False]

    '''
    SM_50cm_obs  = subs_neo["SM_50cm"].loc[subs_neo.index.isin(subs_cable.index)]
    SM_50cm_cable= subs_cable["SM_50cm"].loc[subs_cable.index.isin(subs_neo.index)]
    '''
    SM_mid_obs  = subs_neo["SM_mid"].loc[subs_neo.index.isin(subs_cable.index)]
    SM_mid_cable= subs_cable["SM_mid"].loc[subs_cable.index.isin(subs_neo.index)]

    mask           = np.isnan(SM_mid_obs)
    SM_mid_obs     = SM_mid_obs[mask == False]
    SM_mid_cable   = SM_mid_cable[mask == False]

    SM_deep_obs  = subs_neo["SM_deep"].loc[subs_neo.index.isin(subs_cable.index)]
    SM_deep_cable= subs_cable["SM_deep"].loc[subs_cable.index.isin(subs_neo.index)]

    mask           = np.isnan(SM_deep_obs)
    SM_deep_obs    = SM_deep_obs[mask == False]
    SM_deep_cable  = SM_deep_cable[mask == False]

    Esoil_r   = stats.pearsonr(Esoil_obs, Esoil_cable)[0]
    Esoil_MSE = mean_squared_error(Esoil_obs, Esoil_cable)

    Trans_r   = stats.pearsonr(Trans_obs, Trans_cable)[0]
    Trans_MSE = mean_squared_error(Trans_obs, Trans_cable)

    Esoil_Trans_r   = stats.pearsonr(Esoil_obs/Trans_obs,Esoil_cable/Trans_cable)[0]
    Esoil_Trans_MSE = mean_squared_error(Esoil_obs/Trans_obs,Esoil_cable/Trans_cable)
    SM_50cm_r       = stats.pearsonr(SM_50cm_obs, SM_50cm_cable)[0]
    SM_50cm_MSE     = mean_squared_error(SM_50cm_obs, SM_50cm_cable)
    SM_mid_r        = stats.pearsonr(SM_mid_obs, SM_mid_cable)[0]
    SM_mid_MSE      = mean_squared_error(SM_mid_obs, SM_mid_cable)
    SM_deep_r       = stats.pearsonr(SM_deep_obs, SM_deep_cable)[0]
    SM_deep_MSE     = mean_squared_error(SM_deep_obs, SM_deep_cable)

    return Esoil_r, Trans_r, Esoil_Trans_r, SM_50cm_r, SM_mid_r, SM_deep_r,\
           Esoil_MSE, Trans_MSE, Esoil_Trans_MSE, SM_50cm_MSE, SM_mid_MSE, SM_deep_MSE;


def read_obs_esoil(fobs_Esoil, ring):

   est_esoil = pd.read_csv(fobs_Esoil, usecols = ['Ring','Date','wuTP','EfloorPred'])
   est_esoil['Date'] = pd.to_datetime(est_esoil['Date'],format="%d/%m/%Y",infer_datetime_format=False)
   est_esoil['Date'] = est_esoil['Date'] - pd.datetime(2011,12,31)
   est_esoil['Date'] = est_esoil['Date'].dt.days
   est_esoil = est_esoil.sort_values(by=['Date'])
   # divide neo into groups
   if ring == 'amb':
       subs = est_esoil[(est_esoil['Ring'].isin(['R2','R3','R6'])) & (est_esoil.Date > 366)]
   elif ring == 'ele':
       subs = est_esoil[(est_esoil['Ring'].isin(['R1','R4','R5'])) & (est_esoil.Date > 366)]
   else:
       subs = est_esoil[(est_esoil['Ring'].isin([ring]))  & (est_esoil.Date > 366)]

   subs = subs.groupby(by=["Date"]).mean()
   subs['wuTP']   = subs['wuTP'].clip(lower=0.)
   subs['wuTP']   = subs['wuTP'].replace(0., float('nan'))
   subs['EfloorPred'] = subs['EfloorPred'].clip(lower=0.)
   subs['EfloorPred'] = subs['EfloorPred'].replace(0., float('nan'))

   return subs

def read_obs_trans(fobs_Trans, ring):
   est_trans = pd.read_csv(fobs_Trans, usecols = ['Ring','Date','volRing'])
   est_trans['Date'] = pd.to_datetime(est_trans['Date'],format="%d/%m/%Y",infer_datetime_format=False)
   est_trans['Date'] = est_trans['Date'] - pd.datetime(2011,12,31)
   est_trans['Date'] = est_trans['Date'].dt.days
   est_trans = est_trans.sort_values(by=['Date'])
   # divide neo into groups
   if ring == 'amb':
       subs = est_trans[(est_trans['Ring'].isin(['R2','R3','R6'])) & (est_trans.Date > 366)]
   elif ring == 'ele':
       subs = est_trans[(est_trans['Ring'].isin(['R1','R4','R5'])) & (est_trans.Date > 366)]
   else:
       subs = est_trans[(est_trans['Ring'].isin([ring]))  & (est_trans.Date > 366)]

   subs = subs.groupby(by=["Date"]).mean()
   subs['volRing']   = subs['volRing'].clip(lower=0.)
   subs['volRing']   = subs['volRing'].replace(0., float('nan'))
   return subs

def read_obs_tdr(fobs, ring):

    tdr = pd.read_csv(fobs, usecols = ['Ring','Date','swc.tdr'])
    tdr['Date'] = pd.to_datetime(tdr['Date'],format="%Y-%m-%d",infer_datetime_format=False)
    tdr['Date'] = tdr['Date'] - pd.datetime(2011,12,31)
    tdr['Date'] = tdr['Date'].dt.days
    tdr = tdr.sort_values(by=['Date'])
    # divide neo into groups
    if ring == 'amb':
        subset = tdr[(tdr['Ring'].isin(['R2','R3','R6'])) & (tdr.Date > 366)]
    elif ring == 'ele':
        subset = tdr[(tdr['Ring'].isin(['R1','R4','R5'])) & (tdr.Date > 366)]
    else:
        subset = tdr[(tdr['Ring'].isin([ring]))  & (tdr.Date > 366)]

    subset = subset.groupby(by=["Date"]).mean()/100.
    subset = subset.xs('swc.tdr', axis=1, drop_level=True)
    return subset

def read_obs_neo(fobs_neo, ring):

    neo = pd.read_csv(fobs_neo, usecols = ['Ring','Depth','Date','VWC'])
    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)
    neo['Date'] = neo['Date'] - pd.datetime(2011,12,31)
    neo['Date'] = neo['Date'].dt.days
    neo = neo.sort_values(by=['Date','Depth'])

    if ring == 'amb':
        subset = neo[neo['Ring'].isin(['R2','R3','R6'])]
    elif ring == 'ele':
        subset = neo[neo['Ring'].isin(['R1','R4','R5'])]
    else:
        subset = neo[neo['Ring'].isin([ring])]

    subset = subset.groupby(by=["Depth","Date"]).mean()
    subset = subset.xs('VWC', axis=1, drop_level=True)
    x     = subset.index.get_level_values(1).values
    y     = subset.index.get_level_values(0).values
    value = subset.values

    X     = subset[(25)].index.values[20:]
    Y     = np.arange(0.5,460,1)

    grid_X, grid_Y = np.meshgrid(X,Y)

    grid_data = griddata((x, y) , value, (grid_X, grid_Y), method='nearest')

    neo_data = pd.DataFrame(subset[(25)].index.values[20:], columns=['dates'])
    neo_data["SM_50cm"] = np.mean(grid_data[0:50,:],axis=0)/100.
    neo_data["SM_mid"]  = np.mean(grid_data[50:200,:],axis=0)/100.
    neo_data["SM_deep"] = np.mean(grid_data[200:460,:],axis=0)/100.
    neo_data = neo_data.set_index('dates')
    return neo_data

def read_cable(fcable, ring):
    cable = nc.Dataset(fcable, 'r')
    Time  = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    cable_data = pd.DataFrame(cable.variables['TVeg'][:,0,0]*1800., columns=['TVeg'])
    cable_data['ESoil'] = cable.variables['ESoil'][:,0,0]*1800.
    cable_data['Evap'] = cable.variables['Evap'][:,0,0]*1800.

    if layer == "6":
        cable_data['SM_50cm'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.022 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.058 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.154 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*(0.5-0.022-0.058-0.154) )/0.5
        cable_data['SM_mid']  = ( cable.variables['SoilMoist'][:,3,0,0]*0.143 \
                                 + cable.variables['SoilMoist'][:,4,0,0]*1.085\
                                 + cable.variables['SoilMoist'][:,5,0,0]*0.272)/1.5
        cable_data['SM_deep']= cable.variables['SoilMoist'][:,5,0,0]
    elif layer == "31uni":
        cable_data['SM_50cm'] = ( cable.variables['SoilMoist'][:,0,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,1,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,2,0,0]*0.15 \
                                 + cable.variables['SoilMoist'][:,3,0,0]*0.05 )/0.5
        cable_data['SM_mid']  = cable.variables['SoilMoist'][:,3,0,0]*0.1
        for i in np.arange(4,13):
            cable_data['SM_mid']  = cable_data['SM_mid'] + cable.variables['SoilMoist'][:,i,0,0]*0.15
        cable_data['SM_mid']  = (cable_data['SM_mid'] + cable.variables['SoilMoist'][:,13,0,0]*0.05)/1.5

        cable_data['SM_deep']= cable.variables['SoilMoist'][:,13,0,0]*0.1
        for i in np.arange(14,30):
            cable_data['SM_deep']  = cable_data['SM_deep'] + cable.variables['SoilMoist'][:,i,0,0]*0.15
        cable_data['SM_deep']  = (cable_data['SM_deep'] + cable.variables['SoilMoist'][:,30,0,0]*0.1)/2.6

    cable_data['dates'] = Time
    cable_data = cable_data.set_index('dates')
    cable_data = cable_data.resample("D").agg('mean')
    cable_data.index = cable_data.index - pd.datetime(2011,12,31)
    cable_data.index = cable_data.index.days
    cable_data = cable_data.sort_values(by=['dates'])

    return cable_data

def plotting(metrics,ring):

    fig = plt.figure(figsize=[15,10])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color'] = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor'] = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    width = 1.
    ax1.imshow(metrics, interpolation='nearest')

    fig.savefig("EucFACE_metrics_%s.png" % (ring), bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    cases_6 = [ "met_only_6_","met_LAI_6_","met_LAI_vrt_6_","met_LAI_vrt_swilt-watr-ssat_6_",\
              "met_LAI_vrt_swilt-watr-ssat_SM_6_","met_LAI_vrt_swilt-watr-ssat_SM_6_litter",\
              "met_LAI_vrt_swilt-watr-ssat_SM_6_Or","met_LAI_vrt_swilt-watr-ssat_SM_6_fw-Haverd",\
              "met_LAI_vrt_swilt-watr-ssat_SM_6_fw-hie-exp","met_LAI_vrt_swilt-watr-ssat_SM_6_fw-hie-watpot"]
    cases_31 = ["met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter","met_LAI_vrt_swilt-watr-ssat_SM_31uni_Or",\
                "met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-Haverd", "met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-hie-exp",\
                "met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-hie-watpot"]

    rings  = ["amb"]#["R1","R2","R3","R4","R5","R6","amb","ele"]
    metrics= np.zeros((len(cases_6)+len(cases_31),12))

    for ring in rings:
        layer =  "6"
        for i,case_name in enumerate(cases_6):
            print(i)
            print(case_name)
            fobs_Esoil = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_underET.csv"
            fobs_Trans = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_sapflow.csv"
            fobs_tdr = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
            fobs_neo = "/srv/ccrc/data25/z5218916/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
            fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            metrics[i,:] = main(fobs_Esoil, fobs_Trans, fobs_tdr, fobs_neo, fcable, case_name, ring, layer)
        print("i = %s" % str(i))
        j = i
        layer =  "31uni"
        for i,case_name in enumerate(cases_31):
            print(case_name)
            fobs_Esoil = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_underET.csv"
            fobs_Trans = "/srv/ccrc/data25/z5218916/data/Eucface_data/FACE_PACKAGE_HYDROMET_GIMENO_20120430-20141115/data/Gimeno_wb_EucFACE_sapflow.csv"
            fobs_tdr = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_average_above_the_depth/swc_tdr.csv"
            fobs_neo = "/srv/ccrc/data25/z5218916/data/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
            fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            metrics[i+j+1,:] = main(fobs_Esoil, fobs_Trans, fobs_tdr, fobs_neo, fcable, case_name, ring, layer)
        #print(metrics)
        plotting(metrics,ring)
        #metrics.to_csv("EucFACE_amb_%slayers_%s_gw_on_or_on.csv" %(layers, case_name))
        np.savetxt("EucFACE_metrics_%s.csv" % (ring), metrics, delimiter=",")
'''

ax1.plot(subset.index, subset.values,   c="green", lw=1.0, ls="-", label="tdr")
ax1.plot(x, SoilMoist.values,c="orange", lw=1.0, ls="-", label="swc")

tmp1 = SoilMoist['SoilMoist'].loc[SoilMoist.index.isin(subset.index)]
tmp2 = subset.loc[subset.index.isin(SoilMoist.index)]
mask = np.isnan(tmp2)
print(mask)
tmp1 = tmp1[mask == False]
tmp2 = tmp2[mask == False]
#print(np.isnan(tmp1).values.any())
#print(np.isnan(tmp2).values.any())
cor_tdr = stats.pearsonr(tmp1,tmp2)
mse_tdr = mean_squared_error(tmp2, tmp1)
ax1.set_title("r = % 5.3f , MSE = % 5.3f" %(cor_tdr[0], np.sqrt(mse_tdr)))
print("-----------------------------------------------")
print(mse_tdr)
ax1.plot(x, swilt,           c="black", lw=1.0, ls="-", label="swilt")
ax1.plot(x, sfc,             c="black", lw=1.0, ls="-.", label="sfc")
ax1.plot(x, ssat,            c="black", lw=1.0, ls=":", label="ssat")
ax3.plot(x, Fwsoil.values,   c="forestgreen", lw=1.0, ls="-", label="Fwsoil")
ax5.plot(x, TVeg['TVeg'].rolling(window=5).mean(),     c="green", lw=1.0, ls="-", label="Trans") #.rolling(window=7).mean()
ax5.plot(x, ESoil['ESoil'].rolling(window=5).mean(),    c="orange", lw=1.0, ls="-", label="ESoil") #.rolling(window=7).mean()
ax5.scatter(subs_Trans.index, subs_Trans['volRing'], marker='o', c='',edgecolors='blue', s = 2., label="Trans Obs") # subs['EfloorPred']
ax5.scatter(subs_Esoil.index, subs_Esoil['wuTP'], marker='o', c='',edgecolors='red', s = 2., label="ESoil Obs") # subs['EfloorPred']

#ax7.plot(x, Tair['Tair'].rolling(window=7).mean(),     c="red",    lw=1.0, ls="-", label="Tair")
#ax7.plot(x, VegT['VegT'].rolling(window=7).mean(),     c="orange", lw=1.0, ls="-", label="VegT")

#rects2 = ax2.bar(x, Rainf['Rainf'], width, color='royalblue', label='Obs')
#ax4.plot(x, Qair['Qair'].rolling(window=7).mean(),     c="royalblue", lw=1.0, ls="-", label="Qair")
#ax6.plot(x, Wind['Wind'].rolling(window=7).mean(),     c="darkgoldenrod", lw=1.0, ls="-", label="Wind")
#ax8.plot(x, Rnet['Rnet'].rolling(window=7).mean(),     c="crimson", lw=1.0, ls="-", label="Rnet")

cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
xtickslocs    = [367,732,1097,1462,1828,2193,2558]

#plt.setp(ax1.get_xticklabels(), visible=False)
#ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
#ax1.set_ylabel("CABLE Runoff (mm)")
#ax1.axis('tight')
#ax1.set_xlim(0,2374)

plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
ax1.set_ylabel("VWC (m3/m3)")
ax1.axis('tight')
ax1.set_ylim(0,0.5)
ax1.set_xlim(367,2739)
ax1.legend()

plt.setp(ax3.get_xticklabels(), visible=False)
ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
ax3.set_ylabel("Fwsoil (-)")
ax3.axis('tight')
ax3.set_ylim(0.,1.2)
ax3.set_xlim(367,2739)

#plt.setp(ax5.get_xticklabels(), visible=False)
ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates) ####
ax5.set_ylabel("ET ($mm d^{-1}$)")
ax5.axis('tight')
ax5.set_ylim(0.,4.5)
ax5.set_xlim(367,2739)
ax5.legend()
'''
