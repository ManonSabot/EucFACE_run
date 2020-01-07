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

def main(fobs, fcable, case_name, ring, layer):

    neo = pd.read_csv(fobs, usecols = ['Ring','Depth','Date','VWC'])
    neo['Date'] = pd.to_datetime(neo['Date'],format="%d/%m/%y",infer_datetime_format=False)
    neo['Date'] = neo['Date'] - pd.datetime(2012,12,31)
    neo['Date'] = neo['Date'].dt.days
    neo = neo.sort_values(by=['Date','Depth'])

    print(neo['Depth'].unique())

    if ring == 'amb':
        subset = neo[neo['Ring'].isin(['R2','R3','R6'])]
    elif ring == 'ele':
        subset = neo[neo['Ring'].isin(['R1','R4','R5'])]
    else:
        subset = neo[neo['Ring'].isin([ring])]

    subset = subset.groupby(by=["Depth","Date"]).mean()
    subset = subset.xs('VWC', axis=1, drop_level=True)
    subset[:] = subset[:]/100.
    #date  = subset[(25)].index.values

# _________________________ CABLE ___________________________
    cable = nc.Dataset(fcable, 'r')

    Time = nc.num2date(cable.variables['time'][:],cable.variables['time'].units)
    if layer == "6":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=[1.1, 5.1, 15.7, 43.85, 118.55, 316.4])
    elif layer == "13":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns = \
                    [1.,4.5,10.,19.5,41,71,101,131,161,191,221,273.5,386])
    elif layer == "31uni":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns = \
                   [7.5,   22.5 , 37.5 , 52.5 , 67.5 , 82.5 , 97.5 , \
                    112.5, 127.5, 142.5, 157.5, 172.5, 187.5, 202.5, \
                    217.5, 232.5, 247.5, 262.5, 277.5, 292.5, 307.5, \
                    322.5, 337.5, 352.5, 367.5, 382.5, 397.5, 412.5, \
                    427.5, 442.5, 457.5 ])
    elif layer == "31exp":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=\
                  [ 1.021985, 2.131912, 2.417723, 2.967358, 3.868759, 5.209868,\
                    7.078627, 9.562978, 12.75086, 16.73022, 21.58899, 27.41512,\
                    34.29655, 42.32122, 51.57708, 62.15205, 74.1341 , 87.61115,\
                    102.6711, 119.402 , 137.8918, 158.2283, 180.4995, 204.7933,\
                    231.1978, 259.8008, 290.6903, 323.9542, 359.6805, 397.9571,\
                    438.8719 ])
    elif layer == "31para":
        SoilMoist = pd.DataFrame(cable.variables['SoilMoist'][:,:,0,0], columns=\
                    [ 1.000014,  3.47101, 7.782496, 14.73158, 24.11537, 35.73098, \
                      49.37551, 64.84607, 81.93976, 100.4537, 120.185 , 140.9308, \
                      162.4881, 184.6541, 207.2259, 230.    , 252.7742, 275.346 , \
                      297.512 , 319.0693, 339.8151, 359.5464, 378.0603, 395.154 , \
                      410.6246, 424.2691, 435.8847, 445.2685, 452.2176, 456.5291, \
                      459.0001 ])
    SoilMoist['dates'] = Time
    SoilMoist = SoilMoist.set_index('dates')
    SoilMoist = SoilMoist.resample("D").agg('mean')
    SoilMoist.index = SoilMoist.index - pd.datetime(2012,12,31)
    SoilMoist.index = SoilMoist.index.days
    SoilMoist = SoilMoist.stack() # turn multi-columns into one-column
    SoilMoist = SoilMoist.reset_index() # remove index 'dates'
    SoilMoist = SoilMoist.rename(index=str, columns={"level_1": "Depth"})
    SoilMoist = SoilMoist.sort_values(by=['Depth','dates'])

    date_start_cable = pd.datetime(2013,1,1) - pd.datetime(2012,12,31)
    date_end_cable   = pd.datetime(2019,6,30) - pd.datetime(2012,12,31)
    date_start_cable = date_start_cable.days
    date_end_cable   = date_end_cable.days

    ntimes      = len(np.unique(SoilMoist['dates']))
    dates       = np.unique(SoilMoist['dates'].values)
    x_cable     = SoilMoist['dates'].values
    y_cable     = SoilMoist['Depth'].values
    value_cable = SoilMoist.iloc[:,2].values

    '''
    x_cable     = np.concatenate(( dates, SoilMoist['dates'].values,dates)) # Time
    y_cable     = np.concatenate(([0]*ntimes,SoilMoist['Depth'].values,[460]*ntimes))# Depth
    value_cable = np.concatenate(( SoilMoist.iloc[:ntimes,2].values, \
                                   SoilMoist.iloc[:,2].values,         \
                                   SoilMoist.iloc[-(ntimes):,2].values ))
    '''
    # add the 12 depths to 0
    X_cable     = np.arange(date_start_cable,date_end_cable,1) # 2013-1-1 to 2019-6-30
    Y_cable     = [25,50,75,100,125,150,200,250,300,350,400,450]
    grid_X_cable, grid_Y_cable = np.meshgrid(X_cable,Y_cable)

    # interpolate
    grid_cable = griddata((x_cable, y_cable) , value_cable, (grid_X_cable, grid_Y_cable),\
                 method='nearest') #'cubic')#'linear')#'nearest')
    print(grid_cable.shape)

# ____________________ Plot obs _______________________
    fig = plt.figure(figsize=[30,15],constrained_layout=True)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    almost_black = '#262626'
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black
    plt.rcParams['text.color'] = almost_black
    plt.rcParams['axes.edgecolor'] = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black
    cmap = plt.cm.viridis_r

    ax1 = fig.add_subplot(431)
    ax2 = fig.add_subplot(432)
    ax3 = fig.add_subplot(433)
    ax4 = fig.add_subplot(434)
    ax5 = fig.add_subplot(435)
    ax6 = fig.add_subplot(436)
    ax7 = fig.add_subplot(437)
    ax8 = fig.add_subplot(438)
    ax9 = fig.add_subplot(439)
    ax10= fig.add_subplot(4,3,10)
    ax11= fig.add_subplot(4,3,11)
    ax12= fig.add_subplot(4,3,12)

    ax1.plot(X_cable, grid_cable[0,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax2.plot(X_cable, grid_cable[1,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax3.plot(X_cable, grid_cable[2,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax4.plot(X_cable, grid_cable[3,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax5.plot(X_cable, grid_cable[4,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax6.plot(X_cable, grid_cable[5,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax7.plot(X_cable, grid_cable[6,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax8.plot(X_cable, grid_cable[7,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax9.plot(X_cable, grid_cable[8,:],  c="green", lw=1.0, ls="-", label="CABLE")
    ax10.plot(X_cable, grid_cable[9,:], c="green", lw=1.0, ls="-", label="CABLE")
    ax11.plot(X_cable, grid_cable[10,:],c="green", lw=1.0, ls="-", label="CABLE")
    ax12.plot(X_cable, grid_cable[11,:],c="green", lw=1.0, ls="-", label="CABLE")

    ax1.scatter(subset[(25)].index.values, subset[(25)].values, marker='.', label="obs")
    ax2.scatter(subset[(50)].index.values, subset[(50)].values, marker='.', label="obs")
    ax3.scatter(subset[(75)].index.values, subset[(75)].values, marker='.', label="obs")
    ax4.scatter(subset[(100)].index.values, subset[(100)].values, marker='.', label="obs")
    ax5.scatter(subset[(125)].index.values, subset[(125)].values, marker='.', label="obs")
    ax6.scatter(subset[(150)].index.values, subset[(150)].values, marker='.', label="obs")
    ax7.scatter(subset[(200)].index.values, subset[(200)].values, marker='.', label="obs")
    ax8.scatter(subset[(250)].index.values, subset[(250)].values, marker='.', label="obs")
    ax9.scatter(subset[(300)].index.values, subset[(300)].values, marker='.', label="obs")
    ax10.scatter(subset[(350)].index.values,subset[(350)].values, marker='.', label="obs")
    ax11.scatter(subset[(400)].index.values,subset[(400)].values, marker='.', label="obs")
    ax12.scatter(subset[(450)].index.values,subset[(450)].values, marker='.', label="obs")

    cleaner_dates = ["2013","2014","2015","2016","2017","2018","2019"]
    xtickslocs    = [1,365,730,1095,1461,1826,2191]

    dd = [25,50,75,100,125,150,200,250,300,350,400,450]
    cor_neo = np.zeros(12)
    mse_neo = np.zeros(12)

    for i,d in enumerate(dd):
        tmp1 = grid_cable[i,np.isin(X_cable,subset[(d)].index)]
        tmp2 = subset[(d)][np.isin(subset[(d)].index,X_cable)]
        mask = tmp2 > 0.0
        tmp1 = tmp1[mask]
        tmp2 = tmp2[mask]
        cor_neo[i]= stats.pearsonr(tmp1,tmp2)[0]
        mse_neo[i]= mean_squared_error(tmp1, tmp2)
        print(cor_neo[i])
        print(mse_neo[i])

    ax1.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax1.set_title("25cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[0], np.sqrt(mse_neo[0])))
    ax1.axis('tight')
    ax2.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax2.set_title("50cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[1],np.sqrt(mse_neo[1])))
    ax2.axis('tight')
    ax3.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax3.set_title("75cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[2], np.sqrt(mse_neo[2])))
    ax3.axis('tight')
    ax4.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax4.set_title("100cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[3], np.sqrt(mse_neo[3])))
    ax4.axis('tight')
    ax5.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax5.set_title("125cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[4], np.sqrt(mse_neo[4])))
    ax5.axis('tight')
    ax6.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax6.set_title("150cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[5], np.sqrt(mse_neo[5])))
    ax6.axis('tight')
    ax7.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax7.set_title("200cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[6], np.sqrt(mse_neo[6])))
    ax7.axis('tight')
    ax8.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax8.set_title("250cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[7], np.sqrt(mse_neo[7])))
    ax8.axis('tight')
    ax9.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax9.set_title("300cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[8], np.sqrt(mse_neo[8])))
    ax9.axis('tight')
    ax10.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax10.set_title("350cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[9], np.sqrt(mse_neo[9])))
    ax10.axis('tight')
    ax11.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax11.set_title("400cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[10], np.sqrt(mse_neo[10])))
    ax11.axis('tight')
    ax12.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax12.set_title("450cm, r=% 5.3f, RMSE=% 5.3f" %(cor_neo[11], np.sqrt(mse_neo[11])))
    ax12.axis('tight')

    ax1.set_xlim([0,2374])
    ax1.set_ylim([0.0,0.5])
    ax2.set_xlim([0,2374])
    ax2.set_ylim([0.0,0.5])
    ax3.set_xlim([0,2374])
    ax3.set_ylim([0.0,0.5])
    ax4.set_xlim([0,2374])
    ax4.set_ylim([0.0,0.5])
    ax5.set_xlim([0,2374])
    ax5.set_ylim([0.0,0.5])
    ax6.set_xlim([0,2374])
    ax6.set_ylim([0.0,0.5])
    ax7.set_xlim([0,2374])
    ax7.set_ylim([0.0,0.5])
    ax8.set_xlim([0,2374])
    ax8.set_ylim([0.0,0.5])
    ax9.set_xlim([0,2374])
    ax9.set_ylim([0.0,0.5])
    ax10.set_xlim([0,2374])
    ax10.set_ylim([0.0,0.5])
    ax11.set_xlim([0,2374])
    ax11.set_ylim([0.0,0.5])
    ax12.set_xlim([0,2374])
    ax12.set_ylim([0.0,0.5])

    ax1.legend()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax6.get_xticklabels(), visible=False)
    plt.setp(ax7.get_xticklabels(), visible=False)
    plt.setp(ax8.get_xticklabels(), visible=False)
    plt.setp(ax9.get_xticklabels(), visible=False)

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    plt.setp(ax6.get_yticklabels(), visible=False)
    plt.setp(ax8.get_yticklabels(), visible=False)
    plt.setp(ax9.get_yticklabels(), visible=False)
    plt.setp(ax11.get_yticklabels(), visible=False)
    plt.setp(ax12.get_yticklabels(), visible=False)

    #plt.suptitle('Volumetric Water Content - %s (m3/m3)' %(case_name))
    plt.suptitle('Volumetric Water Content (m3/m3)')
    fig.savefig("EucFACE_neo_%s_%s.png" % (os.path.basename(case_name).split("/")[-1], ring), bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    layer = "31uni"
    '''
    if layer == "6":
	cases = [ "met_only_6_","met_only_6_gw-off"]
	"""
        cases = [ "met_only_6_","met_LAI_6_","met_LAI_vrt_6_","met_LAI_vrt_swilt-watr-ssat_6_",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_6_","met_LAI_vrt_swilt-watr-ssat_SM_6_litter",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_6_Or","met_LAI_vrt_swilt-watr-ssat_SM_6_fw-Haverd",\
                  "met_LAI_vrt_swilt-watr-ssat_SM_6_fw-hie-exp","met_LAI_vrt_swilt-watr-ssat_SM_6_fw-hie-watpot"]
	"""
    elif layer == "31para":
        cases = ["ctl_met_LAI_vrt_SM_swilt-watr_31para"]
    elif layer == "31exp":
        cases = ["ctl_met_LAI_vrt_SM_swilt-watr_31exp"]
    elif layer == "31uni":
        cases = ["met_LAI_vrt_swilt-watr-ssat_SM_31uni_litter","met_LAI_vrt_swilt-watr-ssat_SM_31uni_Or",\
                 "met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-hie-exp",\
                 "met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-Haverd","met_LAI_vrt_swilt-watr-ssat_SM_31uni_fw-hie-watpot"]
    '''

    #cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs",\
    #                  "met_LAI_vrt_swilt-watr-ssat_SM_31uni_hydsx01-hydsx*_fw-hie-exp"))

#    cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen_fw-hie-exp_31uni_1/outputs",\
#                       "met_LAI_vrt_swilt-watr-ssat_SM_31uni_bch=*_soil_moisture_fix_or_fix_check"))
    cases = glob.glob(os.path.join("/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs",\
                        "met_LAI_vrt_swilt-watr-ssat_SM_31uni_"))
    #                 "met_LAI_vrt_swilt-watr-ssat_SM_31uni_GW-wb_SM-fix_or_fix"))
    #                   "met_LAI_vrt_swilt-watr-ssat_SM_31uni_GW-wb_SM-fix_fw-hie-exp"))
    rings = ["amb"]#"R1","R2","R3","R4","R5","R6",,"ele"
    for case_name in cases:
        for ring in rings:
            fobs = "/srv/ccrc/data25/z5218916/cable/EucFACE/Eucface_data/swc_at_depth/FACE_P0018_RA_NEUTRON_20120430-20190510_L1.csv"
            #fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            #fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run_sen/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            #fcable ="/srv/ccrc/data25/z5218916/cable/EucFACE/EucFACE_run/outputs/%s/EucFACE_%s_out.nc" % (case_name, ring)
            fcable ="%s/EucFACE_%s_out.nc" % (case_name, ring)
            main(fobs, fcable, case_name, ring, layer)
