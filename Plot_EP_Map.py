#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Oct  3 15:17:51 2021

@author: lzhou
"""

import numpy as np
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
shapely.speedups.enabled

#%%

def Find_Exceedance_Probability(indata,threshold,record_period):
    
    indata['EP'] = 0
    points = [1]
    rr = 0
    while len(points)>0:
        points = indata[indata.iloc[:,rr]>threshold].index
        if len(points)>0:
            indata.loc[points,'EP'] = (rr+1.)/record_period
            rr = rr + 1
    
    return indata
            

#%%
wrk_dir = '/home/lzhou/Precipitation/Precipitation_Scripts'
os.chdir(wrk_dir)

data_type = 'ERA5'
case_name = 'ERA5_1000km_12'
factor = 'total'
data_dir0 = '/home/lzhou/Precipitation/Output'
data_dir1 = os.path.join(data_dir0,'Merged_Output')
figure_dir = os.path.join(data_dir0,'Figures')

IMERG_period = 20.
APHRO_period = 18.
ERA5_period = 2020.-1980.

if data_type == 'IMERG':
    thresholds = [1000,500,200]
elif data_type == 'ERA5':
    thresholds = [5000, 2000]
else:
    thresholds = [500,200]

#%% load assembled event total precipitation data
filename = case_name+'_'+factor+'_precip.pkl'
infile = os.path.join(data_dir1,filename)
df = pd.read_pickle(infile)
df.sort_values(by=['lat','lon'],ignore_index=True,inplace=True)
#%% sort event total preciptation for each grid
if case_name == 'IMERG_1500km_12':
    data0 = df.iloc[:,23:].to_numpy()   # This is counting from 2001-2020, IMERG_15km_12 Data
elif case_name == 'IMERG_1000km_12':
    data0 = df.iloc[:,19:].to_numpy()
elif ('APHRO' in case_name) or ('ERA5' in case_name):
    data0 = df.iloc[:,2:].to_numpy()
    
data1 = np.sort(-data0,axis=1) 
data2 = -data1
sorted_df = pd.DataFrame(data2)
sorted_df['lat'] = df['lat']
sorted_df['lon'] = df['lon']
sorted_df = gpd.GeoDataFrame(sorted_df, \
                             geometry=gpd.points_from_xy(sorted_df.lon, sorted_df.lat), \
                             crs="epsg:4326")
    
#%% only keep grids that are within China boundary
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape = world[world.name=='China'].copy()
cn_shape.reset_index(drop=True,inplace=True)
idx = sorted_df.geometry.apply(lambda x: x.within(cn_shape.iloc[0].geometry))
china_points = sorted_df.loc[idx].copy()

#%%

font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
#%%
for ii in np.arange(0,len(thresholds)):
    if 'IMERG' in case_name:
        record_period = IMERG_period
    elif 'APHRO' in case_name:
        record_period = APHRO_period
    elif 'ERA5' in case_name:
        record_period = ERA5_period
        
    Find_Exceedance_Probability(china_points, thresholds[ii], record_period)
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=-0.3)
    #cmax = china_points.iloc[:,int(ranks[ii])].mean()+china_points.iloc[:,int(ranks[ii])].std()*3.
    china_points[china_points.EP>0].plot(column='EP', ax=ax, cmap='Spectral_r', \
    #                  vmin=0, vmax=cmax, \
                                         cax=cax,legend=True, legend_kwds={'label': "Exceedance Probability"})
    cn_shape.boundary.plot(color='black',ax=ax)

    #plt.text(0.3, 0.9, 'Max. total precip = %d mm'%china_points.iloc[:,int(ranks[ii])].max(), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    #ax.title.set_text(case_name + '_ReturnPeriod' + str(rps[ii]) + 'Y')
    title_str = case_name + ' ' + 'EP @ ' + str(thresholds[ii]) + 'mm total precip.'
    ax.set_title(title_str,fontweight='bold')
    ax.set_xlim([70,140])
    ax.set_ylim([15,55])    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')                              
    figname = case_name+'_EP_'+str(thresholds[ii]) + 'mm_Event_Total_Precip' + '.png'
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, figname))
    plt.close(fig)
