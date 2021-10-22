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
wrk_dir = '/home/lzhou/Precipitation/Precipitation_Scripts'
os.chdir(wrk_dir)

data_type = 'APHRO'
case_name = 'APHRO_1000km_12'
factor = 'total'
data_dir0 = '/home/lzhou/Precipitation/Output'
data_dir1 = os.path.join(data_dir0,'Merged_Output')
figure_dir = os.path.join(data_dir0,'Figures')

#%% load assembled event total precipitation data
filename = case_name+'_'+factor+'_precip.pkl'
infile = os.path.join(data_dir1,filename)
df = pd.read_pickle(infile)
df.sort_values(by=['lat','lon'],ignore_index=True,inplace=True)

#%% ignore grids where total precipitation is less than 10mm
cols = df.columns
cols = cols[2:]
df[cols] = df[cols].where(df[cols]>=10., np.nan)

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
if data_type == 'IMERG':
    rps = np.array([20,10,5,4,2,1])
    ranks = 20/rps
elif data_type == 'APHRO':
    rps = np.array([18,9,6,3,2,1])
    ranks = 18/rps
elif data_type == 'ERA5':
    rps = np.array([40,20,10,8,5,4,2,1])
    ranks = 40/rps
#%%

font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
#%%
for ii in np.arange(0,len(ranks)):
    print(ii)
    fig,ax = plt.subplots(1,1,figsize=(8,6))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=-0.3)
    cmax = china_points.iloc[:,int(ranks[ii])].mean()+china_points.iloc[:,int(ranks[ii])].std()*3.
    china_points.plot(column=int(ranks[ii]), ax=ax, 
                      cmap='Spectral_r', \
                      vmin=0, vmax=cmax, \
                      cax=cax,legend=True, legend_kwds={'label': "Event Total Precip. (mm)"})
    cn_shape.boundary.plot(color='black',ax=ax)

    plt.text(0.3, 0.9, 'Max. total precip = %d mm'%china_points.iloc[:,int(ranks[ii])].max(), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    #ax.title.set_text(case_name + '_ReturnPeriod' + str(rps[ii]) + 'Y')
    title_str = case_name + '_' + str(rps[ii]) + 'Y'
    ax.set_title(title_str,fontweight='bold')
    ax.set_xlim([70,140])
    ax.set_ylim([15,55])    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')                              
    figname = case_name+'_RP'+str(rps[ii]) + 'Y_Event_Total_Precip' + '.png'
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, figname))
    plt.close(fig)
