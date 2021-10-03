#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:17:51 2021

@author: lzhou
"""

import numpy as np
#import glob
import os
import pandas as pd
#import pickle
import geopandas as gpd
#from datetime import timedelta
#from shapely.geometry import Point
import matplotlib.pyplot as plt
#import geopandas as gpd
#from pyproj import CRS
#from shapely.ops import cascaded_union
#import seaborn as sns
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
shapely.speedups.enabled
#%%
wrk_dir = '/home/lzhou/Precipitation/Precipitation_Scripts'
os.chdir(wrk_dir)

data_type = 'IMERG'
case_name = 'IMERG_1500km_12'
factor = 'total'
output_dir = '/home/lzhou/Precipitation/Output'
data_dir = os.path.join(output_dir,case_name)

#%% combine total precipitation data from each event

# dummy = os.listdir(data_dir)
# files = [x for x in dummy if (factor in x) and ('pkl' in x)]
# files.sort()
# #%%
# os.chdir(data_dir)
# first_call = 1
# for infile in files:
#     print(infile)
#     data = pd.read_pickle(infile)
#     data.rename(columns={'precip':infile[:6]},inplace=True)
#     print(len(data))
    
#     if (first_call==1) :
#         df = data.copy()
#         first_call = 0
#     else:
#         df = df.merge(data,on=['lat','lon'],how = 'outer',)
# #%%
# filename = case_name+'_'+factor+'_precip.pkl'
# df.to_pickle(filename)

#%% load assembled event total precipitation data
filename = case_name+'_'+factor+'_precip.pkl'
infile = os.path.join(data_dir,filename)
df = pd.read_pickle(infile)
df.sort_values(by=['lat','lon'],ignore_index=True,inplace=True)
#%% sort event total preciptation for each grid
data0 = df.iloc[:,23:].to_numpy()
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
rps = np.array([20,10,5,4,2,1])
ranks = 20/rps
#%%
for ii in np.arange(0,len(ranks)):
    print(ii)
    fig,ax = plt.subplots(1,1,figsize=(12,8))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    cmax = china_points.iloc[:,int(ranks[ii])].mean()+china_points.iloc[:,int(ranks[ii])].std()*3.
    china_points.plot(column=int(ranks[ii]), ax=ax, 
                      cmap='Spectral_r', \
                      vmin=0, vmax=cmax, \
                      cax=cax,legend=True, legend_kwds={'label': "Event Total Precip. (mm)"})
    cn_shape.boundary.plot(color='black',ax=ax)
#df.plot(ax=ax,markersize=2,color='red')
    plt.text(0.3, 0.9, 'Max. total precip = %d mm'%china_points.iloc[:,int(ranks[ii])].max(), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.title.set_text(case_name + '_' + str(rps[ii]) + 'Y')
    ax.set_xlim([70,140])
    ax.set_ylim([15,55])    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')                              
    figname = case_name+'_RP'+str(rps[ii]) + 'Y_Event_Total_Precip' + '.png'
    fig.savefig(os.path.join('/home/lzhou/Precipitation/Output','IMERG_1500km_12', figname))
    plt.close(fig)
