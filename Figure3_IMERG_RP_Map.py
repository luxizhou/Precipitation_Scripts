#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:34:07 2022

@author: lzhou
"""

import os
import numpy as np
import pandas as pd
#import xarray as xr
#import rioxarray as rio
#from affine import Affine
#from datetime import timedelta
#from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns; sns.set_style('white')

import geopandas as gpd
#from pyproj import CRS
#from shapely.ops import cascaded_union
#import seaborn as sns; sns.set_theme()
#import matplotlib.pyplot as plt
#import netCDF4 as nc
import shapely
shapely.speedups.enabled

import warnings 
warnings.filterwarnings("ignore")

import precipitation_utils 


os.chdir('/home/lzhou/Precipitation/Precipitation_Scripts')

IMERG_folder = '/media/lzhou/Extreme SSD/Precipitation/IMERG'
CMA_folder = '/home/lzhou/Precipitation/Data/CMA_Historical_Data/Wind_Rainfall/'
Output_folder = '/home/lzhou/Precipitation/Output'
Output_folder2 = '/home/lzhou/Precipitation/Precipitation_Scripts/Output/'
Figure_folder = os.path.join(Output_folder,'Figures')

case = 'IMERG_10degree_12'
item = 'precipitationCal'
threshold = 10.

#%%
if 'IMERG' in case:
    rps = np.array([20,10,5,4,2,1])
    ranks = 20/rps-1
elif 'APHRO' in case:
    rps = np.array([18,9,6,3,2,1])
    ranks = 18/rps-1
elif 'ERA5' in case:
    rps = np.array([40,20,10,8,5,4,2,1])
    ranks = 40/rps-1

# plots configuration
font = {'family' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)

#%% Get the domain of greater China
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape = world[world.name=='China'].copy()
cn_shape.reset_index(drop=True,inplace=True)
tw_shape = world[world.name=='Taiwan'].copy()
tw_shape.reset_index(drop=True,inplace=True)
greater_china = cn_shape.geometry.union(tw_shape.geometry)


#%% Total precipitation analysis
filename = item+'_event_total_precip.pkl' 
infile = os.path.join(Output_folder,case,filename)
event_total = pd.read_pickle(infile)
event_total = event_total[event_total.precip>=threshold].reset_index(drop=True)
#event_total = event_total[event_total.CMAID>200100]
event_total = event_total[(event_total.CMAID>200100)&(event_total.CMAID<202100)]
#%%
if 'IMERG' in case:
    sorted_df = precipitation_utils.sort_precipitation_at_grid(event_total,'CMAID','precip',0)
#%%
dummy_df = sorted_df.loc[:,~pd.isna(sorted_df).all(axis=0)]

'''
event_total_pivot = event_total.pivot(index=['lat','lon'],columns='CMAID',values='precip').reset_index()
# sort precipitation in each grid
data = event_total_pivot.to_numpy()
if 'IMERG' in case:
    data1 = np.sort(-data[:,20:],axis=1) #only consider data since 2001 as IMERG is only availabe for part of year 2020
data2 = -data1
sorted_df = pd.DataFrame(data2)
sorted_df['lat'] = event_total_pivot.lat
sorted_df['lon'] = event_total_pivot.lon
'''
# make geodata frame of precipitation and only keep points within greater China
sorted_df = gpd.GeoDataFrame(dummy_df, \
                             geometry=gpd.points_from_xy(dummy_df.lon, dummy_df.lat), \
                             crs="epsg:4326")
idx = sorted_df.within(greater_china.iloc[0])
china_points = sorted_df.loc[idx].copy()

filename = case + '_sorted_total_precip_2001_2020.pkl'
ofile = os.path.join(Output_folder,case,filename)
china_points.to_pickle(ofile)

#%% plot RP maps
for ii in np.arange(0,len(ranks)):
    print(int(ranks[ii]))
    fig,ax = plt.subplots(1,1,figsize=(16,16))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=-0.3)
    cmax = china_points.loc[:,int(ranks[ii])].mean()+china_points.loc[:,int(ranks[ii])].std()*3.
    china_points.plot(column=int(ranks[ii]), ax=ax, 
                      cmap='Spectral_r', \
                      vmin=0, vmax=cmax, \
                      cax=cax,legend=True, legend_kwds={'label': "Event Total Precip. (mm)"})
    greater_china.boundary.plot(color='black',ax=ax)

    plt.text(0.3, 0.9, 'Max. total precip = %d mm'%china_points.loc[:,int(ranks[ii])].max(), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    #ax.title.set_text(case_name + '_ReturnPeriod' + str(rps[ii]) + 'Y')
    title_str = case + '_' + str(rps[ii]) + 'Y'
    ax.set_title(title_str,fontweight='bold')
    ax.set_xlim([70,140])
    ax.set_ylim([15,55])    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')                              
    figname = case +'_RP'+str(rps[ii]) + 'Y_Event_Total_Precip_until2020' + '.png'
    fig.tight_layout()
    fig.savefig(os.path.join(Figure_folder, figname))
    plt.close(fig)



#%% load and sort daiy precip at each grid

yy = 2001
filename = item+'_' + str(yy) +'.pkl' 
infile = os.path.join(Output_folder,case,filename)
df = pd.read_pickle(infile)
df = df.reset_index()
df = df[df.precip>=threshold]
df['lat'] = df.y.round(2)
df['lon'] = df.x.round(2)
df.drop(columns=['x','y'],inplace=True)
df.sort_values(by=['lat','lon','time'],inplace=True)
df = df.reset_index(drop=True)   

# take out duplicate rows of (lat,lon,time)
idx = df[['lat','lon','time']].duplicated(keep='first')
uniques = df.loc[~idx].copy()
sorted_df = precipitation_utils.sort_precipitation_at_grid(uniques,'time','precip',0)

# only keep points on land
sorted_df = pd.merge(sorted_df,china_points[['lat','lon']],on=['lat','lon'],how='inner')


# remove columngs that has no values
imerg_daily_precip = sorted_df.loc[:,~pd.isna(sorted_df).all(axis=0)]

print(yy,'year max:',imerg_daily_precip.max().max())
#imerg_daily_precip = sorted_df.iloc[:,0:22]

for yy in np.arange(2002,2021):
    print(yy)
    filename = item+'_' + str(yy) +'.pkl' 
    infile = os.path.join(Output_folder,case,filename)
    df = pd.read_pickle(infile)
    df = df.reset_index()
    df = df[df.precip>=threshold]
    df['lat'] = df.y.round(2)
    df['lon'] = df.x.round(2)
    df.drop(columns=['x','y'],inplace=True)
    df.sort_values(by=['lat','lon','time'],inplace=True)
    df = df.reset_index(drop=True)
    
    # take out duplicate rows of (lat,lon,time)
    idx = df[['lat','lon','time']].duplicated(keep='first')
    uniques = df.loc[~idx].copy()
    sorted_df = precipitation_utils.sort_precipitation_at_grid(uniques,'time','precip',0)
    dummy_df = pd.merge(sorted_df,china_points[['lat','lon']],on=['lat','lon'],how='inner')
    sorted_df = dummy_df.loc[:,~pd.isna(dummy_df).all(axis=0)]
    
    print(yy,'year max:',sorted_df.max().max())

    
    merge_df = pd.merge(imerg_daily_precip,sorted_df,on=['lat','lon'],how='outer')
    merge_df = merge_df.set_index(['lat','lon'])
    data = merge_df.values
    data1 = np.sort(-data,axis=1) #only consider data since 2001 as IMERG is only availabe for part of year 2020
    data2 = -data1
    imerg_daily_precip = pd.DataFrame(data2,index=merge_df.index)

    dummy_df = imerg_daily_precip.loc[:,~pd.isna(imerg_daily_precip).all(axis=0)]
    imerg_daily_precip = dummy_df.reset_index()
    # only keep points on land
    #imerg_daily_precip = pd.merge(imerg_daily_precip,china_points[['lat','lon']],on=['lat','lon'],how='inner')

#%%
# make geodata frame of precipitation and only keep points within greater China
imerg_daily_precip = gpd.GeoDataFrame(imerg_daily_precip, geometry=gpd.points_from_xy(imerg_daily_precip.lon, imerg_daily_precip.lat), \
                                      crs="epsg:4326")
#%%
filename = case + '_sorted_daily_precip_2001_2020.pkl'
ofile = os.path.join(Output_folder,case,filename)
imerg_daily_precip.to_pickle(ofile)
#%% plot RP maps
for ii in np.arange(0,len(ranks)):
    print(int(ranks[ii]))
    fig,ax = plt.subplots(1,1,figsize=(16,16))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=-0.3)
    cmax = imerg_daily_precip.loc[:,int(ranks[ii])].mean()+imerg_daily_precip.loc[:,int(ranks[ii])].std()*3.
    imerg_daily_precip.plot(column=int(ranks[ii]), ax=ax, 
                      cmap='Spectral_r', \
                      vmin=0, vmax=cmax, \
                      cax=cax,legend=True, legend_kwds={'label': "Event Total Precip. (mm)"})
    greater_china.boundary.plot(color='black',ax=ax)

    plt.text(0.3, 0.9, 'Max. daily precip = %d mm'%imerg_daily_precip.loc[:,int(ranks[ii])].max(), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    #ax.title.set_text(case_name + '_ReturnPeriod' + str(rps[ii]) + 'Y')
    title_str = case + '_Daily_Precip_' + str(rps[ii]) + 'Y'
    ax.set_title(title_str,fontweight='bold')
    ax.set_xlim([70,140])
    ax.set_ylim([15,55])    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')                              
    figname = case +'_RP'+str(rps[ii]) + 'Y_Daily_Precip_until2019' + '.png'
    fig.tight_layout()
    fig.savefig(os.path.join(Figure_folder, figname))
    plt.close(fig)




