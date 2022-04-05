#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:54:26 2022

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

# plots configuration
font = {'family' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)

#%% load IMERG data

filename = case + '_sorted_total_precip_2001_2018.pkl'
infile = os.path.join(Output_folder,case,filename)
imerg_total_precip = pd.read_pickle(infile)

filename = case + '_sorted_daily_precip_2001_2018.pkl'
infile = os.path.join(Output_folder,case,filename)
imerg_daily_precip = pd.read_pickle(infile)

#%% load station data
stations = pd.read_csv(os.path.join(CMA_folder,'China_Sta.csv'))
stations['lat'] = stations['lat'].apply(lambda x: float(x[:2])+int(x[3:5])/60.)
stations['long'] = stations['long'].apply(lambda x: float(x[:3])+int(x[4:6])/60.)
stations.rename(columns={"long": "lon"},inplace=True)
stations.drop(columns=['ID'],inplace=True)
#%% load daily precip from CMA
daily_precip = pd.read_csv(os.path.join(CMA_folder,'1949-2018_DailyPrecipitation.csv'),names=['CMAID','TyphoonID','StationID','date','daily_precip'])
daily_precip = daily_precip[daily_precip.daily_precip>threshold]
daily_precip = daily_precip[daily_precip.CMAID>200100]
daily_precip['time'] = pd.to_datetime(daily_precip.date)
daily_precip.drop(columns='date',inplace=True)
daily_precip = daily_precip.merge(stations[['StationID','lat','lon']], on='StationID')

# take out duplicate rows of (lat,lon,time)
idx = daily_precip[['StationID','time']].duplicated(keep='first')
uniques = daily_precip.loc[~idx].copy()
sorted_df = precipitation_utils.sort_precipitation_at_grid(uniques,'time', 'daily_precip', 0,'StationID')
cma_daily_preicp = sorted_df.loc[:,~pd.isna(sorted_df).all(axis=0)]
#%%
dummy = cma_daily_preicp.set_index('StationID')
cma_freq = dummy.isna().sum(axis=1).reset_index().rename(columns={0:'Counts'}).sort_values(by='Counts',ignore_index=True)
select_stations = cma_freq[cma_freq.Counts<=114]

#%% load CMA total precipitation data
total_precip = pd.read_csv(os.path.join(CMA_folder,'1949-2018_TotalPrecipitation.csv'), \
                           header=None,names=['SerialID','TCID','StationID','Total_Precip'])
total_precip['SerialID'] = total_precip['SerialID'].astype(int)

total_precip = total_precip[(total_precip.SerialID>200100)]

#filter out data points for which the total event precipitation is less than 10mm.
total_precip = total_precip[total_precip.Total_Precip>threshold]
# add lat lon to total_preicp 
total_precip = total_precip.merge(stations[['StationID','lat','lon']],on='StationID')
sorted_df = precipitation_utils.sort_precipitation_at_grid(total_precip, 'SerialID', 'Total_Precip',0, pivot_index='StationID')
cma_total_precip = sorted_df.loc[:,~pd.isna(sorted_df).all(axis=0)]


#%% find IMERG grids for CMA stations
imerg_lats = np.sort(imerg_total_precip.lat.unique())
imerg_lons = np.sort(imerg_total_precip.lon.unique())
for idx,row in stations.iterrows():
    diff_lat = -np.power(imerg_lats-row['lat'],2)
    diff_lon = -np.power(imerg_lons-row['lon'],2)
    lat_i = np.argmax(diff_lat)
    lon_i = np.argmax(diff_lon)

    stations.loc[idx,'imerg_lat1'] = imerg_lats[lat_i]
    stations.loc[idx,'imerg_lon1'] = imerg_lons[lon_i]
    
    if diff_lat[lat_i] == diff_lat[lat_i+1]:
        stations.loc[idx,'imerg_lat2'] = imerg_lats[lat_i+1]  
    if diff_lon[lon_i] == diff_lon[lon_i+1]:
        stations.loc[idx,'imerg_lon2'] = imerg_lats[lon_i+1]  
#%%
cma_daily_preicp = cma_daily_preicp.merge(stations[['StationID', 'StationName', 'lat', 'lon', 'AltitudeSensor', \
                                                    'imerg_lat1', 'imerg_lon1', 'imerg_lon2', 'imerg_lat2']],on='StationID')

cma_total_precip = cma_total_precip.merge(stations[['StationID', 'StationName', 'lat', 'lon', 'AltitudeSensor', \
                                                    'imerg_lat1', 'imerg_lon1', 'imerg_lon2', 'imerg_lat2']],on='StationID')

#%%
ii = 0
cma_row = cma_daily_preicp[cma_daily_preicp.StationID==select_stations['StationID'].iloc[ii]]
imerg_row = imerg_daily_precip[(imerg_daily_precip.lat==cma_row['imerg_lat1'].values[0]) & \
                               (imerg_daily_precip.lon==cma_row['imerg_lon1'].values[0])]
 
if pd.isna(cma_row['imerg_lon2'].values[0])==False:
    imerg_row2 = imerg_daily_precip[(imerg_daily_precip.lat==cma_row['imerg_lat1'].values[0]) & \
                                    (imerg_daily_precip.lon==cma_row['imerg_lon2'].values[0])]
    imerg_row = pd.concat([imerg_row,imerg_row2],axis=0)
    
if pd.isna(cma_row['imerg_lat2'].values[0])==False:
    imerg_row3 = imerg_daily_precip[(imerg_daily_precip.lat==cma_row['imerg_lat2'].values[0]) & \
                                    (imerg_daily_precip.lon==cma_row['imerg_lon1'].values[0])]
    imerg_row = pd.concat([imerg_row,imerg_row3],axis=0)

if (pd.isna(cma_row['imerg_lon2'].values[0])==False) & \
    (pd.isna(cma_row['imerg_lat2'].values[0])==False):
    imerg_row4 = imerg_daily_precip[(imerg_daily_precip.lat==cma_row['imerg_lat2'].values[0]) & \
                                    (imerg_daily_precip.lon==cma_row['imerg_lon2'].values[0])]
    imerg_row = pd.concat([imerg_row,imerg_row4],axis=0)


# cma data
aa_y = cma_row.values[0,1:-8]
aa_x = [x/18. for x in np.arange(1,len(aa_y)+1)]
# imerg data
bb_y = imerg_row.values[0,2:-1]
bb_x = [x/18. for x in np.arange(1,len(bb_y)+1)]
#%%
fig,ax = plt.subplots()
ax=plt.plot(aa_x,aa_y,'o',markersize=2)
ax=plt.plot(bb_x,bb_y,'o',markersize=2)
plt.legend(['CMA','IMERG'])
plt.title(cma_row.StationName.values[0])

#