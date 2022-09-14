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
from hazard import GPD
from scipy.stats import genpareto, scoreatpercentile


#os.chdir('/home/lzhou/Precipitation/Precipitation_Scripts')

#IMERG_folder = '/media/lzhou/Extreme SSD/Precipitation/IMERG'
#CMA_folder = '/home/lzhou/Precipitation/Data/CMA_Historical_Data/Wind_Rainfall/'
#Output_folder = '/home/lzhou/Precipitation/Output'
#Output_folder2 = '/home/lzhou/Precipitation/Precipitation_Scripts/Output/'


IMERG_folder = r'D:\Precipitation\IMERG'
CMA_folder = r'D:\Precipitation\CMA_Historical_Data'
Output_folder = r'D:\Precipitation\Output'
Output_folder2 = r'D:\Precipitation\Precipitation_Scripts\Output'
Figure_folder = os.path.join(Output_folder,'Figures')


case = 'IMERG_10degree_12'
item = 'precipitationCal'
threshold = 10.

# plots configuration
font = {'family' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)

#%% load IMERG data

filename = case + '_sorted_total_precip_2001_2020.pkl'
infile = os.path.join(Output_folder,case,filename)
imerg_total_precip = pd.read_pickle(infile)
imerg_total_precip.drop(columns='geometry',inplace=True)

filename = case + '_sorted_daily_precip_2001_2020.pkl'
infile = os.path.join(Output_folder,case,filename)
imerg_daily_precip = pd.read_pickle(infile)
imerg_daily_precip.drop(columns='geometry',inplace=True)


#%% load station data
stations = pd.read_csv(os.path.join(CMA_folder,'Wind_Rainfall','China_Sta.csv'))
stations['lat'] = stations['lat'].apply(lambda x: float(x[:2])+int(x[3:5])/60.)
stations['long'] = stations['long'].apply(lambda x: float(x[:3])+int(x[4:6])/60.)
stations.rename(columns={"long": "lon"},inplace=True)
stations.drop(columns=['ID'],inplace=True)

#%% find IMERG grids for CMA stations
imerg_lats = np.sort(imerg_daily_precip.lat.unique())
imerg_lons = np.sort(imerg_daily_precip.lon.unique())

#imerg_lats1 = np.sort(imerg_total_precip.lat.unique())
#imerg_lons1 = np.sort(imerg_total_precip.lon.unique())

for idx,row in stations.iterrows():
    #print(row)
    #sid = row['StationI#total_precip.plot.scatter(x='Total_Precip',y='imerg_precip',hue='k',ax=ax)
    #lat = stations.lat.iloc[0]
    #lon = stations.lon.iloc[0]
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
        
#%% load daily precip from CMA
daily_precip = pd.read_csv(os.path.join(CMA_folder,'Wind_Rainfall','1949-2018_DailyPrecipitation.csv'),names=['CMAID','TyphoonID','StationID','date','daily_precip'])
daily_precip = daily_precip[daily_precip.daily_precip>threshold]
daily_precip = daily_precip[daily_precip.CMAID>200100]
daily_precip['time'] = pd.to_datetime(daily_precip.date)
daily_precip.drop(columns='date',inplace=True)

# take out duplicate rows of (lat,lon,time)
idx = daily_precip[['StationID','time']].duplicated(keep='first')
uniques = daily_precip.loc[~idx].copy()
uniques.drop(columns=['TyphoonID','CMAID'],inplace=True)
sorted_df = precipitation_utils.sort_precipitation_at_grid(uniques,'time', 'daily_precip', 0,'StationID')
cma_daily_preicp = sorted_df.loc[:,~pd.isna(sorted_df).all(axis=0)]
cma_daily_preicp.set_index('StationID',inplace=True)
#%% choose stations that have at least 100 data points, which yield 16 stations
cma_freq = uniques.value_counts('StationID').to_frame(name='Count').reset_index()
select_stations = cma_freq[cma_freq.Count>=100]
select_stations = select_stations.merge(stations,on='StationID')
#%%

imerg_daily_precip.set_index(['lat','lon'],inplace=True)
#%%
cma_years = np.floor(18*365.25)
imerg_years = np.floor(20*365.25)
#%%
ii = 1
sid = select_stations.StationID.loc[ii]
cma_data = cma_daily_preicp.loc[sid].dropna().values
#%%

imerg_lat = select_stations.imerg_lat1.loc[ii]
imerg_lon = select_stations.imerg_lon1.loc[ii]
imerg_data = imerg_daily_precip.loc[imerg_lat,imerg_lon].dropna().values

data = imerg_data.copy()
years = imerg_years
rate = len(data)/years
shape,location,scale=genpareto.fit(data)
mu=data.mean()
Rpeval = GPD.gpdReturnLevel([10,20], mu, shape, scale, rate)




























#%%
if ~pd.isnull(select_stations.imerg_lat2.loc[ii]):
    imerg_lat2 = select_stations.imerg_lat2.loc[ii]
    imerg_data1 = imerg_daily_precip.loc[imerg_lat2,imerg_lon].dropna().values
    
#%%




























#%%
merge1 = imerg_daily_precip.merge(select_stations,right_on=['imerg_lat1','imerg_lon1'],left_on=['lat','lon'],how='right')
merge1.drop(columns=['lat_x','lon_x'],inplace=True)
merge1.rename(columns={'lat_y':'orig_lat','lon_y':'orig_y'},inplace=True)

merge2 = merge1.merge(imerg_daily_precip,left_on=['imerg_lat1','imerg_lon2'],right_on=['lat','lon'],how='left')
merge2.drop(columns=['lat','lon','geometry_y'],inplace=True)
merge2.rename(columns={'geometry_x':'geometry'},inplace=True)
#%%
merge3 = merge2.merge(imerg_daily_precip,left_on=['imerg_lat2','imerg_lon1'],right_on=['lat','lon'],how='left')
merge3.drop(columns=['lat','lon','geometry_y'],inplace=True)
merge3.rename(columns={'geometry_x':'geometry'},inplace=True)
#%%
merge4 = merge3.merge(imerg_daily_precip,left_on=['imerg_lat2','imerg_lon2'],right_on=['lat','lon'],how='left')
merge4.drop(columns=['lat','lon','geometry_y'],inplace=True)
merge4.rename(columns={'geometry_x':'geometry'},inplace=True)
#%%
total_precip['imerg_precip'] = merge4[['precip_1','precip_2','precip_3','precip_4']].mean(axis=1)

imerg_daily_precip

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