# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:39:32 2021

@author: zhouluxi
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import glob
import rasterio
#import re
#import cartopy.crs as ccrs
from datetime import datetime, timedelta
from shapely.geometry import LineString, Point, Polygon, MultiPoint
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import CRS
from shapely.ops import cascaded_union
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
#import netCDF4 as nc
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
shapely.speedups.enabled

wkdir = '/home/lzhou/Precipitation/Precipitation_Scripts'
era5_folder = '/home/lzhou/Precipitation/Data/ERA5' 
cma_folder = '/media/lzhou/Extreme SSD/TC/CMA_Historical_Data/Wind_Rainfall'
Output_folder = r'/home/lzhou/Precipitation/Output'
Output_folder2 = os.path.join(wkdir,'Output')

#%% load total precipitation data
total_precip = pd.read_csv(os.path.join(cma_folder,'1949-2018_TotalPrecipitation.csv'), \
                header=None,names=['SerialID','TCID','StationID','Total_Precip'])
total_precip['SerialID'] = total_precip['SerialID'].astype(int)

#%% load station data
stations = pd.read_csv(os.path.join(cma_folder,'China_Sta.csv'))
stations['lat'] = stations['lat'].apply(lambda x: float(x[:2])+int(x[3:5])/60.)
stations['long'] = stations['long'].apply(lambda x: float(x[:3])+int(x[4:6])/60.)
stations.rename(columns={"long": "lon"},inplace=True)
#%% 201523 shanghai cma precip

cma_precip = total_precip[total_precip.SerialID==201523] 
cma_precip = cma_precip.merge(stations[['StationID','lat','lon']],on='StationID')

#%%
era5_file = os.path.join(era5_folder,'ERA5_Total_Precip_2015.nc')
#ds = nc4.Dataset(era5_file,'r')
#%%
data = xr.open_dataset(era5_file)
era5_daily = data.groupby('time.date').sum()    # get the daily sum (unit:m)
era5_daily['tp'] = era5_daily['tp']*1000.       # convert precipitation from m to mm.
era5_daily = era5_daily.rename({'tp':'precip'})
#%%
doy_offset = datetime(2014,12,31)
day1 = datetime(2015,9,30)
day2 = datetime(2015,10,7)
doy1 = (day1-doy_offset).days
doy2 = (day2-doy_offset).days
doys = np.arange(doy1,doy2+1)

#%% get track info :201523

cma_tracks = gpd.read_file(os.path.join(Output_folder2,'CMA_Best_Tracks_Nodes.shp'))
track = cma_tracks[cma_tracks.CMAID==201523].copy()
track['datetime'] = pd.to_datetime(track[['Year','Month','Day','Hour']])
track = track.reset_index(drop=True)
for i in np.arange(0,len(track)):
    track.loc[i,'doy']=(track.loc[i,'datetime']-doy_offset).days

track_proj = track.to_crs(2345)
#plt.figure()
first_day=1

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape = world[world.name=='China'].copy()
cn_shape.reset_index(drop=True,inplace=True)
#%%
# plt.figure()

# for i in np.arange(1,len(doys)+1):
#     print(i)
#     ax=plt.subplot(2,4,i)
#     era5_daily['precip'][doys[i-1]-1,:,:].plot(ax=ax)



#%%
plt.figure()
for i in np.arange(1,len(doys)+1):
    print(i)
    ax=plt.subplot(2,4,i)
    track_doys = track.doy.to_list()
    if (doys[i-1] in track_doys):
        track[track.doy==doys[i-1]].plot(ax=ax,color='r',markersize=4)

        buffer = track_proj[track_proj.doy==doys[i-1]].buffer(1500000)
    else:
        if first_day==1:
            track[track.doy==doys[0]].plot(ax=ax,color='r',markersize=4)

            buffer = track_proj[track_proj.doy==track_proj.doy.min()].buffer(1500000)
        else:
            track[track.doy==track.doy.max()].plot(ax=ax,color='r',markersize=4)

            buffer = track_proj[track_proj.doy==track_proj.doy.max()].buffer(1500000)
  
    polygons = buffer.geometry
    bounds = gpd.GeoSeries(cascaded_union(polygons),crs=2345)
    bounds = bounds.to_crs(4326)
    bounds.boundary.plot(ax=ax)
    aa = era5_daily['precip'][doys[i-1]-1,:,:].to_dataframe()
    aa = aa.reset_index().drop(columns='date')
    aa['geometry'] = aa.apply(lambda x: Point((float(x.longitude),float(x.latitude))),axis=1)
    
    aa = gpd.GeoDataFrame(aa,geometry='geometry',crs="epsg:4326")
  
    idx1 = aa.geometry.apply(lambda x: x.within(bounds[0]))
    bb = aa.loc[idx1]
    cc = bb.dropna()

    cc = gpd.GeoDataFrame(cc,geometry='geometry',crs='epsg:4326')


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    cmax = cc.precip.mean() + 3*cc.precip.std()
    cc.plot(column='precip', ax=ax, vmin=0,vmax=cmax, cmap='Blues',cax=cax, \
                legend=True, legend_kwds={'label': "Total Precip. (mm)"})
    plt.text(0.3, 0.9, 'Max. total precip = %d mm'%cc.precip.max(), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    cn_shape.boundary.plot(color='black',ax=ax)  
    track.plot(ax=ax,markersize=2)
    ax.plot(cma_precip['lon'],cma_precip['lat'],linestyle='',marker='.',markersize=4,color='k')

    if first_day==1:
        dd = cc.copy()
        first_day=0
    else:
        dd = pd.merge(dd,cc,right_index=True,left_index=True,how='outer')
        dd = dd[['precip_x','precip_y']]
        dd = dd.fillna(0)
        dd['precip'] = dd['precip_x']+dd['precip_y']
        dd = dd[['precip']]
#%%
# ee = pd.merge(dd,aa[['geometry','latitude','longitude']],right_index=True,left_index=True,how='inner')
# ee = gpd.GeoDataFrame(ee,geometry='geometry',crs='epsg:4326')

# #%% load China boundary
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# cn_shape = world[world.name=='China'].copy()
# cn_shape.reset_index(drop=True,inplace=True)


# fig,ax = plt.subplots(1,1)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.01)
# cmax = ee.precip.mean() + ee.precip.std()
# ee.plot(column='precip', ax=ax, cmap='Blues',cax=cax, \
#         legend=True, legend_kwds={'label': "Total Precip. (mm)"})
# cn_shape.boundary.plot(color='black',ax=ax)
# #%%
# df_wgs.plot(ax=ax,markersize=2,color='red')
# ax.title.set_text(df.Name.iloc[0] + ' ' + days[0] + '-' + days[-1])
# ax.set_xlim([70,140])
# ax.set_ylim([15,55])    
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')                              

    

