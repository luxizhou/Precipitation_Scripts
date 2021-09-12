#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 19:42:06 2021

@author: lzhou
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from shapely.geometry import LineString, Point, Polygon, MultiPoint
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import CRS
from shapely.ops import cascaded_union
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import netCDF4 as nc
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
shapely.speedups.enabled

#%% set parameters
zone = 1000000.      # buffer zone around tracks, to be consistent with input nodes/track file
pre_days = 1        # number of days before the node's time to include precipitation   
post_days = 2       # number of days after hte node's time to include precipitation

proj_epsg = 2345    # projeced CRS 
init_epsg = 4326    # geographical CRS
#%%

ERA5_folder = '/media/lzhou/Extreme SSD/Precipitation/ERA5'
CMA_folder = '/home/lzhou/Precipitation/TC_Tracks/CMA_Historical_Data/Wind_Rainfall/'
Output_folder = '/home/lzhou/Precipitation/Output'
Output_folder2 = '/home/lzhou/Precipitation/Precipitation_Scripts/Output/'

#%% load China boundary
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape = world[world.name=='China'].copy()
cn_shape.reset_index(drop=True,inplace=True)

#%% load track file
cma_track_file = 'CMA_Tracks_Nodes_'+str(int(zone/1000))+'km.shp'
nodes = gpd.read_file(os.path.join(Output_folder2,cma_track_file))
nodes['Time'] = pd.to_datetime(nodes[['Year','Month','Day']]) 

#%% read in era5 data
files = os.listdir(ERA5_folder)
era5_files = [x for x in files if 'nc' in x]
era5_files.sort()
for era5_file in era5_files[:35]:
    print(era5_file)
    era5_file = os.path.join(ERA5_folder,era5_file)
    year = int(era5_file[-7:-3])
    doy_offset = datetime(year-1,12,31)
    
    data = xr.open_dataset(era5_file) # read hourly precipitation from ERA5 file (unit:m)
    era5_daily = data.groupby('time.date').sum()    # get the daily sum (unit:m)
    era5_daily['tp'] = era5_daily['tp']*1000.       # convert precipitation from m to mm.
    era5_daily = era5_daily.rename({'tp':'precip'})

    # get the coordiantes of era5 data
    #era_latmax = era5_daily.coords['latitude'].values.max()
    #era_lonmax = era5_daily.coords['longitude'].values.max()
    #era_latmin = era5_daily.coords['latitude'].values.min()
    #era_lonmin = era5_daily.coords['longitude'].values.min()
    
    era_lats = era5_daily.latitude.values
    era_lons = era5_daily.longitude.values
    era_dates = era5_daily.date.values

    cmaids = nodes[(nodes.CMAID<(year+1)*100)&(nodes.CMAID>=year*100)].CMAID.unique()
    #cmaids = nodes[nodes.CMAID<200101].CMAID.unique()
    #cmaids = [201805]

    for cmaid in cmaids:
        print(cmaid)
        df = nodes[nodes.CMAID==cmaid].copy()
        # get a rough extent of impact 
        buffer_zone = df.buffer(zone)
        polygons = buffer_zone.geometry
        bounds = gpd.GeoSeries(cascaded_union(polygons))
        bounds.crs = CRS.from_epsg(proj_epsg)
        bounds_wgs = bounds.to_crs(epsg=init_epsg)
    
        # prepare coordinates
        minlon, minlat, maxlon, maxlat = bounds_wgs.geometry.total_bounds
        idlon1 = np.argmin(abs(era_lons-minlon))
        idlon2 = np.argmin(abs(era_lons-maxlon))
        idlat1 = np.argmin(abs(era_lats-maxlat))
        idlat2 = np.argmin(abs(era_lats-minlat))
        dummy = era5_daily[dict(date=1, latitude=np.arange(idlat1,idlat2+1), \
                                longitude=np.arange(idlon1,idlon2+1))]
        coords = dummy.to_dataframe().reset_index().drop(columns=['precip','date'])
        coords['geometry'] = coords.apply(lambda x: Point((float(x.longitude),float(x.latitude))),axis=1)
        coords = gpd.GeoDataFrame(coords,geometry='geometry',crs="epsg:4326")
  
        # get the range of dates for extracting precipitation
        day_init = df.Time.iloc[0]-pd.DateOffset(days=pre_days)
        day_end = df.Time.iloc[-1]+pd.DateOffset(days=post_days+1)
        tt = np.arange(day_init,day_end, timedelta(days=1))
        days = [x.item().strftime("%Y%m%d") for x in tt]
        doy1 = (day_init-doy_offset).days
        doy2 = (day_end-doy_offset).days
        doys = np.arange(doy1,doy2)-1

        for dd in days:
            coords[dd]=False
        
        # Extract precipitation grids
        grouped = df.groupby('Time')
        for name, group in grouped:
            buffer1 = group.buffer(zone)
            polygons = buffer1.geometry
            bounds1 = gpd.GeoSeries(cascaded_union(polygons))
            bounds1.crs = CRS.from_epsg(proj_epsg)
            bounds1_wgs = bounds1.to_crs(epsg=init_epsg)
    
            idx1 = coords.geometry.apply(lambda x: x.within(bounds1_wgs[0]))
            d1 = group.Time.iloc[0]-pd.DateOffset(days=pre_days)
            d2 = group.Time.iloc[0]+pd.DateOffset(days=post_days+1)
            tt1 = np.arange(d1,d2, timedelta(days=1))
            tts = [x.item().strftime("%Y%m%d") for x in tt1]

            #tt1 = int(group.Time.iloc[0].strftime("%Y%m%d"))
            #tts = [str(x) for x in np.arange(tt-pre_days,tt+post_days+1)]
            #print(tts)
            # Call a function to get a bound of all track nodes
            # Call a function in extracting precipitation
            for dd in tts:
                coords.loc[idx1,dd]=True
    
        # Extract precipitation
        first_call = 1
    
        for (ii,jj) in zip(days,doys):

            #print(ii,jj)
            precip = era5_daily[dict(date=jj, latitude=np.arange(idlat1,idlat2+1), \
                                     longitude=np.arange(idlon1,idlon2+1))]
            dummy1 = precip.to_dataframe().reset_index().drop(columns=['date'])
            dummy1['Time'] = ii
            dummy2 = dummy1[coords[ii]]
            if first_call == 1:
                precip_daily = dummy2.copy()
                first_call = 0
            else: 
                precip_daily = precip_daily.append(dummy2)   
            
            precip_total = precip_daily.groupby(['latitude','longitude'])['precip'].sum().to_frame().reset_index()
        #%%
        daily_file = str(cmaid) + '_' + str(df.Year.iloc[0]) + '_' + \
            df.Name.iloc[0] + '_daily_precip_ERA5_' + \
            str(int(zone/1000))+'km_' + str(int(pre_days)) + \
            str(int(post_days))+'.pkl'
        total_file = str(cmaid) + '_' + str(df.Year.iloc[0]) + '_' + \
            df.Name.iloc[0] + '_total_precip_ERA5_' + \
            str(int(zone/1000))+'km_' + str(int(pre_days))+str(int(post_days))+'.pkl'
    
        precip_daily.to_pickle(os.path.join(Output_folder, 'ERA5', daily_file))
        precip_total.to_pickle(os.path.join(Output_folder, 'ERA5', total_file))
    
        #%% Plotting
        df_wgs = df.to_crs(epsg=init_epsg)
        # convert dataframe to geodataframe for plotting
    
        precip_total = gpd.GeoDataFrame(precip_total, \
                                        geometry=gpd.points_from_xy(precip_total.longitude, precip_total.latitude), \
                                        crs="epsg:4326")
        fig,ax = plt.subplots(1,1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.01)
        cmax = precip_total.precip.mean() + precip_total.precip.std()
        precip_total.plot(column='precip', ax=ax, \
    #                    cmap='Blues', scheme='quantiles', \
                          cmap='Blues', vmin=0,vmax=cmax, cax=cax, \
    #                  cmap='Blues', scheme='naturalbreaks', cax=cax, \
                          legend=True, legend_kwds={'label': "Total Precip. (mm)"})
        cn_shape.boundary.plot(color='black',ax=ax)
        df_wgs.plot(ax=ax,markersize=2,color='red')
        ax.title.set_text(df.Name.iloc[0] + ' ' + days[0] + '-' + days[-1])
        ax.set_xlim([70,140])
        ax.set_ylim([15,55])    
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')                              
        figname = str(cmaid) + '_' + str(df.Year.iloc[0]) + '_' + \
            df.Name.iloc[0] + '_total_precip_ERA5_' + str(int(zone/1000)) + \
            'km_' + str(int(pre_days)) + str(int(post_days)) + '.png'
    
        fig.savefig(os.path.join(Output_folder,'ERA5', figname))
        plt.close(fig)





# precip_daily = gpd.GeoDataFrame(precip_daily, \
#                                 geometry=gpd.points_from_xy(precip_daily.lon, precip_daily.lat), \
#                                 crs="epsg:4326")
    
# fig, axs = plt.subplots(2,4,figsize=(12,8))
# #fig.subplots_adjust(hspace =0.1, wspace=0.1)
# axs = axs.ravel()
# jj = 0
# grouped = precip_daily.groupby('Time')
# for name, group in grouped:
#     #print(name)
#     m = axs[jj]
#     group.plot(column='precip',cmap='Blues',vmin=0,vmax=200,ax=m)
#     cn_shape.boundary.plot(color='black',ax=m)
#     df_wgs.plot(ax=m,markersize=2,color='red')
#     m.title.set_text(name)
#     m.set_xlim([70,140])
#     m.set_ylim([15,55])


#     jj = jj+1  
# fig.savefig(os.path.join(Output_folder,'2017_Hato_IMERG_extract.png'))                              
#%%


#3B-DAY.MS.MRG.3IMERG.20170820-S000000-E235959.V06.nc4