#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:00:15 2021

@author: lzhou
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import CRS
from shapely.ops import cascaded_union
import seaborn as sns; sns.set_theme()
#import matplotlib.pyplot as plt
#import netCDF4 as nc
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
shapely.speedups.enabled

#%% set parameters
zone = 1500000.      # buffer zone around tracks, to be consistent with input nodes/track file
pre_days = 1        # number of days before the node's time to include precipitation   
post_days = 2       # number of days after hte node's time to include precipitation

proj_epsg = 2345    # projeced CRS 
init_epsg = 4326    # geographical CRS
#%%
IMERG_folder = '/media/lzhou/Extreme SSD/Precipitation/IMERG'
CMA_folder = '/home/lzhou/Precipitation/TC_Tracks/CMA_Historical_Data/Wind_Rainfall/'
Output_folder = '/home/lzhou/Precipitation/Output'
Output_folder2 = '/home/lzhou/Precipitation/Precipitation_Scripts/Output/'
#%% Make coordinates for IMERG data
# find the extent of data 
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape = world[world.name=='China'].copy()
cn_shape.reset_index(drop=True,inplace=True)

#%%
cma_track_file = 'CMA_Tracks_Nodes_'+str(int(zone/1000))+'km.shp'
nodes = gpd.read_file(os.path.join(Output_folder2,cma_track_file))
nodes['Time'] = pd.to_datetime(nodes[['Year','Month','Day']])  
#cmaids = nodes[(nodes.CMAID<201409)&(nodes.CMAID>201407)].CMAID.unique()
cmaids = nodes[nodes.CMAID>200004].CMAID.unique()
#%% example HATO
#idx = nodes.index[(nodes.Year==2017)&(nodes.Name=='HATO')]
for cmaid in cmaids:
    print(cmaid)
    df = nodes[nodes.CMAID==cmaid].copy()
    df_proj = df.to_crs(proj_epsg)
    # get a rough extent of impact 
    buffer_zone = df_proj.buffer(zone)
    polygons = buffer_zone.geometry
    bounds = gpd.GeoSeries(cascaded_union(polygons))
    bounds.crs = CRS.from_epsg(proj_epsg)
    bounds_wgs = bounds.to_crs(epsg=init_epsg)
        
    # prepare coordinates
    minlon, minlat, maxlon, maxlat = bounds_wgs.geometry.total_bounds
    lons1 = np.arange(-179.95,180,0.1)
    lats1 = np.arange(-89.95,90,0.1)
    idlon2 = int((maxlon-lons1[0])/0.1)
    idlon1 = int((minlon-lons1[0])/0.1)
    idlat2 = int((maxlat-lats1[0])/0.1)
    idlat1 = int((minlat-lats1[0])/0.1)
    latitude = lats1[idlat1:idlat2+1]
    longitude = lons1[idlon1:idlon2+1]
    lat,lon = np.meshgrid(latitude,longitude)
    lon1d = np.reshape(lon,-1)
    lat1d = np.reshape(lat,-1)
    
    dummy = np.array([lat1d, lon1d])
    coords = pd.DataFrame(dummy.T, columns=['lat', 'lon'])
    coords['geometry'] = coords.apply(lambda x: Point((float(x.lon),float(x.lat))),axis=1)
    coords = gpd.GeoDataFrame(coords,geometry='geometry',crs="epsg:4326")
    
    # get the range of dates for extracting precipitation
    day_init = df.Time.iloc[0]-pd.DateOffset(days=pre_days)
    day_end = df.Time.iloc[-1]+pd.DateOffset(days=post_days+1)
    tt = np.arange(day_init,day_end, timedelta(days=1))
    days = [x.item().strftime("%Y%m%d") for x in tt]
    #print('days: ',days)
    
#%%    
    for dd in days:
        coords[dd]=False
        
    # Extract precipitation grids
    grouped = df_proj.groupby('Time')
    for name, group in grouped:
        #print(name)
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
    #%%
    # Extract precipitation
    first_call = 1
    for ii in days:
        filename = "3B-DAY.MS.MRG.3IMERG."+ii+"-S000000-E235959.V06.nc4"
        infile = os.path.join(IMERG_folder, filename)
        ds = xr.open_dataset(infile)
        
        #print(infile, ' ', os.path.isfile(infile))
        #ds = nc.Dataset(infile, 'r')
        precip = ds['precipitationCal'][0, idlon1:idlon2+1, idlat1:idlat2+1]
        dummy1 = precip.to_dataframe().reset_index()
        dummy1['time'] = ii
        dummy1['lat'] = lat1d
        dummy1['lon'] = lon1d
        dummy1 = dummy1.rename(columns={'precipitationCal':'precip'})
        #dummy = np.array([lat1d, lon1d, precip1d])
        #dummy1 = pd.DataFrame(dummy.T, columns=['lat', 'lon', 'precip'])
        #dummy1['Time'] = ii
        dummy2 = dummy1[coords[ii]]
        if first_call == 1:
            precip_daily = dummy2.copy()
            first_call = 0
        else: 
            precip_daily = precip_daily.append(dummy2)   
            
    precip_total = precip_daily.groupby(['lat','lon'])['precip'].sum().to_frame().reset_index()
    #%%
    daily_file = str(cmaid) + '_' + str(df.Year.iloc[0]) + '_' + df.Name.iloc[0] + \
                 '_daily_precip_IMERG_' + str(int(zone/1000))+'km_' + \
                 str(int(pre_days)) + str(int(post_days))+'.pkl'
    total_file = str(cmaid) + '_' + str(df.Year.iloc[0]) + '_' + df.Name.iloc[0] + \
                 '_total_precip_IMERG_' + str(int(zone/1000))+'km_' + \
                 str(int(pre_days))+str(int(post_days))+'.pkl'
    
    precip_daily.to_pickle(os.path.join(Output_folder, 'IMERG_1500km_12', daily_file))
    precip_total.to_pickle(os.path.join(Output_folder, 'IMERG_1500km_12', total_file))
    
    #%% Plotting
    #df_wgs = df.to_crs(epsg=init_epsg)
    # convert dataframe to geodataframe for plotting
    
    precip_total = gpd.GeoDataFrame(precip_total, \
                                    geometry=gpd.points_from_xy(precip_total.lon, precip_total.lat), \
                                    crs="epsg:4326")
    fig,ax = plt.subplots(1,1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    cmax = precip_total.precip.mean() + 3*precip_total.precip.std()
    precip_total.plot(column='precip', ax=ax, \
    #                    cmap='Blues', scheme='quantiles', \
                      cmap='Blues', vmin=0,vmax=cmax, cax=cax, \
    #                  cmap='Blues', scheme='naturalbreaks', cax=cax, \
                      legend=True, legend_kwds={'label': "Total Precip. (mm)"})
    cn_shape.boundary.plot(color='black',ax=ax)
    df.plot(ax=ax,markersize=2,color='red')
    plt.text(0.3, 0.9, 'Max. total precip = %d mm'%precip_total.precip.max(), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.title.set_text('IMERG ' + df.Name.iloc[0] + ' ' + days[0] + '-' + days[-1])
    ax.set_xlim([70,140])
    ax.set_ylim([15,55])    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')                              
    figname = str(cmaid) + '_' + str(df.Year.iloc[0]) + '_' + df.Name.iloc[0] + \
              '_total_precip_IMERG_' + str(int(zone/1000)) + 'km_' + \
              str(int(pre_days)) + str(int(post_days)) + '.png'
    
    fig.savefig(os.path.join(Output_folder,'IMERG_1500km_12', figname))
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
