#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:00:15 2021

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
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

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



#%% set parameters
#zone = 1000000.      # buffer zone around tracks, to be consistent with input nodes/track file

item = 'precipitationCal'

zone = 15
pre_days = 1        # number of days before the node's time to include precipitation   
post_days = 2       # number of days after hte node's time to include precipitation

proj_epsg = 2345    # projeced CRS 
init_epsg = 4326    # geographical CRS
#%%
IMERG_folder = '/media/lzhou/Extreme SSD/Precipitation/IMERG'
CMA_folder = '/home/lzhou/Precipitation/Data/CMA_Historical_Data/Wind_Rainfall/'
Output_folder = '/home/lzhou/Precipitation/Output'
Output_folder2 = '/home/lzhou/Precipitation/Precipitation_Scripts/Output/'
case = 'IMERG_10degree_12'
output_path = os.path.join(Output_folder,case)
'''
#%% Get the domain of greater China
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape = world[world.name=='China'].copy()
cn_shape.reset_index(drop=True,inplace=True)
tw_shape = world[world.name=='Taiwan'].copy()
tw_shape.reset_index(drop=True,inplace=True)
greater_china = cn_shape.geometry.union(tw_shape.geometry)
'''
#%%
cma_track_file = 'CMA_Tracks_Nodes_'+str(zone)+'degree.shp'
nodes = gpd.read_file(os.path.join(Output_folder2,cma_track_file))
nodes['Time'] = pd.to_datetime(nodes[['Year','Month','Day']])  
#cmaids = nodes[(nodes.CMAID<201409)&(nodes.CMAID>201407)].CMAID.unique()
#cmaids = nodes[nodes.CMAID>200004].CMAID.unique()
'''
The function below has been moved to precipitation_utils.py

def extract_IMERG_precipitation(df,item='precipitationCal',output_path=None):
    datasets = []
    for idx, data in df.groupby('Time'):
        
        # get bound of points in a day
        buffer_zone = data.buffer(zone)
        polygons = buffer_zone.geometry
        bounds = gpd.GeoSeries(cascaded_union(polygons))
        bounds.crs = CRS.from_epsg(init_epsg)
        #minlon, minlat, maxlon, maxlat = bounds.geometry.total_bounds
        
        # prepare input precipitation data
        datestr = idx.strftime("%Y%m%d")
        filename = "3B-DAY.MS.MRG.3IMERG."+datestr+"-S000000-E235959.V06.nc4"
        infile = os.path.join(IMERG_folder, filename)
        da = xr.load_dataset(infile,decode_coords="all")
        ds = xr.DataArray(da[item][0,:,:].values,coords=[da.lon,da.lat],dims=['x','y'])
        #ds = xr.DataArray(da.precipitationCal.values,coords=[da.time,da.lon,da.lat],dims=['time','x','y'])

        ds.rio.set_crs(init_epsg,inplace=True) 
        # clip part of the precipitation data by the bound
        daily_precip = ds.rio.clip(bounds,init_epsg).transpose()
        datasets.append(daily_precip)
        
    event_precip = xr.concat(datasets,dim='time')
    event_precip = event_precip.assign_coords({"time":df.Time.unique()})
    event_precip.attrs['CMAID']=df.CMAID.iloc[0]

    if isinstance(output_path,str):
        if os.path.exists(output_path)==False:
            os.makedirs(output_path)
        filename = item+'_'+str(df.CMAID.iloc[0])+'.nc'
        outfile = os.path.join(output_path,filename)
        event_precip.to_netcdf(outfile)
        
    return event_precip

'''

for yy in np.arange(2010,2021):
    if yy == 2000:
        yearly_nodes = nodes[(nodes.Year==yy)&(nodes.CMAID>200004)]
    else:
        yearly_nodes = nodes[nodes.Year==yy]

    cmaids = yearly_nodes.CMAID.unique()

    # Extract precipitation data
    first_call = 1
    df = yearly_nodes[yearly_nodes.CMAID==cmaids[0]]
    event_precip = precipitation_utils.extract_IMERG_precipitation(df,zone,IMERG_folder,item=item,output_path=output_path)
    for tt in np.arange(0,len(event_precip)):
        daily_da = event_precip[tt,:,:].to_dataframe(name='precip').drop(columns=['spatial_ref'])
        daily_da['CMAID'] = cmaids[0]
        if first_call == 1:
            first_call = 0 
            whole_year_precip = daily_da.copy()
        else:
            whole_year_precip = pd.concat([whole_year_precip,daily_da],axis=0)

    for cmaid in cmaids[1:]:
        print(cmaid)
        df = yearly_nodes[yearly_nodes.CMAID==cmaid]
        event_precip = precipitation_utils.extract_IMERG_precipitation(df,zone,IMERG_folder,item=item,output_path=output_path)
        for tt in np.arange(0,len(event_precip)):
            daily_da = event_precip[tt,:,:].to_dataframe(name='precip').drop(columns=['spatial_ref'])
            daily_da['CMAID'] = cmaid
            whole_year_precip = pd.concat([whole_year_precip,daily_da],axis=0)
    
    # Save whole year data to pickle 
    filename = item+'_'+str(yy)+'.pkl'
    outfile = os.path.join(output_path,filename)
    whole_year_precip.to_pickle(outfile)
