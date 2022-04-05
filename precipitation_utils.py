#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:11:17 2022

@author: lzhou
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pyproj import CRS
from shapely.ops import cascaded_union
import seaborn as sns; sns.set_theme()

def extract_IMERG_precipitation(df,zone,IMERG_folder,init_epsg=4326,item='precipitationCal',output_path=None):

    '''
    This function extract precipitation data from IMERG nc files based on input 
    geopandas dataframe df, which consist points.
    '''

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

def sort_precipitation_at_grid(df,col_name,val_name,cols_to_sort,pivot_index=['lat','lon']):
    df_pivot = df.pivot(index=pivot_index,columns=col_name,values=val_name)#.reset_index()
    # sort precipitation in each grid
    data = df_pivot.to_numpy()
    data1 = np.sort(-data[:,cols_to_sort:],axis=1) #only consider data since 2001 as IMERG is only availabe for part of year 2020
    data2 = -data1
    sorted_df = pd.DataFrame(data2,index=df_pivot.index)
    sorted_df = sorted_df.reset_index()
    
    return sorted_df

