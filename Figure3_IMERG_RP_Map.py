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

#import precipitation_utils 


IMERG_folder = '/media/lzhou/Extreme SSD/Precipitation/IMERG'
CMA_folder = '/home/lzhou/Precipitation/Data/CMA_Historical_Data/Wind_Rainfall/'
Output_folder = '/home/lzhou/Precipitation/Output'
Output_folder2 = '/home/lzhou/Precipitation/Precipitation_Scripts/Output/'
Figure_folder = os.path.join(Output_folder,'Figures')

case = 'IMERG_10degree_12'
item = 'precipitationCal'
threshold = 10.

#%% Get the domain of greater China
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape = world[world.name=='China'].copy()
cn_shape.reset_index(drop=True,inplace=True)
tw_shape = world[world.name=='Taiwan'].copy()
tw_shape.reset_index(drop=True,inplace=True)
greater_china = cn_shape.geometry.union(tw_shape.geometry)

filename = item+'_event_total_precip.pkl' 
infile = os.path.join(Output_folder,case,filename)
event_total = pd.read_pickle(infile)
event_total = event_total[event_total.precip>=threshold].reset_index(drop=True)


    #%%

    
# Number of measurements of daily precipitation at each shanghai data stations:
daily_station_freq = daily_precip.StationID.value_counts().to_frame().reset_index().rename(columns={'index':'StationID','StationID':'Count'})
daily_station_freq = daily_station_freq.merge(stations,on='StationID')
daily_station_freq = gpd.GeoDataFrame(daily_station_freq, geometry=gpd.points_from_xy(daily_station_freq.lon, daily_station_freq.lat), \
                                crs="epsg:4326")
