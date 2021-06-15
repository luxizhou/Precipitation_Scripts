# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import os
import glob
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

Shanghai_folder = r'D:\Precipitation\CMA_Historical_Data\Wind_Rainfall'

#%% load total precipitation data
total_precip = pd.read_csv(os.path.join(Shanghai_folder,'1949-2018_TotalPrecipitation.csv'), \
                header=None,names=['SerialID','TCID','StationID','Total_Precip'])
total_precip['SerialID'] = total_precip['SerialID'].astype(int)

#%% load station data
stations = pd.read_csv(os.path.join(Shanghai_folder,'China_Sta.csv'))
stations['lat'] = stations['lat'].apply(lambda x: float(x[:2])+int(x[3:5])/60.)
stations['long'] = stations['long'].apply(lambda x: float(x[:3])+int(x[4:6])/60.)
stations.rename(columns={"long": "lon"},inplace=True)

#%%
# Number of measurements of total precipitation at each shanghai data stations:
station_freq = total_precip.StationID.value_counts().to_frame().reset_index().rename(columns={'index':'StationID','StationID':'Count'})
station_freq = station_freq.merge(stations,on='StationID')
#station_freq = gpd.GeoDataFrame(station_freq,geometry='geometry')
#m=station_freq.plot(column='Count',legend=True)
#cn_shape.boundary.plot(ax=m)
#plt.title('Number of data points per station')
#%%
