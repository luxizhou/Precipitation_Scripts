#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 11:18:40 2022

@author: lzhou
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:00:15 2021

@author: lzhou
"""
import os
#import pandas as pd
#import xarray as xr
#from datetime import timedelta
#from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
#from pyproj import CRS
#from shapely.ops import cascaded_union
import seaborn as sns; sns.set_theme(style='white')
#import matplotlib.pyplot as plt
#import netCDF4 as nc
import shapely
#from mpl_toolkits.axes_grid1 import make_axes_locatable
shapely.speedups.enabled

#%% set parameters
#zone = 1000000.      # buffer zone around tracks, to be consistent with input nodes/track file
#pre_days = 1        # number of days before the node's time to include precipitation   
#post_days = 2       # number of days after hte node's time to include precipitation

proj_epsg = 2345    # projeced CRS 
init_epsg = 4326    # geographical CRS
#%%
IMERG_folder = '/media/lzhou/Extreme SSD/Precipitation/IMERG'
CMA_folder = '/home/lzhou/Precipitation/Data/CMA_Historical_Data'
#C/home/lzhou/Precipitation/Data/CMA_Historical_Data
Output_folder = '/home/lzhou/Precipitation/Output'
Output_folder2 = '/home/lzhou/Precipitation/Precipitation_Scripts/Output/'
#case = 'IMERG_1000km_12'
#%% Make coordinates for IMERG data
# find the extent of data 
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape = world[world.name=='China'].copy()
cn_shape.reset_index(drop=True,inplace=True)
#%%
tw_shape = world[world.name=='Taiwan'].copy()
tw_shape.reset_index(drop=True,inplace=True)

#%%
stations = pd.read_csv(os.path.join(CMA_folder,'Wind_Rainfall','China_Sta.csv'))
stations['lat'] = stations['lat'].apply(lambda x: float(x[:2])+int(x[3:5])/60.)
stations['long'] = stations['long'].apply(lambda x: float(x[:3])+int(x[4:6])/60.)
stations.rename(columns={"long": "lon"},inplace=True)
#%%
fig, ax = plt.subplots(figsize=(8,8))
stations.plot.scatter(x='lon',y='lat',ax=ax)
cn_shape.geometry.boundary.plot(ax=ax)
tw_shape.geometry.boundary.plot(ax=ax)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.savefig('Fig1.png',bbox_inches='tight')