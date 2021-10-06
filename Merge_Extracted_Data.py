#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:34:47 2021

@author: lzhou
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:17:51 2021

@author: lzhou
"""

#import numpy as np
import os
import pandas as pd
#import geopandas as gpd
#import matplotlib.pyplot as plt
import shapely
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib
shapely.speedups.enabled

#%%
#wrk_dir = '/home/lzhou/Precipitation/Precipitation_Scripts'
#os.chdir(wrk_dir)

data_type = 'APHRO'
case_name = 'APHRO_1000km_12'
factor = 'total'
data_dir0 = '/home/lzhou/Precipitation/Output/'
data_dir1 = os.path.join(data_dir0,case_name)
output_dir = os.path.join(data_dir0,'Merged_Output')
#figure_dir = '/home/lzhou/Precipitation/Output/Figures'
#%% combine total precipitation data from each event

dummy = os.listdir(data_dir1)
files = [x for x in dummy if (factor in x) and ('pkl' in x)]
files.sort()
#%%
os.chdir(data_dir1)
first_call = 1
for infile in files:
    print(infile)
    data = pd.read_pickle(infile)
    if data_type=='ERA5':
        data.rename(columns={'precip':infile[:6],'latitude':'lat','longitude':'lon'},inplace=True)              
    else:
        data.rename(columns={'precip':infile[:6]},inplace=True)        
    #print(len(data))
    
    if (first_call==1) :
        df = data.copy()
        first_call = 0
    else:
        df = df.merge(data,on=['lat','lon'],how = 'outer')
#%%
os.chdir(output_dir)
filename = case_name+'_'+factor+'_precip.pkl'
df.to_pickle(filename)
