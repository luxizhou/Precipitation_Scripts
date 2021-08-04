# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import netCDF4 as nc
import os
#import geopandas as gpd
#from shapely.geometry import Point, LineString, Polygon
from matplotlib.backends.backend_pdf import PdfPages

#%%
#%%
def get_EP(df,precip_col,Record_Period):
    
    df['Rank'] = df[precip_col].rank(method='min',ascending=False)
    df['EP'] = df.Rank/Record_Period
    df['RP'] = 1./df.EP
    
    return df

def find_IMERG_precip(slat,slon,data):
    
    lats = data.sort_values(by='lat').lat.unique()
    lons = data.sort_values(by='lon').lon.unique()

    lati = np.abs(lats-slat).argmin()
    slat1 = lats[lati]
    if slat1-slat>0.049:
        slat1 = [slat1, lats[lati-1]]
    elif slat-slat1>0.049:
        slat1 = [slat1,lats[lati+1]]

    loni = np.abs(lons-slon).argmin()
    slon1 = lons[loni]
    if slon1-slon>0.049:
        slon1 = [slon1, lons[loni-1]]
    elif slon-slon1>0.049:
        slon1 = [slon1,lons[loni+1]]    

    try:
        idx1 = data[data.lat==slat1].index
    except:
        idx1 = data[data.lat.isin(slat1)].index
    
    try:
        idx2 = data[data.lon==slon1].index
    except:
        idx2 = data[data.lon.isin(slon1)].index
        
    grids = data.loc[idx2&idx1]
    
    return grids

#%%
#Shanghai_folder = r'D:\Precipitation\CMA_Historical_Data\Wind_Rainfall'
Shanghai_folder = '/home/lzhou/Precipitation/TC_Tracks/CMA_Historical_Data/Wind_Rainfall'
IMERG_folder = '/home/lzhou/Precipitation/Output/IMERG_500km_12'

#%% load processed typhoon precipitation from IMERG data
# files = os.listdir(IMERG_folder)
# files = [x for x in files if ('total' in x) & ('pkl' in x)]

# first_read=1
# nn = 1
# for filename in files:
#     if np.mod(nn,10) ==0:
#         print(nn)

#     df = pd.read_pickle(os.path.join(IMERG_folder,filename))
#     df['CMAID'] = int(filename[:6])
#     if (first_read==1):
#         data = df.copy()
#         first_read=0
#     else:
#         data = data.append(df,ignore_index=True)
    
#     nn = nn + 1
    
#data.to_pickle(os.path.join(IMERG_folder,'Total_Precip_by_event.pkl'))   

#%%
Record_Period = 19      #1949-2018
#%%
#%% load IMERG data

imerg_data = pd.read_pickle(os.path.join(IMERG_folder,'Total_Precip_by_event.pkl'))  
imerg_data = imerg_data[imerg_data.CMAID<201900]

#%% load total precipitation data
total_precip = pd.read_csv(os.path.join(Shanghai_folder,'1949-2018_TotalPrecipitation.csv'), \
                header=None,names=['SerialID','TCID','StationID','Total_Precip'])
total_precip['SerialID'] = total_precip['SerialID'].astype(int)
total_precip = total_precip[total_precip.SerialID>=200000]
#%% load station data
stations = pd.read_csv(os.path.join(Shanghai_folder,'China_Sta.csv'))
stations['lat'] = stations['lat'].apply(lambda x: float(x[:2])+int(x[3:5])/60.)
stations['long'] = stations['long'].apply(lambda x: float(x[:3])+int(x[4:6])/60.)
stations.rename(columns={"long": "lon"},inplace=True)

#%%
# Number of measurements of total precipitation at each shanghai data stations:
station_freq = total_precip.StationID.value_counts().to_frame().reset_index().rename(columns={'index':'StationID','StationID':'Count'})
station_freq = station_freq.merge(stations,on='StationID')
#%% loop through top 20 most record stations
pp = PdfPages(os.path.join(IMERG_folder,'EP_curves.pdf'))
for ii in np.arange(0,20):
    sid = station_freq.StationID.iloc[ii] # Haikou should be the station with most record
    s_total_precip = total_precip[total_precip.StationID==sid].copy()
    #s_total_precip = s_total_precip[s_total_precip.SerialID>=200000]
    s_total_precip = get_EP(s_total_precip,'Total_Precip',Record_Period)    

    idx = stations[stations.StationID==sid].index
    sname = stations.loc[idx[0],'StationName']
    s_lat = stations.loc[idx[0],'lat']
    s_lon = stations.loc[idx[0],'lon']

    grids = find_IMERG_precip(s_lat,s_lon,imerg_data)
    #grids = grids[grids.CMAID<201900]
    i_total_precip = grids.groupby('CMAID')['precip'].mean().to_frame()
    i_total_precip = get_EP(i_total_precip,'precip',19)
        
    ax = s_total_precip.plot(x='EP',y='Total_Precip',marker='.',linestyle='',label='Shanghai Data')
    i_total_precip.plot(x='EP',y='precip',marker='.',linestyle='',label='IMERG',ax=ax)

    ax.set_xlabel('Annal Exceedence Frequency')
    ax.set_ylabel('Total Precipitation (mm)')
    ax.set_xscale('log')
    ax.set_title(sname)
    fig = plt.gcf()
    pp.savefig(fig)
    plt.close(fig)

pp.close()
    





