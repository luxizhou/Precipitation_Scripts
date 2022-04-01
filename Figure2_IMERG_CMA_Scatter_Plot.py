#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:21:44 2022

@author: lzhou
"""

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns; sns.set_theme(style='white')

import geopandas as gpd
import shapely
shapely.speedups.enabled
import warnings 
warnings.filterwarnings("ignore")
#import precipitation_utils 

#%% set parameters

IMERG_folder = '/media/lzhou/Extreme SSD/Precipitation/IMERG'
CMA_folder = '/home/lzhou/Precipitation/Data/CMA_Historical_Data/Wind_Rainfall/'
Output_folder = '/home/lzhou/Precipitation/Output'
Output_folder2 = '/home/lzhou/Precipitation/Precipitation_Scripts/Output/'
Figure_folder = os.path.join(Output_folder,'Figures')

case = 'IMERG_10degree_12'
item = 'precipitationCal'
threshold = 10.
#%% load station data
stations = pd.read_csv(os.path.join(CMA_folder,'China_Sta.csv'))
stations['lat'] = stations['lat'].apply(lambda x: float(x[:2])+int(x[3:5])/60.)
stations['long'] = stations['long'].apply(lambda x: float(x[:3])+int(x[4:6])/60.)
stations.rename(columns={"long": "lon"},inplace=True)

#%% load daily precip from CMA
daily_precip = pd.read_csv(os.path.join(CMA_folder,'1949-2018_DailyPrecipitation.csv'),names=['CMAID','TyphoonID','StationID','date','daily_precip'])
daily_precip = daily_precip[daily_precip.daily_precip>threshold]
daily_precip['time'] = pd.to_datetime(daily_precip.date)
daily_precip.drop(columns='date',inplace=True)
daily_precip = daily_precip.merge(stations[['StationID','lat','lon']], on='StationID')

#%% load CMA total precipitation data
total_precip = pd.read_csv(os.path.join(CMA_folder,'1949-2018_TotalPrecipitation.csv'), \
                           header=None,names=['SerialID','TCID','StationID','Total_Precip'])
total_precip['SerialID'] = total_precip['SerialID'].astype(int)
#filter out data points for which the total event precipitation is less than 10mm.
total_precip = total_precip[total_precip.Total_Precip>threshold]
# add lat lon to total_preicp 
total_precip = total_precip.merge(stations[['StationID','lat','lon']],on='StationID')
#%%
'''
# Number of measurements of total precipitation at each shanghai data stations:
station_freq = total_precip.StationID.value_counts().to_frame().reset_index().rename(columns={'index':'StationID','StationID':'Count'})
station_freq = station_freq.merge(stations,on='StationID')
station_freq = gpd.GeoDataFrame(station_freq, geometry=gpd.points_from_xy(station_freq.lon, station_freq.lat), \
                                crs="epsg:4326")
    
# Number of measurements of daily precipitation at each shanghai data stations:
daily_station_freq = daily_precip.StationID.value_counts().to_frame().reset_index().rename(columns={'index':'StationID','StationID':'Count'})
daily_station_freq = daily_station_freq.merge(stations,on='StationID')
daily_station_freq = gpd.GeoDataFrame(daily_station_freq, geometry=gpd.points_from_xy(daily_station_freq.lon, daily_station_freq.lat), \
                                crs="epsg:4326")
'''
#%% Get the domain of greater China
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape = world[world.name=='China'].copy()
cn_shape.reset_index(drop=True,inplace=True)
tw_shape = world[world.name=='Taiwan'].copy()
tw_shape.reset_index(drop=True,inplace=True)
greater_china = cn_shape.geometry.union(tw_shape.geometry)

#%% plot station frequency map
'''
fig,axes= plt.subplots(1,2,figsize=(15,6))
divider = make_axes_locatable(axes[0])
cax = divider.append_axes("right", size="3%", pad=-0.3)
station_freq.plot(column='Count',ax=axes[0],cmap='Spectral_r', cax=cax, \
                  legend=True, label='Event Total Precip.', \
                  legend_kwds={'label': "Number of Records"})
greater_china.boundary.plot(ax=axes[0])
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].legend(loc='upper left')
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="3%", pad=-0.3)
daily_station_freq.plot(column='Count',ax=axes[1],cmap='Spectral_r', cax=cax, \
                        legend=True, label='Daily Precip.', \
                        legend_kwds={'label': "Number of Records"})
greater_china.boundary.plot(ax=axes[1])
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
axes[1].legend(loc='upper left')
#plt.suptitle('Count of Measurement Data at CMA stations',fontweight='bold')
fname = 'CMA_station_'+str(threshold)+'mm_record_count.png'
fig.savefig(os.path.join(Figure_folder, fname),bbox_inches='tight')
#plt.close(fig)
'''
#%% Get event total at each grid point from all years
'''
filename = item+'_' + str(2000) +'.pkl' 
infile = os.path.join(Output_folde#%% Typhoon Impact Frequency based on IMERG extraction
df = pd.read_pickle(infile)
df = df.reset_index()
df['lat'] = df.y.round(2)
df['lon'] = df.x.round(2)
df.drop(columns=['x','y'],inplace=True)
event_total = df.groupby(['lat','lon','CMAID']).sum('precip').reset_index()
event_total = event_total[event_total.precip>0.]

for yy in np.arange(2001,2021):
    print(yy)
    filename = item+'_' + str(yy) +'.pkl' 
    infile = os.path.join(Output_folder,case,filename)
    df = pd.read_pickle(infile)
    df = df.reset_index()
    df['lat'] = df.y.round(2)
    df['lon'] = df.x.round(2)
    df.drop(columns=['x','y'],inplace=True)
    total_dummy = df.groupby(['lat','lon','CMAID']).sum('precip').reset_index()
    total_dummy = total_dummy[total_dummy.precip>0.]
    event_total = pd.concat([event_total,total_dummy],axis=0)
filename = item+'_event_total_precip.pkl' 
ofile = os.path.join(Output_folder,case,filename)
event_total.to_pickle(ofile)#total_precip.plot.scatter(x='Total_Precip',y='imerg_precip',hue='k',ax=ax)

'''
#%% load event total at each grid point from all years
filename = item+'_event_total_precip.pkl' 
infile = os.path.join(Output_folder,case,filename)
event_total = pd.read_pickle(infile)
event_total = event_total[event_total.precip>=threshold]

#%% Typhoon Impact Frequency based on IMERG extraction
data = event_total[(event_total.CMAID>200100) &(event_total.CMAID<201900)]
impact_freq = pd.DataFrame(data.groupby(['lat', 'lon']).size().rename('Count')).reset_index()
impact_freq = gpd.GeoDataFrame(impact_freq,geometry=gpd.points_from_xy(impact_freq.lon,impact_freq.lat), \
                               crs="epsg:4326")
data_mask = impact_freq.within(greater_china.iloc[0])
impact_freq = impact_freq.loc[data_mask]
impact_freq['freq'] = impact_freq.Count/(2018-2000)
#%%

# Number of measurements of total precipitation at each shanghai data stations:
station_freq = total_precip[total_precip.SerialID>200100].StationID.value_counts().to_frame().reset_index().rename(columns={'index':'StationID','StationID':'Count'})
station_freq = station_freq.merge(stations,on='StationID')
station_freq['freq'] = station_freq.Count/(2018-2000)
station_freq = gpd.GeoDataFrame(station_freq, geometry=gpd.points_from_xy(station_freq.lon, station_freq.lat), \
                                crs="epsg:4326")

#%%

fig,ax= plt.subplots(figsize=(12,12))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=-0.3)
impact_freq.loc[data_mask].plot(column='freq', ax=ax, vmin=0, vmax=10, \
                                cmap='Spectral_r', cax=cax, legend=True,  
                                #label='Event Total Precip.', \
                                legend_kwds={'label': "Annual Frequency"})
    
station_freq.plot(column='freq',ax=ax, vmin=0, vmax=10, cmap='Spectral_r', \
                  ec='k',cax=cax, legend=True, label='CMA observation', \
                  legend_kwds={'label': "Annual Frequency"})
    
greater_china.boundary.plot(ax=ax)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(loc='upper left')

fname = str(threshold)+'mm_frequency_map.png'
fig.savefig(os.path.join(Figure_folder, fname),bbox_inches='tight')
#plt.close(fig)
#%% just a test for plot
'''
event = event_total[event_total.CMAID==200808]
event = gpd.GeoDataFrame(event, geometry=gpd.points_from_xy(event.lon, event.lat), crs="epsg:4326")
fig,ax= plt.subplots(1,1,figsize=(12,8))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=-0.3)
event.plot(column='precip',ax=ax,cmap='Purples', vmin=0, vmax=200, cax=cax, legend=True, legend_kwds={'label': "Total Precip"})
greater_china.boundary.plot(ax=ax)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
#ax.set_title('Stations with Precip. data from CMA TC Database',fontweight='bold')
#fig.savefig(os.path.join(Figure_folder, 'CMA_station_record_count.png'),bbox_inches='tight')
'''
#%% find IMERG grids for CMA stations
imerg_lats = np.sort(event_total.lat.unique())
imerg_lons = np.sort(event_total.lon.unique())
for idx,row in stations.iterrows():
    #print(row)
    #sid = row['StationI#total_precip.plot.scatter(x='Total_Precip',y='imerg_precip',hue='k',ax=ax)
    #lat = stations.lat.iloc[0]
    #lon = stations.lon.iloc[0]
    diff_lat = -np.power(imerg_lats-row['lat'],2)
    diff_lon = -np.power(imerg_lons-row['lon'],2)
    lat_i = np.argmax(diff_lat)
    lon_i = np.argmax(diff_lon)
    total_precip.loc[total_precip.StationID==row['StationID'],'imerg_lat1']=imerg_lats[lat_i]
    total_precip.loc[total_precip.StationID==row['StationID'],'imerg_lon1']=imerg_lons[lon_i]
    stations.loc[idx,'imerg_lat1'] = imerg_lats[lat_i]
    stations.loc[idx,'imerg_lon1'] = imerg_lons[lon_i]
    
    if diff_lat[lat_i] == diff_lat[lat_i+1]:
        total_precip.loc[total_precip.StationID==row['StationID'],'imerg_lat2']=imerg_lats[lat_i+1]
        stations.loc[idx,'imerg_lat2'] = imerg_lats[lat_i+1]  
    if diff_lon[lon_i] == diff_lon[lon_i+1]:
        total_precip.loc[total_precip.StationID==row['StationID'],'imerg_lon2']=imerg_lons[lon_i+1]  
        stations.loc[idx,'imerg_lon2'] = imerg_lats[lon_i+1]  


#%% merge total_precip(CMA data) with event_total(IMERG data)

total_precip.rename(columns={'SerialID':'CMAID'},inplace=True)
total_precip = total_precip[total_precip.CMAID>200004].reset_index(drop=True)

merge1 = total_precip.merge(event_total,left_on=['CMAID','imerg_lat1','imerg_lon1'],right_on=['CMAID','lat','lon'],how='left')
merge1.drop(columns=['lat_y','lon_y'],inplace=True)
merge1.rename(columns={'lat_x':'orig_lat','lon_x':'orig_y'},inplace=True)

merge2 = merge1.merge(event_total,left_on=['CMAID','imerg_lat1','imerg_lon2'],right_on=['CMAID','lat','lon'],how='left')
merge2.drop(columns=['lat','lon'],inplace=True)

merge3 = merge2.merge(event_total,left_on=['CMAID','imerg_lat2','imerg_lon1'],right_on=['CMAID','lat','lon'],how='left')
merge3.drop(columns=['lat','lon'],inplace=True)
merge3.rename(columns={'precip_x':'precip_1','precip_y':'precip_2','precip':'precip_3'},inplace=True)

merge4 = merge3.merge(event_total,left_on=['CMAID','imerg_lat2','imerg_lon2'],right_on=['CMAID','lat','lon'],how='left')
merge4.drop(columns=['lat','lon'],inplace=True)
merge4.rename(columns={'precip':'precip_4'},inplace=True)

total_precip['imerg_precip'] = merge4[['precip_1','precip_2','precip_3','precip_4']].mean(axis=1)
#merge4['imerg_precip'] = total_precip['imerg_precip']

#%% linear fit for points

# first filter out points without imerg data
data = total_precip[~total_precip.imerg_precip.isnull()]

# do linear fit
Y = data.imerg_precip
X = data.Total_Precip
X = sm.add_constant(X)
model = sm.OLS(Y,X)
result = model.fit()
b,a = result.params

# do prediction for plotting
x = np.arange(10,10000)
x = sm.add_constant(x)
y_predict = result.predict(x)

#%%
fig,ax= plt.subplots(1,1,figsize=(6,6))
ax.plot(x,y_predict,linestyle='-',c='k',label='Linear Fit')
ax.plot(x,x,linestyle='--',c='k',label='1:1')
#ax.legend()
sns.scatterplot(x=total_precip.Total_Precip,y=total_precip.imerg_precip,fc='none',ec='k')
ax.set(xscale="log", yscale="log")
ax.set_xlim([threshold,1000])
ax.set_ylim([threshold,1000])
ax.set_xlabel('CMA Observation [mm]')
ax.set_ylabel('IMERG Observation [mm]')
ax.set_title('Event Total Precipitation')
txt = 'y={:.2f}x+{:.2f}'.format(a,b) + '\n' + r'$R^{2}$'+' = {:.2f}'.format(result.rsquared)
ax.text(50,600,txt)
fname = case+'_event_total_precip_' + str(threshold)+'mm_scatter.png'
outfile = os.path.join(Figure_folder,fname)
fig.savefig(outfile,bbox_inches='tight')
#ax.legend(['Linear Fit','1:1'])

#%% process daily precip data
daily_precip_orig = daily_precip.merge(stations[['StationID','imerg_lat1', 'imerg_lon1', 'imerg_lon2', 'imerg_lat2']],on='StationID',how='left')

for yy in np.arange(2000,2019):
    if yy == 2000:
        daily_precip = daily_precip_orig[(daily_precip_orig.CMAID>200004)&(daily_precip_orig.CMAID<200100)].reset_index(drop=True)
    else:
        daily_precip = daily_precip_orig[(daily_precip_orig.CMAID>yy*100)&(daily_precip_orig.CMAID<(yy+1)*100)].reset_index(drop=True)    
    print(yy)
    filename = item+'_' + str(yy) +'.pkl' 
    infile = os.path.join(Output_folder,case,filename)
    df = pd.read_pickle(infile)
    df = df.reset_index()
    df = df[df.precip>=10.]
    df['lat'] = df.y.round(2)
    df['lon'] = df.x.round(2)
    df.drop(columns=['x','y'],inplace=True)

    merge1 = daily_precip.merge(df,left_on=['CMAID','time','imerg_lat1','imerg_lon1'],right_on=['CMAID','time','lat','lon'],how='left')
    merge1.drop(columns=['lat_y','lon_y'],inplace=True)
    merge1.rename(columns={'lat_x':'orig_lat','lon_x':'orig_y'},inplace=True)
    #%%
    merge2 = merge1.merge(df,left_on=['CMAID','time','imerg_lat1','imerg_lon2'],right_on=['CMAID','time','lat','lon'],how='left')
    merge2.drop(columns=['lat','lon'],inplace=True)
    #%%
    merge3 = merge2.merge(df,left_on=['CMAID','time','imerg_lat2','imerg_lon1'],right_on=['CMAID','time','lat','lon'],how='left')
    merge3.drop(columns=['lat','lon'],inplace=True)
    merge3.rename(columns={'precip_x':'precip_1','precip_y':'precip_2','precip':'precip_3'},inplace=True)
    #%%
    merge4 = merge3.merge(df,left_on=['CMAID','time','imerg_lat2','imerg_lon2'],right_on=['CMAID','time','lat','lon'],how='left')
    merge4.drop(columns=['lat','lon'],inplace=True)
    merge4.rename(columns={'precip':'precip_4'},inplace=True)
    
    if yy == 2000:
        matched_daily_precip = merge4.copy()
    else:
        matched_daily_precip = pd.concat([matched_daily_precip,merge4],axis=0)
#%%
matched_daily_precip['imerg_precip'] = matched_daily_precip[['precip_1', 'precip_2', 'precip_3', 'precip_4']].mean(axis=1)
#%% linear fit daily precip data and plot

# first filter out points without imerg data
data = matched_daily_precip[~matched_daily_precip.imerg_precip.isnull()]

# do linear fit
Y = data.imerg_precip
X = data.daily_precip
X = sm.add_constant(X)
model = sm.OLS(Y,X)
result = model.fit()
b,a = result.params

# do prediction for plotting
x = np.arange(10,1000)
x = sm.add_constant(x)
y_predict = result.predict(x)

#%%
fig,ax= plt.subplots(1,1,figsize=(6,6))
ax.plot(x,y_predict,linestyle='-',c='k',label='Linear Fit')
ax.plot(x,x,linestyle='--',c='k',label='1:1')
#ax.legend()
sns.scatterplot(x=total_precip.Total_Precip,y=total_precip.imerg_precip,fc='none',ec='k')
ax.set(xscale="log", yscale="log")
ax.set_xlim([threshold,1000])
ax.set_ylim([threshold,1000])
ax.set_xlabel('CMA Observation [mm]')
ax.set_ylabel('IMERG Observation [mm]')
ax.set_title('Daily Precipitation')
txt = 'y={:.2f}x+{:.2f}'.format(a,b) + '\n' + r'$R^{2}$'+' = {:.2f}'.format(result.rsquared)
ax.text(50,600,txt)
fname = case+'_daily_precip_' + str(threshold)+'mm_scatter.png'
outfile = os.path.join(Figure_folder,fname)
fig.savefig(outfile,bbox_inches='tight')