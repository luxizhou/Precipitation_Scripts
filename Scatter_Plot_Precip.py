#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:51:45 2021

@author: lzhou
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os 
import geopandas as gpd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.backends.backend_pdf import PdfPages

#%%

font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
#%%

def get_EP(df,precip_col,Record_Period):
    
    df['Rank'] = df[precip_col].rank(method='min',ascending=False)
    df['EP'] = df.Rank/Record_Period
    df['RP'] = 1./df.EP
    df.sort_values(by='RP',ascending=False,inplace=True)
    
    return df

def find_precip_grids(slat,slon,precip_grids):
    
    precip_grids['lat_diff'] = np.abs(precip_grids.lat-slat).round(2)
    precip_grids['lon_diff'] = np.abs(precip_grids.lon-slon).round(2)
    
    precip_idx = precip_grids[(precip_grids.lat_diff==precip_grids.lat_diff.min())&(precip_grids.lon_diff==precip_grids.lon_diff.min())].index
    
    return precip_idx


#%%
#Shanghai_folder = r'D:\Precipitation\CMA_Historical_Data\Wind_Rainfall'
Shanghai_folder = '/home/lzhou/Precipitation/Data/CMA_Historical_Data/Wind_Rainfall'
data_folder0 = '/home/lzhou/Precipitation/Output'
data_folder = os.path.join(data_folder0,'Merged_Output')
figure_folder = os.path.join(data_folder0,'Figures')

case_name = 'IMERG_1000km_12'
factor = 'total'
figures_name = case_name + '_scatter_plot_total_precip.pdf'
#%%
same_length = 1
# #common_period = 18      #1949-2018
# Shanghai_period = 2018-1948
# IMERG_period = 2020-2000
# APHRO_period = 2015-1997
# ERA_Period = 2020-1980
#%% load IMERG data
filename = case_name + '_' + factor + '_precip.pkl'
data = pd.read_pickle(os.path.join(data_folder,filename))  
grids = data.iloc[:,:2].copy()

#%% load total precipitation data
total_precip = pd.read_csv(os.path.join(Shanghai_folder,'1949-2018_TotalPrecipitation.csv'), \
                header=None,names=['SerialID','TCID','StationID','Total_Precip'])
total_precip['SerialID'] = total_precip['SerialID'].astype(int)
#%% filter out data points for which the total event precipitation is less than 10mm.
idx = total_precip.Total_Precip>=10.
total_precip = total_precip.loc[idx,:]
#%% load station data
stations = pd.read_csv(os.path.join(Shanghai_folder,'China_Sta.csv'))
stations['lat'] = stations['lat'].apply(lambda x: float(x[:2])+int(x[3:5])/60.)
stations['long'] = stations['long'].apply(lambda x: float(x[:3])+int(x[4:6])/60.)
stations.rename(columns={"long": "lon"},inplace=True)

# Number of measurements of total precipitation at each shanghai data stations:
station_freq = total_precip.StationID.value_counts().to_frame().reset_index().rename(columns={'index':'StationID','StationID':'Count'})
station_freq = station_freq.merge(stations,on='StationID')
#%% plot station_freq

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cn_shape1 = world[(world.name=='China')].copy()
cn_shape2 = world[(world.name=='Taiwan')].copy()
#%%
#cn_shape.reset_index(drop=True,inplace=True)

station_freq = gpd.GeoDataFrame(station_freq, \
                                geometry=gpd.points_from_xy(station_freq.lon, station_freq.lat), \
                                crs="epsg:4326")
#%%
fig,ax= plt.subplots(1,1,figsize=(12,8))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=-0.3)
station_freq.plot(column='Count',ax=ax,cmap='Spectral_r', cax=cax, \
                  legend=True, legend_kwds={'label': "Number of Records"})

cn_shape1.boundary.plot(ax=ax)
cn_shape2.boundary.plot(ax=ax)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Stations with Precip. data from CMA TC Database',fontweight='bold')
fig.savefig(os.path.join(figure_folder, 'CMA_station_record_count.png'),bbox_inches='tight')
plt.close(fig)

#%%
if 'IMERG' in case_name:
    data = data.iloc[:,19:]
elif 'APHRO' in case_name:
    data = data.iloc[:,2:]
#%% ignore grids where total precipitation is less than 10mm
data.where(data>=10.,np.nan,inplace=True)
#%%
if same_length == 1:
    if 'IMERG' in case_name:
        total_precip = total_precip[total_precip.SerialID>=200100].reset_index(drop=True)
        data = data.iloc[:,:345]
    elif 'APHRO' in case_name:
        total_precip = total_precip[(total_precip.SerialID>=199800) & (total_precip.SerialID<201600)].reset_index(drop=True)
    
        #Shanghai_period = IMERG_period
        #figures_name = 'SL_' + figures_name

#%% loop through top 20 most record stations
pp = PdfPages(os.path.join(figure_folder,figures_name))
print(filename)
for ii in np.arange(0,20):
    sid = station_freq.StationID.iloc[ii] # Haikou should be the station with most record
    s_total_precip = total_precip[total_precip.StationID==sid].copy()
    s_total_precip['SerialID']=s_total_precip['SerialID'].astype('int')
    
    idx = stations[stations.StationID==sid].index
    sname = stations.loc[idx[0],'StationName']
    s_lat = stations.loc[idx[0],'lat']
    s_lon = stations.loc[idx[0],'lon']
#%%
    precip_idx = find_precip_grids(s_lat, s_lon, grids)
    precip = data.loc[precip_idx].dropna(axis=1)
    #%%
    if len(precip_idx)>1:
        precip = precip.mean()
        precip = precip.to_frame(name='Total_Precip')
    else:
        dummy = precip.T
        precip = dummy.rename(columns={dummy.columns.values[0]:'Total_Precip'})
        #%%
    precip = precip.reset_index().rename(columns={'index':'SerialID'})        
    precip['SerialID'] = precip['SerialID'].astype(int)
    
    
#%% Merge two datasets

    precip = precip.merge(s_total_precip[['SerialID','Total_Precip']],on='SerialID',how='outer')
    precip.rename(columns={'Total_Precip_x':case_name,'Total_Precip_y':'Shanghai'},inplace=True)
    
    dummy1 = precip.count().to_frame(name=sname)
    dummy2 = ~precip.isna().any(axis=1)
    dummy1.loc['common',sname]=len(dummy2[dummy2==True])
    
    dummy3 = precip.loc[dummy2].values
    rmse = np.sqrt(np.mean((dummy3[:,1]-dummy3[:,2])**2))
    rb = np.sum(dummy3[:,1]-dummy3[:,2])/np.sum(dummy3[:,2])
    corr = np.corrcoef(dummy3[:,1],dummy3[:,2])
    cc = corr[1,0]
    
    if ii == 0:
        info = dummy1.copy()
        stats_table = pd.DataFrame(columns=['Station','RMSE','RB','CC'])
    else:
        info = pd.concat([info,dummy1],axis=1)
    
    #stats_table.iloc[ii,'Station'] = sname
    stats_table = stats_table.append({'Station':sname,'RMSE':rmse,'RB':rb,'CC':cc},ignore_index=True)
        
#%% Plot
    ax=sns.scatterplot(data=precip,x='Shanghai',y=case_name)   
    (a,b) = ax.get_xlim()
    (c,d) = ax.get_ylim()
    axis_max = np.max([b,d])
    ax.set_xlim([0,axis_max])
    ax.set_ylim([0,axis_max])
    plt.plot([0,axis_max],[0,axis_max],linestyle='--')
    plt.text(0.2, 0.9, 'RMSE = %d mm'%rmse,horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    plt.text(0.2, 0.8, 'RB = {:.0%}'.format(rb), horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
    plt.text(0.2, 0.7, 'CC = {:.2f}'.format(cc), horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
    ax.set_title(sname)
    ax.set_xlabel('CMA TC Database')
    fig = plt.gcf()
    pp.savefig(fig,bbox_inches='tight')
    plt.close(fig)

pp.close()

info = info.T
ofile_name = case_name + '_dp_count_at_SH_station.pkl'
info.to_pickle(os.path.join(figure_folder, ofile_name))

ofile_name = case_name + '_SH_data_stats.csv'
stats_table.to_csv(os.path.join(figure_folder,ofile_name),index=False)
