# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from matplotlib.backends.backend_pdf import PdfPages

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
figures_name = 'RP_curves_xlog.pdf'


#%%
same_length = 0
#common_period = 18      #1949-2018
Shanghai_period = 2018-1948
IMERG_period = 2020-2000
APHRO_period = 2015-1997
ERA_Period = 2020-1980
#%% load IMERG data
filename = case_name + '_' + factor + '_precip.pkl'
data = pd.read_pickle(os.path.join(data_folder,filename))  
grids = data.iloc[:,:2].copy()

#%% load total precipitation data
total_precip = pd.read_csv(os.path.join(Shanghai_folder,'1949-2018_TotalPrecipitation.csv'), \
                header=None,names=['SerialID','TCID','StationID','Total_Precip'])
total_precip['SerialID'] = total_precip['SerialID'].astype(int)

#%%
if same_length == 1:
    if 'IMERG' in case_name:
        data = data.iloc[:,19:386]
        total_precip = total_precip[total_precip.SerialID>=200100]
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
pp = PdfPages(os.path.join(figure_folder,figures_name))
print(filename)
for ii in np.arange(0,20):
    sid = station_freq.StationID.iloc[ii] # Haikou should be the station with most record
    s_total_precip = total_precip[total_precip.StationID==sid].copy()
    if same_length==1:
        s_total_precip = s_total_precip[s_total_precip.SerialID>=200000] # to be decided carefully if choose same_length
    else:
        s_total_precip = get_EP(s_total_precip,'Total_Precip',Shanghai_period)    

    idx = stations[stations.StationID==sid].index
    sname = stations.loc[idx[0],'StationName']
    s_lat = stations.loc[idx[0],'lat']
    s_lon = stations.loc[idx[0],'lon']

    precip_idx = find_precip_grids(s_lat, s_lon, grids)
    precip = data.loc[precip_idx].dropna(axis=1)
    if len(precip_idx)>1:
        precip = precip.mean()
        precip = precip.to_frame(name='Total_Precip')
    else:
        dummy = precip.T
        precip = dummy.rename(columns={dummy.columns.values[0]:'Total_Precip'})
        
    target_precip = get_EP(precip,'Total_Precip',IMERG_period)
    
    # plot figures
    if 'RP' in figures_name:    
        ax = s_total_precip.plot(x='RP',y='Total_Precip',marker='.',color='b',linestyle='',label='Shanghai Data')
        target_precip.plot(x='RP',y='Total_Precip',marker='.',color='r',linestyle='',label=case_name,ax=ax)
        plt.legend()
        ax.set_xlabel('Return Period (Year)')
        ax.set_ylabel('Total Precipitation (mm)')
    
    if 'EP' in figures_name:
        ax = s_total_precip.plot(x='EP',y='Total_Precip',marker='.',color='b',linestyle='',label='Shanghai Data')
        target_precip.plot(x='EP',y='Total_Precip',marker='.',color='r',linestyle='',label=case_name,ax=ax)
        plt.legend()
        ax.set_xlabel('Exceedence Probability')
        ax.set_ylabel('Total Precipitation (mm)')
    
    if 'xlog' in figures_name:
        ax.set_xscale('log')
        
    if 'ylog' in figures_name:
        ax.set_yscale('log')
        
    ax.set_title(sname)
    fig = plt.gcf()
    pp.savefig(fig)
    plt.close(fig)

pp.close()
    





