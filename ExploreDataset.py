#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Packages required
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[21]:


# Daiyi's note: Call PrepDataset.py script to import df and df_profile variables
import PrepDataset
from PrepDataset import df,df_profile 


# In[23]:


# Create correlation heatmap  
df_corr=df.drop(columns=df_profile.columns)
f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(df_corr.corr(), cmap = 'viridis',annot=True,ax=ax)
plt.show()


# In[17]:


#Check for missing values
for key,value in df.items():  
    print('Missing values numbers for {}: '.format(key), value.T.isnull().any().sum())


# In[24]:


# Outlier value detection 
df_corr.plot(kind='box', subplots=True, grid=True, title="Attribute Value Detection",layout=(4, 5), sharex=True, sharey=False, legend=True, figsize = (16,16))
plt.figure('tight')


# As we obersaved from the above boxplots for each attribute, there are many outliers especially in eps1, ps1,ps2, ps3, ps4, se, and vs1. but since the number of datapoints is small, we chose to keep the outliers 

# In[4]:


#Visualization of the target variables
for i in enumerate(list(df_profile.columns)):
    print(pd.unique(df_profile[i[1]]))

# Construct 5 subplots for the 5 target variables    
figure,(ax1, ax2,ax3,ax4,ax5) = plt.subplots(figsize=(25, 20 ), 
                       nrows=df_profile.shape[1], ncols=1,sharex = False)
sns.distplot(df_profile['Cooler_efficiency'], hist=True, color = 'red',ax=ax1)
sns.distplot(df_profile['Valve_response'], hist=True,color = 'red',ax=ax2)
sns.distplot(df_profile['Pump_leakage'], hist=True, color = 'red',ax=ax3)
sns.distplot(df_profile['Accumulator_pressure'], hist=True,color = 'red',ax=ax4)
sns.distplot(df_profile['Instability_flag'], hist=True, color = 'red',ax=ax5)


# In[81]:


figure,(ax1, ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(figsize=(25, 20 ), 
                       nrows=8, ncols=1,sharex = False)
df['TS1'].plot(subplots=True, sharex=True,color='tomato',ax=ax1 ) 
ax1.set_title('TS1')
df['CE'].plot(subplots=True, sharex=True,color='tomato',ax=ax2 ) 
ax2.set_title('CE') 
df['CP'].plot(subplots=True, sharex=True,color='tomato',ax=ax3 ) 
ax3.set_title('CP') 
df['SE'].plot(subplots=True, sharex=True,color='tomato',ax=ax4 ) 
ax4.set_title('SE') 
df['FS1'].plot(subplots=True, sharex=True,color='tomato',ax=ax5 ) 
ax5.set_title('FS1') 
df['PS1'].plot(subplots=True, sharex=True,color='tomato',ax=ax6 ) 
ax6.set_title('PS1') 
df['EPS1'].plot(subplots=True, sharex=True,color='tomato',ax=ax7 ) 
ax7.set_title('EPS1') 
df['VS1'].plot(subplots=True, sharex=True,color='tomato',ax=ax8 ) 
ax8.set_title('VS1') 
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()


# In[112]:


print(df_profile.columns) 

df_profile['Cooler_efficiency'].describe(),df_profile['Valve_response'].describe(),df_profile['Pump_leakage'].describe(),df_profile['Accumulator_pressure'].describe(),df_profile['Instability_flag'].describe()


# Based on the above results of the 5 different target variables, we should choose a variable relatively evenly distributed - the feature 'Valve_response' with a mean of 90.69, The median of the sample is 100, meaning more than half of the cycles, the valve is in optimal condition.
