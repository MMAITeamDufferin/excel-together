#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import zoom


# In[6]:


# Set max_columns output to 999
pd.set_option('display.max_columns', 999)

# Extract http://archive.ics.uci.edu/ml/machine-learning-databases/00447/ to same directory as this script

# List files
one = ['.\\data\\TS1.txt', '.\\data\\TS2.txt', '.\\data\\TS3.txt', '.\\data\\TS4.txt', '.\\data\\VS1.txt', '.\\data\\CE.txt', '.\\data\\CP.txt', '.\\data\\SE.txt']
ten = ['.\\data\\FS1.txt', '.\\data\\FS2.txt']
hundred = ['.\\data\\PS1.txt', '.\\data\\PS2.txt', '.\\data\\PS3.txt', '.\\data\\PS4.txt', '.\\data\\PS5.txt', '.\\data\\PS6.txt', '.\\data\\EPS1.txt']

# Parse condition profiles
df_profile = pd.read_table('.\\data\\profile.txt', header=None)
df_profile = df_profile.values.reshape(2205, 1, 5)
df_profile = zoom(df_profile, (1,6000,1))

# Parse 1 Hz measurements
df_one =  np.stack([pd.read_table(x, header=None) for x in one], axis=2)
df_one = zoom(df_one, (1, 100, 1))

# Parse 10 Hz measurements
df_ten =  np.stack([pd.read_table(x, header=None) for x in ten], axis=2)
df_ten = zoom(df_ten, (1, 10, 1))

# Parse 100 Hz measurements
df_hundred = np.stack([pd.read_table(x, header=None) for x in hundred], axis=2)

# Concatenate all data
df = np.concatenate([df_one, df_ten, df_hundred], axis=2)

