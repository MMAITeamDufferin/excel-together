import numpy as np
import pandas as pd
import tensorflow as tf

# Set max_columns output to 999
pd.set_option('display.max_columns', 999)

# Extract http://archive.ics.uci.edu/ml/machine-learning-databases/00447/ to same directory as this script

# List files
one = ['.\\data\\TS1.txt', '.\\data\\TS2.txt', '.\\data\\TS3.txt', '.\\data\\TS4.txt', '.\\data\\VS1.txt', '.\\data\\CE.txt', '.\\data\\CP.txt', '.\\data\\SE.txt']
ten = ['.\\data\\FS1.txt', '.\\data\\FS2.txt']
hundred = ['.\\data\\PS1.txt', '.\\data\\PS2.txt', '.\\data\\PS3.txt', '.\\data\\PS4.txt', '.\\data\\PS5.txt', '.\\data\\PS6.txt', '.\\data\\EPS1.txt']

# Parse 1 Hz measurements
df_one = pd.concat([pd.read_table(x, header=None).stack() for x in one], axis=1)
df_one.columns = (x[7:-4] for x in one)
df_one = df_one.reset_index()
df_one.index = (df_one.index + 1)*100

# Parse 10 Hz measurements
df_ten = pd.concat([pd.read_table(x, header=None).stack() for x in ten], axis=1)
df_ten.columns = (x[7:-4] for x in ten)
df_ten = df_ten.reset_index()
df_ten.index = (df_ten.index +1)*10

# Parse 100 Hz measurements
df_hundred = pd.concat([pd.read_table(x, header=None).stack() for x in hundred], axis=1)
df_hundred.columns = (x[7:-4] for x in hundred)
df_hundred = df_hundred.reset_index()

# Parse condition profiles
df_profile = pd.read_table('.\\data\\profile.txt', header=None)
df_profile.columns = ['Cooler_efficiency', 'Valve_response', 'Pump_leakage', 'Accumulator_pressure', 'Instability_flag']
df_profile.index = (df_profile.index +1)*6000

# Concatenate all data
df = pd.concat([df_one, df_ten, df_hundred, df_profile], axis=1)
df = df.fillna(method='backfill')
df = df.fillna(method='ffill')
df = df.loc[:,~df.columns.duplicated()]

# Explore data
# print(df.shape)
# print(list(df))
# print(df.head())
# print(df.tail())
# print(df.dtypes)
# print(df.describe())

target = 'Valve_response'

# Split data into training, validation, and test sets
val = 0.2
test = 0.1
train = 1 - val - test

X_train = df.loc[:int(train*df.shape[0]), df.columns != target]
X_val = df.loc[int(train*df.shape[0]):int(train*df.shape[0])+int(val*df.shape[0]), df.columns != target]
X_test = df.loc[int(train*df.shape[0])+int(val*df.shape[0]):len(df), df.columns != target]

y_train = df[target][:int(train*df.shape[0]),]
y_val = df[target][int(train*df.shape[0]):int(train*df.shape[0])+int(val*df.shape[0]),]
y_test = df[target][int(train*df.shape[0])+int(val*df.shape[0]):len(df),]

# np.random.permutation(df.shape[0])  # Function to shuffle data



"""
references:
- https://github.com/RobRomijnders/LSTM_tsc
"""