#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf 
from tensorflow import keras
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM 
from keras.layers import GRU
from keras.layers import Bidirectional,MaxPooling1D,Flatten,TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from keras.utils.np_utils import to_categorical 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import keras.callbacks
import os 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Daiyi's note: Call PrepDataset.py script to import df and df_profile variables
import PrepDataset
from PrepDataset import df,df_profile 


# In[3]:


# Split data into training, validation, and test sets
val = 0.2
test = 0.1
train = 1 - val - test

X_train = df[:int(train*df.shape[0])+1:,::,::]  # [i not in [1] for i in range(df.shape[2])]]
X_val = df[int(train*df.shape[0])+1:int(train*df.shape[0])+int(val*df.shape[0])+1:,::,::]  # [i not in [1] for i in range(df.shape[2])]]
X_test = df[int(train*df.shape[0])+int(val*df.shape[0])+1::,::,::]  # [i not in [1] for i in range(df.shape[2])]]

oh_target = (np.arange(df_profile[:,0,1].max()+1) == df_profile[:,0,1][...,None]).astype(int)
oh_target = np.delete(oh_target,np.where(~oh_target.any(axis=0))[0], axis=1)

y_train = oh_target[:int(train*oh_target.shape[0])+1:,]
y_val = oh_target[int(train*oh_target.shape[0])+1:int(train*oh_target.shape[0])+int(val*oh_target.shape[0])+1:,]
y_test = oh_target[int(train*oh_target.shape[0])+int(val*oh_target.shape[0])+1::,]


# In[57]:


#verify shape for each dataset
X_train.shape,X_val.shape,X_test.shape,y_train.shape,y_val.shape,y_test.shape


# In[60]:


len(X_train)


# # Keras Models

# In[6]:


# create and fit the baseline model
# LSTM RNN
model_0 = Sequential()
model_0.add(LSTM(4))
model_0.add(Dense(y_train.shape[1]))
model_0.add(Activation('softmax'))
model_0.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[8]:


#fit model  
#evaluation for the baseline model validation dataset
history0 = model_0.fit(X_train, y_train, epochs=10, batch_size=len(X_train),
                    validation_data=(X_val, y_val), verbose=2, shuffle=False)

plt.plot(history0.history['loss'], label='train')
plt.plot(history0.history['val_loss'], label='test')
plt.legend()
plt.show() 


# In[9]:


predictions_0= model_0.predict(X_val, verbose=0)
print(predictions_0[:1])


# In[4]:


# model_0.save('DaiyiD_model_Keras_0.h5')


# In[10]:


# create and fit model 1
# LSTM RNN with dropout
model_01 = Sequential()
model_01.add(LSTM(10, dropout=0.2, recurrent_dropout=0.2))
model_01.add(Dense(y_train.shape[1]))
model_01.add(Activation('softmax'))
model_01.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#fit model  


# In[11]:


#fit model  
#evaluation for the baseline model validation dataset
history_01 = model_01.fit(X_train, y_train, epochs=25, batch_size=len(X_train),
                    validation_data=(X_val, y_val), verbose=2, shuffle=False)

plt.plot(history_01.history['loss'], label='train')
plt.plot(history_01.history['val_loss'], label='test')
plt.legend()
plt.show() 


# In[13]:


predictions_01= model_01.predict(X_val, verbose=0)
print(predictions_01[:1])


# In[26]:


# create and fit model 2
# LSTM RNN with Sgd optimizer
model_02 = Sequential()
model_02 .add(LSTM(4))
model_02 .add(Dense(y_train.shape[1]))
model_02 .add(Activation('softmax'))
model_02 .compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])


# In[27]:


#fit model  
#evaluation for the baseline model validation dataset
history_02 = model_02.fit(X_train, y_train, epochs=20, batch_size=len(X_train),
                          validation_data=(X_val, y_val), verbose=2, shuffle=False)


# In[28]:


plt.plot(history_02.history['loss'], label='train')
plt.plot(history_02.history['val_loss'], label='test')
plt.legend()
plt.show() 


# In[29]:


plt.plot(history_02.history['acc'], label='train')
plt.plot(history_02.history['val_acc'], label='test')
plt.legend()
plt.show() 


# In[32]:


predictions_02= model_02.predict(X_val, verbose=0)
print(predictions_02[:1])


# In[60]:


# model_1.save('DaiyiD_model_Keras_1.h5')


# In[17]:


# create and fit the model 3
# GRU
model_03 = Sequential()
model_03.add(GRU(4))
model_03.add(Dense(y_train.shape[1]))
model_03.add(Activation('softmax'))
model_03.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[18]:


#fit model  
#evaluation for the baseline model validation dataset
history_03 = model_03.fit(X_train, y_train, epochs=25, batch_size=len(X_train),
                    validation_data=(X_val, y_val), verbose=2, shuffle=False)

plt.plot(history_03.history['loss'], label='train')
plt.plot(history_03.history['val_loss'], label='test')
plt.legend()
plt.show() 


# In[19]:


predictions_03= model_03.predict(X_val, verbose=0)
print(predictions_03[:1])


# In[61]:


# model_2.save('DaiyiD_model_Keras_2.h5')


# In[4]:


# create and fit the model 4
# BiLSTM + CNN

model_04 = Sequential() 
model_04.add(Bidirectional(LSTM(4, return_sequences=True)))
model_04.add(TimeDistributed(Dense(4)))
model_04.add(Activation('softplus'))
model_04.add(MaxPooling1D(5))
model_04.add(Flatten())
model_04.add(Dense(y_train.shape[1]))
model_04.add(Activation('softmax'))
model_04.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[5]:


#fit model  
#evaluation for the baseline model validation dataset
history_04 = model_04.fit(X_train, y_train, epochs=25, batch_size=len(X_train),
                    validation_data=(X_val, y_val), verbose=2, shuffle=False)

plt.plot(history_04.history['loss'], label='train')
plt.plot(history_04.history['val_loss'], label='test')
plt.legend()
plt.show() 


# In[23]:


plt.plot(history_04.history['acc'], label='train')
plt.plot(history_04.history['val_acc'], label='test')
plt.legend()
plt.show() 


# In[7]:


predictions_04= model_04.predict(X_val, verbose=0)
print(predictions_04[:1])


# In[20]:


# create and fit the model 5
#LSTM with earlystopping
model_05= Sequential()
model_05.add(LSTM(4, return_sequences=True,  input_shape=(X_train.shape[1], X_train.shape[2])))
model_05.add(LSTM(4))
model_05.add(Dense(y_train.shape[1]))
model_05.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 


# In[21]:


model_05.save('model_05.h5')


# In[24]:


# Early stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='model_05.h5', monitor='val_loss', save_best_only=True)]
history_05 = model_05.fit(X_train, y_train, epochs=20, batch_size=len(X_train),
                    validation_data=(X_val, y_val), verbose=2, shuffle=False,callbacks=callbacks)


 


# In[30]:


plt.plot(history_05.history['loss'], label='train')
plt.plot(history_05.history['val_loss'], label='test')
plt.legend()
plt.show() 


# In[31]:


plt.plot(history_02.history['acc'], label='train')
plt.plot(history_02.history['val_acc'], label='test')
plt.legend()
plt.show() 


# In[25]:


predictions_05= model_05.predict(X_val, verbose=0)
print(predictions_05[:1])

