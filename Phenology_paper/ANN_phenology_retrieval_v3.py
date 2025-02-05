# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:11:33 2018
@author: HONGQUAN
"""
# Random Forest Algorithm for retrieve the crop phenology
# scipy package: need to import and learn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import random
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler as xscaler 
from sklearn import metrics #includes the metrics

#%% Importing the datasets
ty=7

if ty==1:
        x_path = R'G:\Phenology\Method2\RF\X_RADAR_Canola_v21.xlsx'
        y_path = R'G:\Phenology\Method2\RF\y_BBCH_Canola_v2.xlsx'
elif ty==2:
        x_path = R'G:\Phenology\Method2\RF\X_RADAR_CORN_v21.xlsx'
        y_path = R'G:\Phenology\Method2\RF\Y_Crop_corn_v2.xlsx'
elif ty==6:
        x_path = R'G:\Phenology\Method2\RF\x_radar_soybean_v21.xlsx'
        y_path = R'G:\Phenology\Method2\RF\y_crop_soybean_v2.xlsx'
elif ty==7:
        x_path = R'G:\Phenology\Method2\RF\X_RADAR_Wheat_v21.xlsx'
        y_path = R'G:\Phenology\Method2\RF\Y_Radar_wheat_v2.xlsx'

x = pd.read_excel(x_path).iloc[:,0:17].values # pandas.read_excel(fullpath).iloc[row,column index].values is to read the data
y = pd.read_excel(y_path).iloc[:,0].values

#random.seed(42)
# divide the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0) # train_test_split is to divide the full data into training and test dataset,

print("Here's the dimensions of our data frame:",x.shape)

#%% Construct the ANN

# Feature Scaling for x, rather than y
sc = xscaler() # replace the standardscaler as sc
x_train = sc.fit_transform(x_train) # maybe better to change to different varibale name, standardscaler.fit_transform is scale the training dataset
x_test = sc.transform(x_test) # standardscaler.transform is to scale the test dataset. It is reasonable that both the training and test datasets need to scaled in the same method

#training
y_preds = [] # it will be a maxtrix, one dimension for the tree number, another is for the number of records
score = [] # has the same dimension as y_preds

model = Sequential()
model.add(Dense(100, input_dim=17, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

# Fit the model
model.fit(x_train, y_train, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(x_test)

print('OK')

# %% Plot the the results

y_pred_final=predictions # predict the phenology values using the determined ANN

#for y_pred in y_preds:
print('y_test:', y_test)
print('\n');
print('y_pred_final:',y_pred_final)
    
#%% Plot the results
  
plt.scatter(y_test,y_pred_final,marker='o',c='k')
plt.plot([0, 100], [0, 100], "k--", linewidth=1)
# Setting the title and label
#plt.title('Phenology', fontsize=18)
plt.xlabel('Ground identified BBCH', fontsize=14)
plt.ylabel('Retrieved BBCH', fontsize=14)
plt.ylim([0,100])
plt.xlim([0,100])

R=np.corrcoef(y_test,np.squeeze(np.asarray(y_pred_final)))
rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred_final))
#plt.annotate("RMSE="+str(int(round(rmse,2))), (20, 75),fontsize=14)
plt.annotate("R="+str(round(R[0][1],2)), (20, 80),fontsize=14)
if ty==1:
   plt.annotate("Canola", (40, 95), fontsize=14)
elif ty==2:
   plt.annotate("Corn", (40, 95), fontsize=14)
elif ty==6:
   plt.annotate("Soybean", (40, 95), fontsize=14)
elif ty==7:
   plt.annotate("Wheat", (40, 95), fontsize=14)
