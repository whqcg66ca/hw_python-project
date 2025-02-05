# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 08:57:48 2018

@author: TL7050
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:11:33 2018
@author: HONGQUAN
"""
# Random Forest Algorithm for retrieve the crop phenology
# scipy package: need to import and learn
#import time
import numpy as np # similar to matrix computation
import pandas as pd # read the dataset, data manipulation and analysis
import matplotlib.pyplot as plt # plot the results
#from sklearn.ensemble import RandomForestRegressor #sklearn includes the random forest algorithms
from sklearn.svm import SVR
#from sklearn import svm
#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # includes the train_test_split
from sklearn.preprocessing import StandardScaler as xscaler # includes the preprocessing,standardscaler is to prepare the training dataset
#from sklearn import metrics #includes the metrics

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

# divide the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0) # train_test_split is to divide the full data into training and test dataset,

print("Here's the dimensions of our data frame:",x.shape)

#%% Fitting the Random Forest with different tree numbers from 1 to 100

# Feature Scaling for x, rather than y
sc = xscaler() # replace the standardscaler as sc
x_train = sc.fit_transform(x_train) # maybe better to change to different varibale name, standardscaler.fit_transform is scale the training dataset
x_test = sc.transform(x_test) # standardscaler.transform is to scale the test dataset. It is reasonable that both the training and test datasets need to scaled in the same method

#training
y_preds = [] # it will be a maxtrix, one dimension for the tree number, another is for the number of records
score = [] # has the same dimension as y_preds


#svr=svm.SVR()

#svr=svm.LinearSVR()

######### Results in paper
svr = SVR(kernel='linear')

##### Resykts with C

#svr = SVR(kernel='sigmoid', C=2)

#### Resuls with different kernel function

#svr = SVR(kernel='rbf', gamma=0.16, C=10)

#C=1
#svr = SVR(kernel='poly', degree=8, C=C)

svr.fit(x_train,y_train)

pred_SVR = svr.predict(x_test)

# %% Plot the the results
#Opti_tree_number= rmse[0][1] # transfer the obtained optimal tree number
#
#knn = KNeighborsRegressor(n_neighbors=Opti_tree_number)
##forest = RandomForestRegressor(n_estimators = Opti_tree_number, random_state = 0) # condtruct the obtained optimal random forest
#knn.fit(x_train, y_train)  # train the corresponding best random forest
y_pred_final=pred_SVR# predict the phenology values using the determined random forest structure

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
#plt.show()
R=np.corrcoef(y_test, y_pred_final)
#rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred_final))
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

