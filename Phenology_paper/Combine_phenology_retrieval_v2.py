# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:11:33 2018
@author: HONGQUAN
"""
# Random Forest Algorithm for retrieve the crop phenology
# scipy package: need to import and learn
import numpy as np # similar to matrix computation
import pandas as pd # read the dataset, data manipulation and analysis
import matplotlib.pyplot as plt # plot the results
from sklearn.ensemble import RandomForestRegressor #sklearn includes the random forest algorithms
from sklearn.model_selection import train_test_split # includes the train_test_split
from sklearn.preprocessing import StandardScaler as xscaler # includes the preprocessing,standardscaler is to prepare the training dataset
from sklearn import metrics #includes the metrics

#%% Importing the datasets
x_path = R'G:\Phenology\Method2\RF\X_RADAR_Canola.xlsx'
x = pd.read_excel(x_path).iloc[:,0:15].values # pandas.read_excel(fullpath).iloc[row,column index].values is to read the data
y_path = R'G:\Phenology\Method2\RF\y_BBCH_Canola.xlsx'
y = pd.read_excel(y_path).iloc[:,0].values

# divide the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0) # train_test_split is to divide the full data into training and test dataset,

#%% Fitting the Random Forest with different tree numbers from 1 to 100

# Feature Scaling for x, rather than y
sc = xscaler() # replace the standardscaler as sc
x_train = sc.fit_transform(x_train) # maybe better to change to different varibale name, standardscaler.fit_transform is scale the training dataset
x_test = sc.transform(x_test) # standardscaler.transform is to scale the test dataset. It is reasonable that both the training and test datasets need to scaled in the same method

#training
y_preds = [] # it will be a maxtrix, one dimension for the tree number, another is for the number of records
score = [] # has the same dimension as y_preds

n = 1
#when n changes, the forest attributes change correspondingly. 
# we can also use for loop here, I think, the main difference is that, for doest need the n=n+1, but while is followd by a condition,
while n <= 100:
    forest = RandomForestRegressor(n_estimators = n, random_state = 0) # two parameters of RF, number of tree, randomless
    forest.fit(x_train, y_train)  # fit the RF, x_train, y_train are fixed. no equator. This is a training process
    y_preds.append((forest.predict(x_test), n)) # predict the y_preds for different tree number. data.append is to combine the prediction without deleting the previous data
    sco = forest.score(x_test, y_test) # forest.score is to evalute the performance of the defined forest - forest with a tree number n
    score.append((sco, n)) # synax is data.append((variable1, variable2, variable 3)). The saved variables are (score, number of tree)
    n = n + 1
# after this process, y_preds is a list of different prediction at given tree number

#%% Evaluation to select the optimal tree number by calculating the mae and rmse for each tree number condition
mae = [] # this is to incude the mean absolute error
rmse = [] # this is to include the RMSE

for y_pred in y_preds: # obtain each predicted y (54 elements) totally, in y_preds. y_pred is null for each loop cycle.
    #print('y_test', y_test)  # print the obtained values for the tree number from 1 to 100. y_test is the same for 100 tree condition, as it is our original dataset
    #print('y_pred', y_pred[0]) # print the predicted value for each tree number condition from 1 to 100. It is different due to the change in tree number
    print('---------------------------------------------------')
    print('Tree number:', y_pred[1])
    mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred[0]) # calculate the single value of the mean absolute error for ech tree condition.
    mae.append((mean_absolute_error, y_pred[1])) # combine the singles mae into a list, and also the sequence of tree number
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred[0])) # metrics.mean_sqared_error
    rmse.append((root_mean_squared_error, y_pred[1])) # data.append((variable1,variable2,...)) generates a list data type
    print('Mean Absolute Error: ', mean_absolute_error)
    print('Root Mean Squared Error:', root_mean_squared_error)
    #print('Random forest score:', sc[1])
    print('\n')

mae.sort() # sort according to the first dimension, default. Choose the minimun mae
rmse.sort() # sort according to the first dimension, choose the niminum rmse
score.sort()
score.reverse() # another data function, data.reverse, in opposit to the sorted results. Choose the largest score


# print can show the combination between number and test variables. 
# the index of the data, is data[][]
print("Based on minumum Mean Absolute Error:", mae[0][0],"the best estimator is a random forest system with tree number", mae[0][1])
print('\n')
print("Based on minimum RMSE:", rmse[0][0],"the best estimator is a random forest system with tree number", rmse[0][1])
print('\n')
print("Based on maximum score:", score[0][0],"the best estimator is a random forest system with tree number", score[0][1])
print('\n')

# %% Plot the the results
Opti_tree_number= rmse[0][1] # transfer the obtained optimal tree number

forest = RandomForestRegressor(n_estimators = Opti_tree_number, random_state = 0) # condtruct the obtained optimal random forest
forest.fit(x_train, y_train)  # train the corresponding best random forest
y_pred_final=forest.predict(x_test) # predict the phenology values using the determined random forest structure

#for y_pred in y_preds:
print('y_test:', y_test)
print('\n');
print('y_pred_final:',y_pred_final)
    
#%% Plot the results
  
plt.scatter(y_test, y_pred_final)
plt.plot([0, 100], [0, 100], "k--", linewidth=1)
# Setting the title and label
plt.title('Phenology', fontsize=18)
plt.xlabel('Measured BBCH', fontsize=14)
plt.ylabel('Retrieved BBCH', fontsize=14)
plt.ylim([0,100])
plt.xlim([0,100])
#plt.show()
R=np.corrcoef(y_test, y_pred_final)
plt.annotate("R="+str(round(R[0][1],2)), (20, 80))
