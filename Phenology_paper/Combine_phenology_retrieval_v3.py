# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:11:33 2018
@author: HONGQUAN
"""
# Random Forest Algorithm for retrieve the crop phenology
# scipy package: need to import and learn
import time
import numpy as np # similar to matrix computation
import pandas as pd # read the dataset, data manipulation and analysis
import matplotlib.pyplot as plt # plot the results
from sklearn.ensemble import RandomForestRegressor #sklearn includes the random forest algorithms
from sklearn.model_selection import train_test_split,GridSearchCV # includes the train_test_split
from sklearn.preprocessing import StandardScaler as xscaler # includes the preprocessing,standardscaler is to prepare the training dataset
from sklearn import metrics #includes the metrics

#%% Importing the datasets
x_path = R'G:\Phenology\Method2\RF\X_RADAR_Wheat.xlsx'
x = pd.read_excel(x_path).iloc[:,0:16].values # pandas.read_excel(fullpath).iloc[row,column index].values is to read the data
y_path = R'G:\Phenology\Method2\RF\Y_Radar_wheat.xlsx'
y = pd.read_excel(y_path).iloc[:,0].values

# divide the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0) # train_test_split is to divide the full data into training and test dataset,

print("Here's the dimensions of our data frame:",x.shape)

#%% Hyper optimization of the randomforestregressor

#RandomForestRegressor(n_estimators=’warn’, criterion=’mse’, 
#                      max_depth=None, min_samples_split=2, min_samples_leaf=1, 
#                      min_weight_fraction_leaf=0.0, max_features=’auto’, 
#                      max_leaf_nodes=None, min_impurity_decrease=0.0, 
#                      min_impurity_split=None, bootstrap=True, oob_score=False, 
#                      n_jobs=None, random_state=None, verbose=0, warm_start=False)

# Set the random state for reproductibility
fit_rf = RandomForestRegressor(random_state=0)

# Hyperparameter Optimization
# Utilizing the GridSearchCV functionality, let's create a dictionary 
# with parameters we are looking to optimize to create 
# the best model for our data. 
# Setting the n_jobs to 3 tells the grid search to run three jobs in parallel

#np.random.seed(42)

start = time.time()
param_dist = {'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['mse', 'mae']}
#cv_rf = GridSearchCV(fit_rf, cv = 10,
#                     param_grid=param_dist,
#                    n_jobs = 3)
cv_rf = GridSearchCV(fit_rf, cv = 5,
                     param_grid=param_dist)
cv_rf.fit(x_train, y_train)
print('Best Parameters using grid search: \n',cv_rf.best_params_)
end = time.time()

print('Time taken in grid search: {0: .2f}'.format(end - start))

print('OK')


#%% use the best parameters

# Set best parameters given by grid search
fit_rf.set_params(criterion = 'mse',max_features = 'auto')

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
    fit_rf = RandomForestRegressor(n_estimators = n, random_state = 0,criterion = 'mse',max_features = 'auto') # two parameters of RF, number of tree, randomless
    fit_rf.fit(x_train, y_train)  # fit the RF, x_train, y_train are fixed. no equator. This is a training process
    y_preds.append((fit_rf.predict(x_test), n)) # predict the y_preds for different tree number. data.append is to combine the prediction without deleting the previous data
    sco = fit_rf.score(x_test, y_test) # forest.score is to evalute the performance of the defined forest - forest with a tree number n
    score.append((sco, n)) # synax is data.append((variable1, variable2, variable 3)). The saved variables are (score, number of tree)
    n = n + 1
# after this process, y_preds is a list of different prediction at given tree number
print('OK')
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
  
plt.scatter(y_test,y_pred_final,marker='o',c='k')
#plt.plot(y_test, y_pred_final,'ko')
plt.plot([0, 100], [0, 100], "k--", linewidth=1)
# Setting the title and label
#plt.title('Phenology', fontsize=18)
plt.xlabel('Ground identified BBCH', fontsize=14)
plt.ylabel('Retrieved BBCH', fontsize=14)
plt.ylim([0,100])
plt.xlim([0,100])
#plt.show()
R=np.corrcoef(y_test, y_pred_final)
rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred_final))
plt.annotate("RMSE="+str(round(rmse,2)), (20, 75),fontsize=14)
plt.annotate("R="+str(round(R[0][1],2)), (20, 80),fontsize=14)
plt.annotate("Wheat", (40, 95), fontsize=14)

