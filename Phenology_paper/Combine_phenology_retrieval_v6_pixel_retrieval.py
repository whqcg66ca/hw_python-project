# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:45:18 2019

@author: HONGQUAN
"""
im=22

for ty in [1,2,6,7]:
# Random Forest Algorithm for retrieve the crop phenology
# scipy package: need to import and learn
#    import system as sy
    import time
    import numpy as np # similar to matrix computation
    import pandas as pd # read the dataset, data manipulation and analysis
    import matplotlib.pyplot as plt # plot the results
    from sklearn.ensemble import RandomForestRegressor #sklearn includes the random forest algorithms
    from sklearn.model_selection import train_test_split,GridSearchCV # includes the train_test_split
    from sklearn.preprocessing import StandardScaler as xscaler # includes the preprocessing,standardscaler is to prepare the training dataset
    from sklearn import metrics #includes the metrics
    import scipy.io

#%% Importing the datasets

#    sy.reset_selective 
    names = ['HV', 'RVI', 'Entropy',
             'q', 'PV', '$ \\alpha_1 $',
             '$ \\alpha $', 'PH',
             'Pr','$\phi_{hhvv}$',
             'PF', 'PA',
             '$\\rho_{hhvv}$', 'DERD', 'SERD',
             'SDERD','$ \\theta $']
    if ty==1:
        x_path = R'G:\Phenology\Method2\RF\X_RADAR_Canola_v21.xlsx' ## USE V21
        y_path = R'G:\Phenology\Method2\RF\y_BBCH_Canola_v2.xlsx'
        nm_in='X_canola_17p.mat'
        nm_out='BBCH_pre_canola_17p.mat'
    elif ty==2:
        x_path = R'G:\Phenology\Method2\RF\X_RADAR_CORN_v21.xlsx'
        y_path = R'G:\Phenology\Method2\RF\Y_Crop_corn_v2.xlsx'
        nm_in='X_corn_17p.mat'
        nm_out='BBCH_pre_corn_17p.mat'
    elif ty==6:
        x_path = R'G:\Phenology\Method2\RF\x_radar_soybean_v21.xlsx'
        y_path = R'G:\Phenology\Method2\RF\y_crop_soybean_v2.xlsx'
        nm_in='X_soybean_17p.mat'
        nm_out='BBCH_pre_soybean_17p.mat'
    elif ty==7:
        x_path = R'G:\Phenology\Method2\RF\X_RADAR_Wheat_v21.xlsx'
        y_path = R'G:\Phenology\Method2\RF\Y_Radar_wheat_v2.xlsx'
        nm_in='X_wheat_17p.mat'
        nm_out='BBCH_pre_wheat_17p.mat'
    #x = pd.read_excel(x_path).iloc[:,0:16].values # pandas.read_excel(fullpath).iloc[row,column index].values is to read the data
    
    x = pd.read_excel(x_path,names=names) # pandas.read_excel(fullpath).iloc[row,column index].values is to read the data
    
    #y = pd.read_excel(y_path).iloc[:,0].values
    y = pd.read_excel(y_path)
    
    # divide the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0) # train_test_split is to divide the full data into training and test dataset,
    
    print("Here's the dimensions of our data frame:",x.shape)
    
    
    
    toto=pd.read_excel('I:\SMP16_RS2\RS2_INFO.xlsx',sheet_name='RS2_mosaic', header=None)
    rname=toto[0:33][5]
    if im==16:
        path=''.join(['I:/SMP16_RS2/',rname[im-1],'/subset_mosaic.data/C3/',nm_in])
    else:
        path=''.join(['I:/SMP16_RS2/',rname[im-1],'/mosaic_reprojected.data/C3/',nm_in])
    ###################### Change the path for each day ##########################
    canola_p= scipy.io.loadmat(path)
    print('Done load')
    ##############################################################################
    
    ### Extract the 16 polarimetric parameters and one incidence angle
    mylist=[]
    for key in canola_p.keys():
        mylist.append(canola_p[key])
    X_v=mylist[3]
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
    
    # %% Print the the results using optimal decision tree numbers
    Opti_tree_number= rmse[0][1] # transfer the obtained optimal tree number
    x_valide = sc.transform(X_v)
    forest = RandomForestRegressor(n_estimators = Opti_tree_number, random_state = 0) # condtruct the obtained optimal random forest
    forest.fit(x_train, y_train)  # train the corresponding best random forest
    y_pred_final=forest.predict(x_valide) # predict the phenology values using the determined random forest structure

    
    #for y_pred in y_preds:
    #print('y_test:', y_test)
    #print('\n');
    print('y_pred_final:',y_pred_final)
    plt.figure
    plt.scatter(range(len(y_pred_final)),y_pred_final)
    
    
    if im==16:
        path_out=''.join(['I:/SMP16_RS2/',rname[im-1],'/subset_mosaic.data/C3/',nm_out])
    else:        
        path_out=''.join(['I:/SMP16_RS2/',rname[im-1],'/mosaic_reprojected.data/C3/',nm_out])
    
    scipy.io.savemat(path_out, {'BBCH_pre':y_pred_final})
    print('Done load')
    del y_pred_final
#x_train = sc.fit_transform(x_train) # maybe better to change to different varibale name, standardscaler.fit_transform is scale the training dataset
#x_test = sc.transform(x_test)
#    




















##%% Plot the results
#  
#plt.scatter(y_test,y_pred_final,marker='o',c='k')
##plt.plot(y_test, y_pred_final,'ko')
#plt.plot([0, 100], [0, 100], "k--", linewidth=1)
## Setting the title and label
##plt.title('Phenology', fontsize=18)
#plt.xlabel('Ground identified BBCH', fontsize=14)
#plt.ylabel('Retrieved BBCH', fontsize=14)
#plt.ylim([0,100])
#plt.xlim([0,100])
##plt.show()
#R=np.corrcoef(np.squeeze(np.asarray(y_test)), y_pred_final)
#rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred_final))
#plt.annotate("RMSE="+str(int(round(rmse))), (20, 75),fontsize=14)
#plt.annotate("R="+str(round(R[0][1],2)), (20, 80),fontsize=14)
#plt.annotate("Wheat", (40, 95), fontsize=14)
#
##%% Variable Importance: this could be very interesting for your paper
#
#importances_rf = forest.feature_importances_  # the summation of the importance equal to 1
#indices_rf = np.argsort(importances_rf)[::-1] # the index of the sorted importance from the largest to the smallest
#names_index = names[:]
#
#def variable_importance(importance, indices):  
#    """
#    Purpose:
#    ----------
#    Prints dependent variable names ordered from largest to smallest
#    based on gini or information gain for CART model.
#
#    Parameters:
#    ----------
#    names:      Name of columns included in model
#    importance: Array returned from feature_importances_ for CART
#                   models organized by dataframe index
#    indices:    Organized index of dataframe from largest to smallest
#                   based on feature_importances_
#    Returns:
#    ----------
#    Print statement outputting variable importance in descending order
#    """   
#    print('Feature ranking:')
#    for i in range(len(names_index)):
#        print("%d. The feature '%s' contributes to the phenology retrieval of %f%%"\
#              % (i + 1,names_index[i],importance[i]*100))
#        
#variable_importance(importances_rf, indices_rf) 
#
##%% define a function to plot the importance (not useful)
#
##variable_importance_plot(importance, indices):
#index=np.arange(len(names_index))
#importance_asc = sorted(importances_rf) #%% from least to most important
#
#feature_space = []   
#for i in range(15, -1, -1):
#   feature_space.append(names_index[indices_rf[i]])
##    fig, ax = plt.subplots(figsize=(10, 10))
##    ax.set_axis_bgcolor('#fafafa')
##plt.title('Importances of polarimetric parameters for crop phenology retrieval')
#plt.barh(index,np.asarray(importance_asc)*100,align="center",color = 'k')
#plt.yticks(index,feature_space)
#plt.xlim(0,60)
#plt.ylim(-1, 16)
##plt.xlim(0, max(importance_desc))
#plt.xlabel('Contribution in phenology retrieval (%)')
#plt.ylabel('Polarimetric parameters')
#plt.annotate("Soybean", (45, 1.5), fontsize=14)
#plt.show()
#    plt.close()
#variable_importance_plot(importances_rf, indices_rf)