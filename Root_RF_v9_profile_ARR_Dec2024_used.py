import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR  # Import SVM Regressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor  
from sklearn.preprocessing import StandardScaler as xscaler # includes the preprocessing,standardscaler is to prepare the training dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from xgboost import XGBRegressor

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
import pickle

# sys.path.append(r'L:\HSI_Root_Rot\Method\funs')
# from calculate_metrics import nan_stat_eva_model  


# %%  Step 1: Read the Hyperspectral Shoot data in Excel
shoot_hsi = 'G:/HSI_Root_Rot/Data/Specim_ARR_02122024/Spectral_shoot_DecG8.xlsx'
root_hsi = 'G:/HSI_Root_Rot/Data/Specim_ARR_02122024/Spectral_root_DecG8.xlsx'

df_s1 = pd.read_excel(shoot_hsi, sheet_name='ShootR1toR5', header=0).astype(float)
df_s2 = pd.read_excel(shoot_hsi, sheet_name='ShootR6toR10', header=0).astype(float)
df_s3 = pd.read_excel(shoot_hsi, sheet_name='ShootR11toR15', header=0).astype(float)

waveleth = df_s1.iloc[:, 0].values
dec_2024_Shoot = np.hstack([df_s1.iloc[:, 1:].values, df_s2.iloc[:, 1:].values, df_s3.iloc[:, 1:].values])

df_t1 = pd.read_excel(root_hsi, sheet_name='RootR1toR5', header=0).astype(float)
df_t2 = pd.read_excel(root_hsi, sheet_name='RootR6toR10', header=0).astype(float)
df_t3 = pd.read_excel(root_hsi, sheet_name='RootR11toR15', header=0).astype(float)

dec_2024_root = np.hstack([df_t1.iloc[:, 1:].values, df_t2.iloc[:, 1:].values, df_t3.iloc[:, 1:].values])

dec_truth = pd.read_excel('G:/HSI_Root_Rot/Data/Truth_December2024_v2.xlsx', sheet_name='Feuil1', header=0)
labe_shoot = dec_truth.iloc[:, -3].values.astype(float)
labe_root = dec_truth.iloc[:, -1].values.astype(float)

# Plot Shoot Data
plt.figure()
for i in range(8):
    plt.plot(waveleth.astype(float), dec_2024_Shoot[:, i].astype(float))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.show()

# Plot Root Data
plt.figure()
for i in range(8):
    plt.plot(waveleth, dec_2024_root[:, i])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.show()

#%% Step 2:Preprocessing 
# ###############################################
# Option 1: Remove invaludate values
# X = dec_2024_root.T
# X = X[:, :-3]  # Remove last three columns
# X_col = X[:, 1]
# ind1 = np.where(np.isnan(X_col))[0]
# y = labe_root
# ind2 = np.where(np.isnan(y))[0]
# ind = np.sort(np.concatenate([ind1, ind2]))

# X = np.delete(X, ind, axis=0)
# y = np.delete(y, ind)
###############################################

###############################################
# Option 2: Remove invaludate values
X = dec_2024_root.T
X = X[:, :-3]
y = labe_root

# Remove NaN values
nan_mask = ~np.isnan(X[:, 1]) & ~np.isnan(y)
X = X[nan_mask]
y = y[nan_mask]
###############################################

#################################################
# Option -1: Split the training and test dateset 
# Split data into training and testing sets
# split_ratio = 0.8
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=50)
###################################################

##################################################
# Option -2:  Split the training and test dateset
# Set random seed for reproducibility
np.random.seed(50)
# Split data into training and testing sets
splitRatio = 0.8
splitIdx = np.random.permutation(len(X))
trainIdx = splitIdx[:int(splitRatio * len(X))]
testIdx = splitIdx[int(splitRatio * len(X)):] 
X_train, X_test = X[trainIdx], X[testIdx]
y_train, y_test = y[trainIdx], y[testIdx]
###############################################

#%% Step 3: Test different AI ML algorithms

# Feature Scaling for x, rather than y
sc = xscaler() # replace the standardscaler as sc
x_train = sc.fit_transform(X_train) # maybe better to change to different varibale name, standardscaler.fit_transform is scale the training dataset
x_test = sc.transform(X_test) # standardscaler.transform is to scale the test dataset. It is reasonable that both the training and test datasets need to scaled in the same method


# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': range(5,100,20),      # Number of trees
    'max_depth': [10, 20, 30],     # Maximum depth of each tree
    'max_leaf_nodes': [10, 20, 30], # Maximum number of leaf nodes
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit the GridSearchCV
grid_search.fit(x_train, y_train)

# Get the best parameters and model from grid search
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

# Predict using the best model
y_pred = best_rf_model.predict(x_test)


#%% Step 4: Evaluate the performance of the algorithms

# Evaluate model performance
r_squared = r2_score(y_test, y_pred.flatten())
rmse = root_mean_squared_error(y_test, y_pred.flatten())
cor = np.corrcoef(y_test, y_pred.flatten())

print(f'R2 on Test Data: {r_squared:.4f}')
print(f'RMSE: {rmse:.4f}')


# Plot actual vs predicted
plt.figure()
plt.scatter(y_test, y_pred, c='k', marker='o')
plt.text(6, 1.5, rf'$R^2 = {r_squared:.2f}$')
plt.text(6, 1, f'RMSE = {rmse:.2f}')
plt.xlabel('Visual Rating')
plt.ylabel('Estimated Root Rot')
plt.title('Pea Root Rot')
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.show()

# Save the model to a file
# with open(r'L:\HSI_Root_Rot\Method\rf_model2.pkl', 'wb') as f:
#     pickle.dump(rf_model, f)


#%% Step 5: Variable Importance/this could be very interesting for your paper

####### Option 1: unsorted importance
importances_rf = best_rf_model.feature_importances_  # the summation of the importance equal to 1

plt.figure()
plt.scatter(waveleth[:-3], importances_rf, c='k', marker='x')
mx = 0.1
plt.axvline(x=400, color='b')
plt.axvline(x=500, color='g')
plt.axvline(x=600, color='r')
plt.axvline(x=680, color='k')
plt.axvline(x=750, color='m')
plt.axvline(x=970, color='y')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Importance of wavelength')
plt.ylim([0, mx])
plt.xlim([300, 1100])
plt.show()


####### Option 2: Sorted importance for little number of variables
importances_rf = best_rf_model.feature_importances_  # the summation of the importance equal to 1
indices_rf = np.argsort(importances_rf)[::-1] # the index of the sorted importance from the largest to the smallest
names_index = waveleth[:-3]

def variable_importance(importance, indices):  
    """
    Purpose:
    ----------
    Prints dependent variable names ordered from largest to smallest
    based on gini or information gain for CART model.

    Parameters:
    ----------
    names:      Name of columns included in model
    importance: Array returned from feature_importances_ for CART
                   models organized by dataframe index
    indices:    Organized index of dataframe from largest to smallest
                   based on feature_importances_
    Returns:
    ----------
    Print statement outputting variable importance in descending order
    """   
    print('Feature ranking:')
    for i in range(len(names_index)):
        print("%d. The feature '%s' contributes to the phenology retrieval of %f%%"\
              % (i + 1,names_index[i],importance[i]*100))
        
variable_importance(importances_rf, indices_rf) 

#variable_importance_plot(importance, indices):
index=np.arange(len(names_index))
importance_asc = sorted(importances_rf) #%% from least to most important

feature_space = []   
for i in range(15, -1, -1):
   feature_space.append(names_index[indices_rf[i]])
#    fig, ax = plt.subplots(figsize=(10, 10))
#    ax.set_axis_bgcolor('#fafafa')
#plt.title('Importances of polarimetric parameters for crop phenology retrieval')
plt.figure(1)
plt.barh(index,np.asarray(importance_asc)*100,align="center",color = 'k')
plt.yticks(index,feature_space)
plt.xlim(0,60)
plt.ylim(-1, 16)
#plt.xlim(0, max(importance_desc))
plt.xlabel('Contribution in phenology retrieval (%)')
plt.ylabel('Polarimetric parameters')
plt.annotate("Soybean", (45, 1.5), fontsize=14)


#%% Load and use the models

# Later, load the model back from the file
with open(r'L:\HSI_Root_Rot\Method\ann_model2.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    

y_pred = loaded_model.predict(X_test)

# Evaluate model performance
R, bias, sd, rmse_s, mae, d, R2 = nan_stat_eva_model(y_pred, y_test)
print(f'R on Test Data: {R[0, 1]:.4f}')

# Plot actual vs predicted
plt.figure()
plt.scatter(y_test, y_pred, c='k', marker='o')
plt.text(6, 1.5, f'R = {R[0,1]:.2f}')
plt.text(6, 1, f'RMSE = {rmse_s:.2f}')
plt.xlabel('Visual Rating')
plt.ylabel('Estimated Root Rot')
plt.title('Pea Root Rot')
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.show()


