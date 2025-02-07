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

 
# %% Step 1: Read the Hyperspectral Shoot data in Excel
file_path = 'L:/HSI_Root_Rot/Data/HSI Spectra RootRot_MAIN.xlsx'
arr_2024_shoot = pd.read_excel(file_path, sheet_name='ARR_2024_Shoot').values

waveleth = arr_2024_shoot[:, 0]
arr_shoot_cont = arr_2024_shoot[:, 1:17]
arr_shoot_rep1 = arr_2024_shoot[:, 17:17+16]
arr_shoot_rep2 = arr_2024_shoot[:, 17+16:17+16+16]

# Plot Reflectance for Shoot

plt.figure()
for i in range(15):
    plt.plot(waveleth, arr_shoot_cont[:, i])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.show()

# Read root data
arr_2024_root = pd.read_excel(file_path, sheet_name='ARR_2024_Root').values
arr_root_cont = arr_2024_root[:, 1:17]
arr_root_rep1 = arr_2024_root[:, 17:17+16]
arr_root_rep2 = arr_2024_root[:, 17+16:17+16+16]

# Plot Reflectance for Root Cont
plt.figure()
plt.plot(waveleth, arr_root_cont[:, 0], '-k', label='Cont 1')
plt.plot(waveleth, arr_root_cont[:, 1], '-r', label='Cont 2')
plt.plot(waveleth, arr_root_cont[:, 2], '-b', label='Cont 3')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.show()

# Root/Shoot Reflectance ratio
rr = arr_2024_root[:, 1:]
ss = arr_2024_shoot[:, 1:]
plt.figure()
plt.plot(waveleth, np.nanmean(rr/ss, axis=1), '-k')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (Root/Shoot)')
plt.show()

# Read ground truth data
arr_truth = pd.read_excel('L:/HSI_Root_Rot/Data/Truth3.xlsx', sheet_name='ARR').values
xx_shoot = np.concatenate([arr_shoot_cont.T, arr_shoot_rep1.T, arr_shoot_rep2.T])
xx_root = np.concatenate([arr_root_cont.T, arr_root_rep1.T, arr_root_rep2.T])
yy = arr_truth[:, 6]

#%% Step 2:Preprocessing 
###############################################
# Option 2: Remove invaludate values
X = xx_root
X = X[:, :-3]
y = yy.astype(float)

# Remove NaN values
nan_mask = ~np.isnan(X[:, 1]) & ~np.isnan(y)
X = X[nan_mask]
y = y[nan_mask]
###############################################

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


#%% Evaluate the performance of the algorithms
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