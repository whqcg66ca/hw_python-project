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
# Define file paths
path_hsi = r'L:\HSI_Root_Rot\Data\HSI Spectra RootRot_MAIN.xlsx'
path_truth = r'L:\HSI_Root_Rot\Data\Truth3.xlsx'

# Read shoot hyperspectral data
ARR_2024_Shoot = pd.read_excel(path_hsi, sheet_name='ARR_2024_Shoot', header=0)
waveleth = ARR_2024_Shoot.iloc[:, 0]  # First column
ARR_Shoot_Cont = ARR_2024_Shoot.iloc[:, 1:17]  # Columns 2 to 17
ARR_Shoot_Rep1 = ARR_2024_Shoot.iloc[:, 17:17+16]  # Columns 18 to 33
ARR_Shoot_Rep2 = ARR_2024_Shoot.iloc[:, 17+16:17+16+16]  # Columns 34 to 49

# Read root hyperspectral data
ARR_2024_Root = pd.read_excel(path_hsi, sheet_name='ARR_2024_Root', header=0)
ARR_Root_Cont = ARR_2024_Root.iloc[:, 1:17]  # Columns 2 to 17
ARR_Root_Rep1 = ARR_2024_Root.iloc[:, 17:17+16]  # Columns 18 to 33
ARR_Root_Rep2 = ARR_2024_Root.iloc[:, 17+16:17+16+16]  # Columns 34 to 49

# Read truth labels
# ARR_truth_txt = pd.read_excel(path_truth, sheet_name='ARR', header=None)
# labe_cont = ARR_truth_txt.iloc[0:16, 1].astype(str).tolist()
# labe_rep1 = ARR_truth_txt.iloc[16:32, 1].astype(str).tolist()
# labe_rep2 = ARR_truth_txt.iloc[32:, 1].astype(str).tolist()

# Plot shoot data
plt.figure()
for i in range(16):
    plt.plot(waveleth, ARR_Shoot_Cont.iloc[:, i])
# plt.legend(labe_cont)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.show()

# plt.figure()
# for i in range(16):
#     plt.plot(waveleth, ARR_Shoot_Rep1.iloc[:, i])
# # plt.legend(labe_rep1)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')
# plt.show()

# plt.figure()
# plt.plot(waveleth, ARR_Shoot_Rep2.iloc[:, 0], '-k', 
#          waveleth, ARR_Shoot_Rep2.iloc[:, 1], '-r', 
#          waveleth, ARR_Shoot_Rep2.iloc[:, 2], '-b')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')
# plt.show()

# Plot root data
plt.figure()
for i in range(16):
    plt.plot(waveleth, ARR_Root_Cont.iloc[:, i])
# plt.legend(labe_cont)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.show()

# plt.figure()
# for i in range(16):
#     plt.plot(waveleth, ARR_Root_Rep1.iloc[:, i])
# # plt.legend(labe_rep1)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')
# plt.show()

# plt.figure()
# plt.plot(waveleth, ARR_Root_Rep2.iloc[:, 0], '-k', 
#          waveleth, ARR_Root_Rep2.iloc[:, 1], '-r', 
#          waveleth, ARR_Root_Rep2.iloc[:, 2], '-b')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')
# plt.show()

# # Root-to-shoot reflectance ratio
# rr = ARR_2024_Root.iloc[:, 1:].to_numpy()
# ss = ARR_2024_Shoot.iloc[:, 1:].to_numpy()

# plt.figure()
# plt.plot(rr, ss, '.k')
# plt.xlabel('Root Reflectance')
# plt.ylabel('Shoot Reflectance')
# plt.show()

# plt.figure()
# plt.plot(waveleth, np.nanmean(rr / ss, axis=1), '-k')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance (Root/Shoot)')
# plt.show()

# Read ground truth data
ARR_truth = pd.read_excel(path_truth, sheet_name='ARR', header=0)

XX_Shoot = np.vstack([ARR_Shoot_Cont.to_numpy().T, ARR_Shoot_Rep1.to_numpy().T, ARR_Shoot_Rep2.to_numpy().T])
XX_Root = np.vstack([ARR_Root_Cont.to_numpy().T, ARR_Root_Rep1.to_numpy().T, ARR_Root_Rep2.to_numpy().T])
YY = ARR_truth.iloc[:, 6].to_numpy()

#%% Step 2:Preprocessing 
###############################################
# Option 2: Remove invaludate values
X = XX_Root
X = X[:, :-3]
y = YY.astype(float)

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