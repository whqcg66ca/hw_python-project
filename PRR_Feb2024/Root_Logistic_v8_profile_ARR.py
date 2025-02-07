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
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from xgboost import XGBRegressor

import tensorflow as tf
from keras.layers import Dense
from keras import Sequential

import pickle


# from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(r'D:\HSI_Root_Rot\Method\funs')
from calculate_metrics import nan_stat_eva_model  

# Read the hyperspectral data
file_path = 'D:/HSI_Root_Rot/Data/HSI Spectra RootRot_MAIN.xlsx'
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

# # Plot Reflectance for Rep1
# plt.figure()
# plt.plot(waveleth, arr_shoot_rep1[:, 0], '-k', label='Rep1 1')
# plt.plot(waveleth, arr_shoot_rep1[:, 1], '-r', label='Rep1 2')
# plt.plot(waveleth, arr_shoot_rep1[:, 2], '-b', label='Rep1 3')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')
# plt.legend()
# plt.show()

# # Plot Reflectance for Rep2
# plt.figure()
# plt.plot(waveleth, arr_shoot_rep2[:, 0], '-k', label='Rep2 1')
# plt.plot(waveleth, arr_shoot_rep2[:, 1], '-r', label='Rep2 2')
# plt.plot(waveleth, arr_shoot_rep2[:, 2], '-b', label='Rep2 3')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')
# plt.legend()
# plt.show()

# Read root data
arr_2024_root = pd.read_excel(file_path, sheet_name='ARR_2024_Root').values
arr_root_cont = arr_2024_root[:, 1:17]
arr_root_rep1 = arr_2024_root[:, 17:17+16]
arr_root_rep2 = arr_2024_root[:, 17+16:17+16+16]

# Plot Reflectance for Root
plt.figure()
for i in range(15):
    plt.plot(waveleth,arr_root_cont[:, i])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.show()

# # Plot Root Rep1
# plt.figure()
# plt.plot(waveleth, arr_root_rep1[:, 0], '-k', label='Rep1 1')
# plt.plot(waveleth, arr_root_rep1[:, 1], '-r', label='Rep1 2')
# plt.plot(waveleth, arr_root_rep1[:, 2], '-b', label='Rep1 3')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')
# plt.legend()
# plt.show()

# # Plot Root Rep2
# plt.figure()
# plt.plot(waveleth, arr_root_rep2[:, 0], '-k', label='Rep2 1')
# plt.plot(waveleth, arr_root_rep2[:, 1], '-r', label='Rep2 2')
# plt.plot(waveleth, arr_root_rep2[:, 2], '-b', label='Rep2 3')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')
# plt.legend()
# plt.show()

# Root/Shoot Reflectance ratio
rr = arr_2024_root[:, 1:]
ss = arr_2024_shoot[:, 1:]
plt.figure()
plt.plot(waveleth, np.nanmean(rr/ss, axis=1), '-k')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (Root/Shoot)')
plt.show()

# Read ground truth data
arr_truth = pd.read_excel('D:/HSI_Root_Rot/Data/Truth3.xlsx', sheet_name='ARR').values
xx_shoot = np.concatenate([arr_shoot_cont.T, arr_shoot_rep1.T, arr_shoot_rep2.T])
xx_root = np.concatenate([arr_root_cont.T, arr_root_rep1.T, arr_root_rep2.T])
yy = arr_truth[:, 6]

# Model training and validation
X = xx_shoot
X_col = X[:, 1]
ind1 = np.isnan(X_col)
y = yy.astype(float)
ind2 = np.isnan(y)
ind = np.unique(np.concatenate([np.where(ind1)[0], np.where(ind2)[0]]))

X = np.delete(X, ind, axis=0)
y = np.delete(y, ind)

# Splitting the data
split_ratio = 0.8
split_idx = np.random.permutation(len(X))
train_idx = split_idx[:int(split_ratio * len(X))]
test_idx = split_idx[int(split_ratio * len(X)):]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

#%% Test different AI ML algorithms

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom scoring function based on R2 (same as your existing code)
def custom_scorer(y_true, y_pred):
    _, _, _, rmse_s, mae, _, R2 = nan_stat_eva_model(y_pred.flatten(), y_true)
    return R2

# Create a custom scorer for GridSearchCV
scorer = make_scorer(custom_scorer, greater_is_better=True)


# Define the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)  # You can adjust max_iter if needed

# Define the grid of hyperparameters to search
param_grid = {
    'C': [0.01, 0.1, 1, 10],              # Regularization strength
    'penalty': ['l1', 'l2', 'elasticnet'],  # Penalty terms for regularization
    'solver': ['liblinear', 'saga']         # Solvers for optimization
}

# Initialize GridSearchCV with Logistic Regression
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, scoring=scorer, cv=5, verbose=1, n_jobs=-1)

# Perform the grid search
grid_search.fit(X_train_scaled, y_train)

# Get the best model and parameters
best_log_reg = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Predict using the best model
y_pred = best_log_reg.predict(X_test_scaled)

#%% Evaluate the performance of the algorithms

# Evaluate model performance
R, bias, sd, rmse_s, mae, d, R2 = nan_stat_eva_model(y_pred.flatten(), y_test)
print(f'R on Test Data: {R[0, 1]:.4f}')
print(f'RMSE: {rmse_s:.4f}, MAE: {mae:.4f}, R2: {R2:.4f}')

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

# # Save the model to a file
# with open(r'L:\HSI_Root_Rot\Method\logistic_regression_good.pkl', 'wb') as f:
#     pickle.dump(best_log_reg, f)
    

#%% Load and use the models

# Later, load the model back from the file
with open(r'L:\HSI_Root_Rot\Method\logistic_regression_good.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    

y_pred = loaded_model.predict(X_test_scaled)

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
