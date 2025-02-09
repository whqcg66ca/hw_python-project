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

# Read the Excel file
FRR_2024_Shoot = pd.read_excel(path_hsi, sheet_name='FRR_2024_Shoot', header=0)

# Extract data based on column indices
waveleth = FRR_2024_Shoot.iloc[:, 0]  # First column
FRR_Shoot_Cont = FRR_2024_Shoot.iloc[:, 1:17]  # Columns 2 to 17 (MATLAB uses 1-based index)
FRR_Shoot_Rep1 = FRR_2024_Shoot.iloc[:, 17:17+16]  # Columns 18 to 33
FRR_Shoot_Rep2 = FRR_2024_Shoot.iloc[:, 17+16:17+16+16]  # Columns 34 to 49

# # Read truth labels
# FRR_truth_txt = pd.read_excel(path_truth, sheet_name='FRR', header=None)
# labe_cont = FRR_truth_txt.iloc[0:16, 1].astype(str).tolist()
# labe_rep1 = FRR_truth_txt.iloc[16:32, 1].astype(str).tolist()
# labe_rep2 = FRR_truth_txt.iloc[32:, 1].astype(str).tolist()

# Plot the first dataset
plt.figure()
for i in range(16):
    plt.plot(waveleth, FRR_Shoot_Cont.iloc[:, i])
# plt.legend(labe_cont)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.show()

# Plot the second dataset
plt.figure()
plt.plot(waveleth, FRR_Shoot_Rep2.iloc[:, 0], '-k', 
         waveleth, FRR_Shoot_Rep2.iloc[:, 1], '-r', 
         waveleth, FRR_Shoot_Rep2.iloc[:, 2], '-b')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.show()

# Prepare data
FRR_truth = pd.read_excel("L:/HSI_Root_Rot/Data/Truth3.xlsx", sheet_name='FRR',header=0).values
XX_Shoot = np.vstack((FRR_Shoot_Cont.T, FRR_Shoot_Rep1.T, FRR_Shoot_Rep2.T))
YY = FRR_truth[:, 6]

#%% Step 2:Preprocessing 
###############################################
# Option 2: Remove invaludate values
X = XX_Shoot
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

# Standardize the data
scaler = xscaler ()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom scoring function based on R2 (same as your existing code)
# Custom scoring function based on R2 (same as your existing code)
def custom_scorer(y_true, y_pred):
    r_squared = r2_score(y_pred.flatten(), y_true)
    return r_squared

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


# # Save the model to a file
# with open(r'L:\HSI_Root_Rot\Method\logistic_regression_good.pkl', 'wb') as f:
#     pickle.dump(best_log_reg, f)
    

