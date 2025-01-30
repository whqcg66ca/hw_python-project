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
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
import pickle

sys.path.append(r'L:\HSI_Root_Rot\Method\funs')
from calculate_metrics import nan_stat_eva_model  


# Step 1: Read the Hyperspectral Shoot data in Excel
shoot_hsi = 'L:/HSI_Root_Rot/Data/Specim_ARR_02122024/Spectral_shoot_DecG8.xlsx'
root_hsi = 'L:/HSI_Root_Rot/Data/Specim_ARR_02122024/Spectral_root_DecG8.xlsx'

df_s1 = pd.read_excel(shoot_hsi, sheet_name='ShootR1toR5', header=0).astype(float)
df_s2 = pd.read_excel(shoot_hsi, sheet_name='ShootR6toR10', header=0).astype(float)
df_s3 = pd.read_excel(shoot_hsi, sheet_name='ShootR11toR15', header=0).astype(float)

waveleth = df_s1.iloc[:, 0].values
dec_2024_Shoot = np.hstack([df_s1.iloc[:, 1:].values, df_s2.iloc[:, 1:].values, df_s3.iloc[:, 1:].values])

df_t1 = pd.read_excel(root_hsi, sheet_name='RootR1toR5', header=0).astype(float)
df_t2 = pd.read_excel(root_hsi, sheet_name='RootR6toR10', header=0).astype(float)
df_t3 = pd.read_excel(root_hsi, sheet_name='RootR11toR15', header=0).astype(float)

dec_2024_root = np.hstack([df_t1.iloc[:, 1:].values, df_t2.iloc[:, 1:].values, df_t3.iloc[:, 1:].values])

dec_truth = pd.read_excel('L:/HSI_Root_Rot/Data/Truth_December2024.xlsx', sheet_name='Feuil1', header=0)
labe_shoot = dec_truth.iloc[:, -2].values.astype(float)
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

# Step 4: Partial Least Squares (PLS) Regression Model training and validation
X = dec_2024_Shoot.T
X = X[:, :-3]  # Remove last three columns
X_col = X[:, 1]
ind1 = np.where(np.isnan(X_col))[0]
y = labe_root
ind2 = np.where(np.isnan(y))[0]
ind = np.sort(np.concatenate([ind1, ind2]))

X = np.delete(X, ind, axis=0)
y = np.delete(y, ind)

# Split data into training and testing sets
split_ratio = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=50)



#%% Test different AI ML algorithms

# Define the parameter grid for GridSearchCV for XGBoost
param_grid = {
    'n_estimators': range(50, 200, 50),  # Number of trees
    'max_depth': [6, 10, 15],           # Maximum depth of each tree
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
    'subsample': [0.7, 0.8, 1.0]        # Fraction of samples used per tree
}

# Initialize the XGBoost Regressor model
xgb_model = XGBRegressor(random_state=42)

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and model from grid search
best_params = grid_search.best_params_
best_xgb_model = grid_search.best_estimator_

# Predict using the best model
y_pred = best_xgb_model.predict(X_test)


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

# Save the model to a file
with open(r'L:\HSI_Root_Rot\Method\XGBoost_model2.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
    

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

