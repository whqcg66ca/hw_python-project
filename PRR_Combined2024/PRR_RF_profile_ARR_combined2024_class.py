import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier 
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, cohen_kappa_score
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
dis='L:'

# %% Step 1.1: Read the Hyperspectral data in Dec 2024
shoot_hsi = dis+ '/HSI_Root_Rot/Data/Specim_ARR_02122024/Spectral_shoot_DecG8.xlsx'
root_hsi = dis+'/HSI_Root_Rot/Data/Specim_ARR_02122024/Spectral_root_DecG8.xlsx'

df_s1 = pd.read_excel(shoot_hsi, sheet_name='ShootR1toR5', header=0).astype(float)
df_s2 = pd.read_excel(shoot_hsi, sheet_name='ShootR6toR10', header=0).astype(float)
df_s3 = pd.read_excel(shoot_hsi, sheet_name='ShootR11toR15', header=0).astype(float)

waveleth = df_s1.iloc[:, 0].values
dec_2024_Shoot = np.hstack([df_s1.iloc[:, 1:].values, df_s2.iloc[:, 1:].values, df_s3.iloc[:, 1:].values])

df_t1 = pd.read_excel(root_hsi, sheet_name='RootR1toR5', header=0).astype(float)
df_t2 = pd.read_excel(root_hsi, sheet_name='RootR6toR10', header=0).astype(float)
df_t3 = pd.read_excel(root_hsi, sheet_name='RootR11toR15', header=0).astype(float)

dec_2024_root = np.hstack([df_t1.iloc[:, 1:].values, df_t2.iloc[:, 1:].values, df_t3.iloc[:, 1:].values])

dec_truth = pd.read_excel(dis+'/HSI_Root_Rot/Data/Truth_December2024_v2_class3_7.xlsx', sheet_name='Feuil1', header=0)
labe_shoot = dec_truth.iloc[:, -4].values.astype(float)
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

###############################################
# Option 2: Remove invaludate values
X = dec_2024_Shoot.T
X = X[:, :-3]
y = labe_root

# Remove NaN values
nan_mask = ~np.isnan(X[:, 1]) & ~np.isnan(y)
X = X[nan_mask]
y = y[nan_mask]
###############################################

#%% Step 1.2 Read the Feb 2024 data

# Define file paths
path_hsi =  dis+r'\HSI_Root_Rot\Data\HSI Spectra RootRot_MAIN.xlsx'
path_truth = dis+ r'\HSI_Root_Rot\Data\Truth3_class3.xlsx'

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

# Plot root data
plt.figure()
for i in range(16):
    plt.plot(waveleth, ARR_Root_Cont.iloc[:, i])
# plt.legend(labe_cont)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.show()

# Read ground truth data
ARR_truth = pd.read_excel(path_truth, sheet_name='ARR', header=0)

XX_Shoot = np.vstack([ARR_Shoot_Cont.to_numpy().T, ARR_Shoot_Rep1.to_numpy().T, ARR_Shoot_Rep2.to_numpy().T])
XX_Root = np.vstack([ARR_Root_Cont.to_numpy().T, ARR_Root_Rep1.to_numpy().T, ARR_Root_Rep2.to_numpy().T])
YY = ARR_truth.iloc[:, -1].to_numpy()


X_Feb = XX_Shoot 
X_Feb = X_Feb[:, :-3]
y_Feb = YY.astype(float)

# Remove NaN values
nan_mask = ~np.isnan(X_Feb[:, 1]) & ~np.isnan(y_Feb)
X_Feb = X_Feb[nan_mask]
y_Feb = y_Feb[nan_mask]
###############################################


#%% Step 2: Prprocessing
# Combine December and February matrices
X_combined = np.vstack([X, X_Feb])

# Combine corresponding labels
y_combined = np.hstack([y, y_Feb])  # Use hstack since y is 1D

print("Combined X shape:", X_combined.shape)
print("Combined y shape:", y_combined.shape)


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
splitIdx = np.random.permutation(len(X_combined ))
trainIdx = splitIdx[:int(splitRatio * len(X_combined ))]
testIdx = splitIdx[int(splitRatio * len(X_combined )):] 
X_train, X_test = X_combined [trainIdx], X_combined [testIdx]
y_train, y_test = y_combined[trainIdx], y_combined[testIdx]
###############################################

# %% Step 3: Classification models 
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
rf_model = RandomForestClassifier (random_state=42)

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
# plt.title('Pea Root Rot')
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.show()

# Evaluate model performance using classification metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

print(f'Accuracy on Test Data: {accuracy:.4f}')
print(f'Cohen Kappa Coefficient: {kappa:.4f}')
print('Confusion Matrix:\n', cm)

# Plot Confusion Matrix
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_test)))
plt.xticks(tick_marks, np.unique(y_test))
plt.yticks(tick_marks, np.unique(y_test))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#%% Step 5: Variable Importance/this could be very interesting for your paper

####### Option 1: unsorted importance
importances_rf = best_rf_model.feature_importances_  # the summation of the importance equal to 1
# Normalize VIP scores between 0 and 1
importances_rf_norm = (importances_rf - np.min(importances_rf)) / (np.max(importances_rf) - np.min(importances_rf))

plt.figure()
plt.scatter(waveleth[:-3], importances_rf_norm, c='k', marker='x')
mx = 1
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