import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import pickle
from sklearn.metrics import r2_score, mean_squared_error

# Function to evaluate model performance (similar to nan_stat_eva_model)
# def nan_stat_eva_model(y_pred, y_true):
#     R = np.corrcoef(y_pred.flatten(), y_true)
#     bias = np.mean(y_pred - y_true)
#     sd = np.std(y_pred - y_true)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = np.mean(np.abs(y_pred - y_true))
#     d = np.sum((y_pred - y_true) ** 2)
#     R2 = r2_score(y_true, y_pred)
#     return R, bias, sd, rmse, mae, d, R2

#%% Step 1: Read the hyperspectral data
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


#%% Step 2: Preprocessing 

X = XX_Root 
X = X[:, :-3]
y = YY .astype(float)

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

#%% Step 3:  PLS regression model with 30 components
# Test different numbers of components
rmse = []
for Ncom in range(1, 31):
    pls = PLSRegression(n_components=Ncom)
    pls.fit(X_train, y_train)
    y_pred = pls.predict(X_test).flatten()
    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f'Mean Squared Error on Test Data for {Ncom} components: {rmse[-1]}')

# Plot RMSE vs number of components
plt.figure()
plt.plot(range(1, 31), rmse, 'ok')
plt.xlabel('Number of Components in PLSR')
plt.ylabel('RMSE')
plt.title('Selection of Components')
plt.show()

# Find optimal number of components
N_com = np.argmin(rmse) + 1

# Train final PLS model
pls = PLSRegression(n_components=N_com)
pls.fit(X_train, y_train)
y_pred = pls.predict(X_test).flatten()

# %%Step 4: Evaluate model performance
# Evaluate model
r2 = r2_score(y_test, y_pred)
rmse_s = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'R^2 on Test Data: {r2}')

# Plot actual vs. predicted values
plt.figure()
plt.scatter(y_test, y_pred, c='k', marker='o')
plt.text(6, 1.5, rf"$R^2 = {r2:.2f}$")
plt.text(6, 1, f'RMSE={rmse_s:.2f}')
plt.xlabel('Visual Rating')
plt.ylabel('Estimated Root Rot')
# plt.title('Pea Root Rot')
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.show()


#%% Step 5 Calculate VIP scores
# Calculate Variable Importance in Projection (VIP)
W0 = pls.x_weights_ / np.sqrt(np.sum(pls.x_weights_ ** 2, axis=0))
p = X.shape[1]
sumSq = np.sum(pls.x_scores_ ** 2, axis=0) * np.sum(pls.y_loadings_ ** 2, axis=0)
vipScore = np.sqrt(p * np.sum(sumSq * (W0 ** 2), axis=1) / np.sum(sumSq))

plt.figure()
plt.scatter(waveleth[:-3], vipScore, c='k', marker='x')
mx = 4.5
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



# Save the model to a file
# with open('L:\HSI_Root_Rot\Method\pls_model2.pkl', 'wb') as f:
#     pickle.dump(pls, f)
    
