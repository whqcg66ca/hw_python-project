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

#%% Step 2: Preprocessing 
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
plt.text(4, 2.5, rf"$R^2 = {r2:.2f}$")
plt.text(4, 2, f'RMSE={rmse_s:.2f}')
plt.xlabel('Visual Rating')
plt.ylabel('Estimated Root Rot')
plt.title('Pea Root Rot')
plt.xlim([0, 7])
plt.ylim([0, 7])
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

plt.close('all')

# Save the model to a file
# with open('L:\HSI_Root_Rot\Method\pls_model2.pkl', 'wb') as f:
#     pickle.dump(pls, f)
    
