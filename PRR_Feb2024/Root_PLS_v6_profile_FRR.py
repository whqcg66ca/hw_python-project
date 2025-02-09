import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pickle
from sklearn.metrics import r2_score, mean_squared_error

# def nan_stat_eva_model(y_pred, y_test):
#     R = np.corrcoef(y_test, y_pred, rowvar=False)[0, 1]
#     bias = np.mean(y_pred - y_test)
#     sd = np.std(y_pred - y_test)
#     rmse_s = np.sqrt(np.mean((y_pred - y_test) ** 2))
#     mae = np.mean(np.abs(y_pred - y_test))
#     R2 = R ** 2
#     return R, bias, sd, rmse_s, mae, R2

#%% Step 1: Load data
path_hsi = "L:/HSI_Root_Rot/Data/HSI Spectra RootRot_MAIN.xlsx"
FRR_2024_Shoot = pd.read_excel(path_hsi, sheet_name='FRR_2024_Shoot').values

waveleth = FRR_2024_Shoot[:, 0]
FRR_Shoot_Cont = FRR_2024_Shoot[:, 1:16]
FRR_Shoot_Rep1 = FRR_2024_Shoot[:, 17:17+15]
FRR_Shoot_Rep2 = FRR_2024_Shoot[:, 17+16:17+16+15]

# Load ground truth labels
FRR_truth_txt = pd.read_excel("L:/HSI_Root_Rot/Data/Truth3.xlsx", sheet_name='FRR')
labe_cont = FRR_truth_txt.iloc[:16, 1].astype(str).tolist()
labe_rep1 = FRR_truth_txt.iloc[16:32, 1].astype(str).tolist()
labe_rep2 = FRR_truth_txt.iloc[32:, 1].astype(str).tolist()

# Plot reflectance spectra
plt.figure()
for i in range(16):
    plt.plot(waveleth, FRR_Shoot_Cont[:, i])
plt.legend(labe_cont)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.show()

plt.figure()
plt.plot(waveleth, FRR_Shoot_Rep2[:, 0], '-k', label='Sample 1')
plt.plot(waveleth, FRR_Shoot_Rep2[:, 1], '-r', label='Sample 2')
plt.plot(waveleth, FRR_Shoot_Rep2[:, 2], '-b', label='Sample 3')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.show()

# Prepare data
FRR_truth = pd.read_excel("L:/HSI_Root_Rot/Data/Truth3.xlsx", sheet_name='FRR').values
XX_Shoot = np.vstack((FRR_Shoot_Cont.T, FRR_Shoot_Rep1.T, FRR_Shoot_Rep2.T))
YY = FRR_truth[:, 6]

#%% Step 2: Preprocessing
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

#%% Step 3: Train PLS model
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






# Calculate VIP scores
W0 = pls.x_weights_ / np.sqrt(np.sum(pls.x_weights_ ** 2, axis=0))
p = X_train.shape[1]
sumSq = np.sum(pls.x_scores_ ** 2, axis=0) * np.sum(pls.y_weights_ ** 2, axis=0)
vipScore = np.sqrt(p * np.sum(sumSq * (W0 ** 2), axis=1) / np.sum(sumSq))

# Plot VIP scores
plt.figure()
plt.scatter(waveleth, vipScore, marker='x', c='k')
mx = 4.5
plt.axvline(400, color='b', linestyle='-')
plt.axvline(500, color='g', linestyle='-')
plt.axvline(600, color='r', linestyle='-')
plt.axvline(680, color='k', linestyle='-')
plt.axvline(750, color='m', linestyle='-')
plt.axvline(970, color='y', linestyle='-')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Importance of Wavelength')
plt.ylim([0, mx])
plt.xlim([300, 1100])
plt.show()