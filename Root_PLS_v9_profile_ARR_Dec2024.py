import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Read the Hyperspectral Shoot data in Excel
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

# Test number of components in PLS
rmse = []
num_components = range(1, 41)
for Ncom in num_components:
    pls = PLSRegression(n_components=Ncom)
    pls.fit(X_train, y_train)
    y_pred = pls.predict(X_test)
    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f'Mean Squared Error on Test Data (N={Ncom}): {rmse[-1]}')

# Plot RMSE vs. Number of Components
plt.figure()
plt.plot(num_components, rmse, 'ok')
plt.xlabel('Number of Components in PLSR')
plt.ylabel('RMSE')
plt.title('Selection of Components')
plt.show()

# Find optimal number of components
optimal_ncom = num_components[np.argmin(rmse)]

# Train final PLS model
pls_final = PLSRegression(n_components=optimal_ncom)
pls_final.fit(X_train, y_train)
y_pred_final = pls_final.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
print(f'R^2 on Test Data: {r2}')
print(f'RMSE on Test Data: {rmse_final}')

# Scatter plot of actual vs predicted
plt.figure()
plt.scatter(y_test, y_pred_final, marker='o', color='k')
plt.xlabel('Visual Rating')
plt.ylabel('Estimated Root Rot')
plt.title('Pea Root Rot')
plt.xlim([0, 7])
plt.ylim([0, 7])
plt.text(2.7, 4, f'R²={r2:.2f}')
plt.text(2.7, 3, f'RMSE={rmse_final:.2f}')
plt.show()

# Calculate VIP scores
W0 = pls_final.x_weights_ / np.sqrt(np.sum(pls_final.x_weights_**2, axis=0))
p = X_train.shape[1]
sum_sq = np.sum(pls_final.x_scores_**2, axis=0) * np.sum(pls_final.y_loadings_**2, axis=0)
vip_score = np.sqrt(p * np.sum(sum_sq * (W0**2), axis=1) / np.sum(sum_sq))

# Plot VIP scores
plt.figure()
plt.scatter(waveleth[:-3], vip_score, marker='x', color='k')
plt.axvline(x=400, color='b', linestyle='-')
plt.axvline(x=500, color='g', linestyle='-')
plt.axvline(x=600, color='r', linestyle='-')
plt.axvline(x=680, color='k', linestyle='-')
plt.axvline(x=750, color='m', linestyle='-')
plt.axvline(x=970, color='y', linestyle='-')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Importance of Wavelength')
plt.ylim([0, 4.5])
plt.xlim([300, 1100])
plt.show()
