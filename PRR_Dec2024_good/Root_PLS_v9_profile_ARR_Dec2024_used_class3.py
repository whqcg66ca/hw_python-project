import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
dis='L:'

# %% Step 1: Read the Hyperspectral Shoot data in Excel
shoot_hsi = dis+'/HSI_Root_Rot/Data/Specim_ARR_02122024/Spectral_shoot_DecG8.xlsx'
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

dec_truth = pd.read_excel(dis+'/HSI_Root_Rot/Data/Truth_December2024_v2_class3.xlsx', sheet_name='Feuil1', header=0)
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

#%% Step 2: Prprocessing
# ###############################################
# Option 1: Remove invaludate values
# X = dec_2024_root.T
# X = X[:, :-3]  # Remove last three columns
# X_col = X[:, 1]
# ind1 = np.where(np.isnan(X_col))[0]
# y = labe_root
# ind2 = np.where(np.isnan(y))[0]
# ind = np.sort(np.concatenate([ind1, ind2]))

# X = np.delete(X, ind, axis=0)
# y = np.delete(y, ind)
###############################################

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

# Convert labels to categorical (if needed for classification)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encoding categorical labels

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
splitIdx = np.random.permutation(len(X))
trainIdx = splitIdx[:int(splitRatio * len(X))]
testIdx = splitIdx[int(splitRatio * len(X)):] 
X_train, X_test = X[trainIdx], X[testIdx]
y_train, y_test = y[trainIdx], y[testIdx]
###############################################

# %% Step 3: Regression models 
# Test Number of latent variables
accuracy = []
for Ncom in range(1, 41):
    pls = PLSRegression(n_components=Ncom)
    pls.fit(X_train, y_train)
    
    # Apply LDA on the reduced components (PLS scores)
    X_train_pls = pls.transform(X_train)  # Reduced components
    X_test_pls = pls.transform(X_test)
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_pls, y_train)
    
    # Predict on the test set
    y_pred = lda.predict(X_test_pls)
    accuracy.append(accuracy_score(y_test, y_pred))
    print(f'Accuracy on Test Data for {Ncom} components: {accuracy[-1]}')

# Plot accuracy vs. number of components
plt.figure()
plt.plot(range(1, 41), accuracy, 'ok')
plt.xlabel('Number of components in PLS-LDA')
plt.ylabel('Accuracy')
plt.title('Selection of Components')
plt.show()

# Select optimal number of components
num_com = np.argmax(accuracy) + 1

# Train final model with optimal number of components
pls = PLSRegression(n_components=num_com)
pls.fit(X_train, y_train)

X_train_pls = pls.transform(X_train)
X_test_pls = pls.transform(X_test)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pls, y_train)

y_pred = lda.predict(X_test_pls)


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

# Evaluate the model
accuracy_final = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Final Accuracy on Test Data: {accuracy_final}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot Confusion Matrix
plt.figure()
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.xlabel('Estimated Root Rot')
plt.ylabel('Visual Rating')
plt.colorbar()
plt.xticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_)
plt.yticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_)
plt.show()



# Calculate Variable Importance in Projection (VIP)
W0 = pls.x_weights_ / np.sqrt(np.sum(pls.x_weights_ ** 2, axis=0))
p = X.shape[1]
sumSq = np.sum(pls.x_scores_ ** 2, axis=0) * np.sum(pls.y_loadings_ ** 2, axis=0)
vipScore = np.sqrt(p * np.sum(sumSq * (W0 ** 2), axis=1) / np.sum(sumSq))
# Normalize VIP scores between 0 and 1
vipScore_norm = (vipScore - np.min(vipScore)) / (np.max(vipScore) - np.min(vipScore))

plt.figure()
plt.scatter(waveleth[:-3], vipScore_norm, c='k', marker='x')
mx = 4.5
plt.axvline(x=400, color='b')
plt.axvline(x=500, color='g')
plt.axvline(x=600, color='r')
plt.axvline(x=680, color='k')
plt.axvline(x=750, color='m')
plt.axvline(x=970, color='y')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Importance of wavelength')
plt.ylim([0, 1])
plt.xlim([300, 1100])
plt.show()