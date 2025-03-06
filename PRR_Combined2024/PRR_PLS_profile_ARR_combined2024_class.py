import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
dis='L:'

# %% Step 1.1: Read the Hyperspectral data in Dec 2024
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

#%% Step 1.2 Read the Feb 2024 data

# Define file paths
path_hsi = dis+r'\HSI_Root_Rot\Data\HSI Spectra RootRot_MAIN.xlsx'
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

# Convert labels to categorical (if needed for classification)
y_Feb = label_encoder.fit_transform(y_Feb)  # Encoding categorical labels

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
np.random.seed(30)
# Split data into training and testing sets
splitRatio = 0.8
splitIdx = np.random.permutation(len(X_combined ))
trainIdx = splitIdx[:int(splitRatio * len(X_combined ))]
testIdx = splitIdx[int(splitRatio * len(X_combined )):] 
X_train, X_test = X_combined [trainIdx], X_combined [testIdx]
y_train, y_test = y_combined[trainIdx], y_combined[testIdx]
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
plt.xticks(np.arange(len(label_encoder.classes_)), ['Low','Moderate', 'High'])
plt.yticks(np.arange(len(label_encoder.classes_)), ['Low','Moderate', 'High'], rotation=90)
plt.show()



# Normalize the confusion matrix to display percentages
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentage

plt.figure()
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.xlabel('Estimated Root Rot')
plt.ylabel('Visual Rating')
plt.colorbar()
classes = ['Low', 'Moderate', 'High']
plt.xticks(np.arange(len(classes)), classes)
plt.yticks(np.arange(len(classes)), classes, rotation=90)

# Add percentage text annotations
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, f'{conf_matrix_normalized[i, j]:.1f}%',
                 ha='center', va='center', color='black' if conf_matrix_normalized[i, j] < 50 else 'white')

plt.show()


#%% Calculate Variable Importance in Projection (VIP)
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