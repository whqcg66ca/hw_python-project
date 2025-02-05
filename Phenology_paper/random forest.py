# random forest
# -- coding: UTF-8 --
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#%% Importing the datasets
x_path = R'C:\Users\HONGQUAN\Working\RF\X_RADAR_CORN.xlsx'
x = pd.read_excel(x_path).iloc[:,0:15].values

y_path = R'C:\Users\HONGQUAN\Working\RF\Y_Crop_corn.xlsx'
y = pd.read_excel(y_path).iloc[:,0].values

# divide the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#training
y_preds = []
score = []
n = 1

while n <= 100:
    forest = RandomForestRegressor(n_estimators = n, random_state = 0)
    forest.fit(x_train, y_train)
    y_preds.append((forest.predict(x_test), n))
    sco = forest.score(x_test, y_test)
    score.append((sco, n))
    n = n + 1

#%% evaluating
mae = []
rmse = []

for y_pred in y_preds:
    print('y_test', y_test)
    print('y_pred', y_pred[0])
    print('---------------------------------------------------')
    mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred[0])
    mae.append((mean_absolute_error, y_pred[1]))
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred[0]))
    rmse.append((root_mean_squared_error, y_pred[1]))
    print('Mean Absolute Error: ', mean_absolute_error)
    print('Root Mean Squared Error:', root_mean_squared_error)
    print('\n')

mae.sort()
rmse.sort()
score.sort()
score.reverse()

print("select estimator based on mean absolute error:", mae[0][1],"is the best estimator , which is", mae[0][0])
print("select estimator based on root mean squared error:", rmse[0][1],"is the best estimator, which is", rmse[0][0])
print("select estimator based on accuracy:", score[0][1],"is the best estimator, which is", score[0][0])
print('\n')