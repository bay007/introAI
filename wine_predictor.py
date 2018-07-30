import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# random fores model
from sklearn.ensemble import RandomForestRegressor

# import cross validation pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# import evaluate metrics
from sklearn.metrics import mean_squared_error, r2_score

# for save sklearn models
from sklearn.externals import joblib


data = pd.read_csv("winequality-red.csv", header='infer')
print(data.head())
print(data.shape)
print(data.describe())


y = data.quality
x = data.drop('quality', axis='columns')


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=123, stratify=y)


scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
print(x_train_scaled.mean(axis=0))
print(x_train_scaled.std(axis=0))


pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))


hyperparameters = {'randomforestregressor__max_features': [
    'auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1]}


clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(x_train, y_train)
print(clf.best_params_)
clf.refit


y_pred = clf.predict(x_test)


print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

joblib.dump(clf, 'rf_regressor.so')
