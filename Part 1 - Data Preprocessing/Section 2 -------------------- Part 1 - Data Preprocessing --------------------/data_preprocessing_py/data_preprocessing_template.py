# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#處理missing data
imputer = SimpleImputer( strategy='mean')
imputer: SimpleImputer = imputer.fit(X[:, 1:3]);
X[:, 1:3] = imputer.transform(X[:, 1:3])

#data 分類
X[:,0] = LabelEncoder().fit_transform(X[:,0])

# 將 countries 轉成 dummy variable
ct = ColumnTransformer([("Name_Of_Your_Step", OneHotEncoder(),[0])], remainder="passthrough")
X = ct.fit_transform(X)

y = LabelEncoder().fit_transform(y)

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
