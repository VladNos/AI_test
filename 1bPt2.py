# Load libraries
import numpy as np
from pandas import read_csv
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Load dataset
url = "https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/exam/1/homeData.csv"
names = ['id','date','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']
dataset = read_csv(url, names=names, header=0)
df = pd.DataFrame(dataset)
df = df.iloc[0:150, :]
dfPrice = df[df.columns[2]]
cols = [2, 3, 4, 5, 6, 7]
colsA = [1, 2, 3, 4, 5]
df = df[df.columns[cols]]
dfA = df[df.columns[colsA]]
#normalizare daca nu exista elems negative
print(df.max())
column_maxes = df.max()
df_max = column_maxes.max()
normalized_df = df / df_max
print(normalized_df.max())

reg =linear_model.LinearRegression()
reg.fit(df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']], df[['price']])
print(reg.coef_)
print(reg.predict([[3, 2.5, 1910, 66210,  2]]))