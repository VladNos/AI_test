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
cols = [2, 3, 4, 5, 6, 7]
df = df[df.columns[cols]]
#normalizare daca nu exista elems negative
print(df.max())
column_maxes = df.max()
df_max = column_maxes.max()
normalized_df = df / df_max
print(normalized_df.max())

#normalizare pt elems negative
#column_maxes = df.max()
#df_max = column_maxes.max()
#column_mins = df.min()
#df_min = column_mins.min()
#normalized_df = (df - df_min) / (df_max - df_min)

# shape (cate randuri/coloane)
print(df.shape)

# head
print(df.head(20))

# descriptions
#print(dataset.describe())

# class distribution adica de cate ori apare
#print(dataset.groupby('x').size())


# Split-out validation dataset
array = df.values
X = array[:150, (1, 2, 3, 4,5)]
y = array[:150, 0].reshape(-1, 1)
y = y.astype('int32')
X = X.astype('float')
print(X)

#print(X)

X_train, X_validation, Y_train, Y_validation = train_test_split(normalized_df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']], df[['price']].values.ravel(), test_size=0.33, random_state=1)
print(Y_train)
print(X_train)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=2, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

info = [26, 26, 25, 25, 27, 27, 27, 27, 24, 25, 23, 27, 27, 22, 24, 27, 24, 23, 25, 27, 26, 22, 24, 24, 24, 25, 24, 25, 23, 25]
#x = np.asarray(info)
#x_array = x.reshape(-1, 1)
#print(x)
#model = SVC(gamma='auto')
#model.fit(X_train, Y_train.ravel())
#predictions = model.predict(x_array)

#print(predictions)

#print(sum(predictions))


reg =linear_model.LinearRegression()
reg.fit(df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']], df[['price']])
print(reg.coef_)
print(reg.predict([[3, 2.5,1910, 66210,  2]]))