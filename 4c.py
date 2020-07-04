# Load libraries
import numpy as np
from pandas import read_csv
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
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder

# Load dataset
url = "https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/exam/4/tshirtsNew.csv"
names = ['temperature', 'femaleTshirts', 'maleTshirts', 'competitions', 'location']
dataset = read_csv(url, names=names, header=0)

# shape (cate randuri/coloane)
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution adica de cate ori apare
#print(dataset.groupby('femaleTshirts').size())

# Split-out validation dataset
array = dataset.values
X = array[:, (0, 3, 4)]
y = array[:, 1].reshape(-1, 1)
aux = X[:, 1].ravel()
print("AICI!!!!!!")
print(X)
#print(aux)
label_encoder = LabelEncoder()
Xaux = label_encoder.fit_transform(aux)
#print(Xaux)
aux2 = np.asarray(Xaux)
x_ar = aux2.reshape(-1, 1)
#(x_ar)
Xnou = np.concatenate((X,x_ar),axis=1)
Xnou = Xnou.reshape((-1, 4))
print(Xnou[1:])
print("PANA AICII!!!!")
Xspecial = Xnou
print(Xspecial)
X_train, X_validation, Y_train, Y_validation = train_test_split(Xspecial, y, test_size=0.20, random_state=1)
#print(Y_validation)

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
    kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train.ravel(), cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

info = [[26], ['many'], ['high-school']]
x = np.asarray(info)
#x_array = x.reshape(-1, 1)
print(x)
model = SVC(gamma='auto')
model.fit(X_train, Y_train.ravel())
predictions = model.predict(x_array)

print(predictions)

print(sum(predictions))


