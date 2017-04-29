from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from pandas import read_csv
from pandas.tools.plotting import  scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import utils
from sklearn import preprocessing
import numpy as np
import pandas as pd
import os



os.chdir('.//data//')
filename = 'iris.data.csv'
dataset = read_csv(filename)

array = dataset.values

X = array[:, 0:2]
Y = array[:, 3]

prediction = np.array([5.1,3.5])
validation_size = .20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
Y_train = np.asarray(Y_train, dtype="|S6")

# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
#
#
# results  = []
# names = []
#
# for name, model in models:
#     kfold = KFold(n_splits=10, random_state=seed)
#     cv_result = cross_val_score(model, X_train, Y_train, cv = kfold, scoring='accuracy')
#     results.append(cv_result)
#     names.append(name)
#     print(name, cv_result.mean(), cv_result.std())

lr = LogisticRegression()
lr.fit(X_train, Y_train)

print(lr.predict(prediction))