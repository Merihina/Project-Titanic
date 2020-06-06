# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:30:37 2020

@author: Meriem
"""
#%%

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
train=pd.read_csv('train.csv', sep=',', index_col=0)
Data = pd.read_csv('train.csv', sep=',', index_col=0)
test=pd.read_csv('train.csv', sep=',', index_col=0)
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 14, 25, 35, 60, np.inf]
labels = ['Unknown', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
age_mapping = {'Unknown': None,'Child': 1, 'Teenager': 2, 'Young Adult': 3, 'Adult': 4, 'Senior': 5}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head()
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)
train.head()
#Embarked to numérical values
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#==============================================================================
#Replace various titles with more common names
for dataset in combine:
   
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms','Countess','Miss','Mme','Lady'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Major','Sir','Capt','Col','Don','Jonkheer','Rev','Master'], 'Mr')
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#Map each of the title groups to a numerical value
title_mapping = {"Dr": 1, "Mr": 2, "Mrs": 3}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
#Drop the name feature since it contains no more useful information
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)

#Map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
#Drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
#Drop Cabin values
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
#Drop Ticket values
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
#=============================================================================================================================
#Make train data from train with age information
print("------------------------------------------------------")
print(train.head().to_string())
print("------------------------------------------------------")
train_age = train
dropednan = train_age.dropna()
#Make test data from train without age information
null_columns=train.columns[train.isnull().any()]
x_train_age = dropednan.drop(['AgeGroup'], axis = 1)
y_train_age = dropednan["AgeGroup"]
x_test_AgeGroup = train[train.isnull().any(axis=1)]
x_test_age = x_test_AgeGroup.drop(['AgeGroup'], axis = 1)
from sklearn.model_selection import train_test_split
predictors = x_train_age
target = y_train_age
x_trainage, x_valage, y_trainage, y_valage = train_test_split(predictors, target, test_size = 0.1, random_state = 0)
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_trainage, y_trainage)
y_predage = gbk.predict(x_valage)
acc_gbkage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_gbkage)
#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_trainage, y_trainage)
y_predage = sgd.predict(x_valage)
acc_sgdage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_sgdage)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_trainage, y_trainage)
y_predage = knn.predict(x_valage)
acc_knnage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_knnage)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_trainage, y_trainage)
y_predage = randomforest.predict(x_valage)
acc_randomforestage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_randomforestage)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_trainage, y_trainage)
y_predage = decisiontree.predict(x_valage)
acc_decisiontreeage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_decisiontreeage)
#Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_trainage, y_trainage)
y_predage = perceptron.predict(x_valage)
acc_perceptronage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_perceptronage)
#Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_trainage, y_trainage)
y_predage = linear_svc.predict(x_valage)
acc_linear_svcage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_linear_svcage)
#Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_trainage, y_trainage)
y_predage = svc.predict(x_valage)
acc_svcage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_svcage)
#Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_trainage, y_trainage)
y_predage = logreg.predict(x_valage)
acc_logregage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_logregage)
#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_trainage, y_trainage)
y_predage = gaussian.predict(x_valage)
acc_gaussianage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_gaussianage)
models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbkage, acc_sgdage, acc_knnage, acc_randomforestage, 
              acc_decisiontreeage, acc_perceptronage, acc_linear_svcage, acc_svcage, acc_logregage, 
               acc_gaussianage]})
print(models.sort_values(by='Score', ascending=False))
predictions = randomforest.predict(x_test_age.dropna())
k=0
for i in range(890):
    i+=1
    if  np.isnan(train_age['AgeGroup'][i]) == True:
        train_age['AgeGroup'][i] = predictions[k]
        k+=1
Predicted = train_age
#====================================================================================================================================================================
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived'], axis=1).dropna()
target = train["Survived"].drop([889, 891])


x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.05, random_state = 0)

#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)

#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
#KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)

#Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)
#Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

#Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
#Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)

models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbk, acc_sgd, acc_knn, acc_randomforest, acc_decisiontree, acc_perceptron, acc_linear_svc, acc_svc, acc_logreg, 
               acc_gaussian]})
models.sort_values(by='Score', ascending=False)

#===============================================================================================================================
predictors = Predicted.drop(['Survived'], axis=1).dropna()
target = Predicted["Survived"].drop([889, 891])
x_train, x_vale, y_train, y_vale = train_test_split(predictors, target, test_size = 0.05, random_state = 0)
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)

#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
#KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)

#Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)
#Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

#Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
#Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbk, acc_sgd, acc_knn, acc_randomforest, acc_decisiontree, acc_perceptron, acc_linear_svc, acc_svc, acc_logreg, 
               acc_gaussian]})
print(models.sort_values(by='Score', ascending=False))
#==============================================================================
#Age feature has ~ 20% missing Data 
missing_ages = Data[Data['Age'].isnull()]
mean_ages = Data.groupby(['Sex','Pclass'])['Age'].mean()

def remove_na_ages(row):
    if pd.isnull(row['Age']):
        return mean_ages[row['Sex'],row['Pclass']]
    else:
        return row['Age']

Data['Age'] =Data.apply(remove_na_ages, axis=1)
print(Data.Age.head(20).to_string())


