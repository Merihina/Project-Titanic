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
test=pd.read_csv('train.csv', sep=',', index_col=0)
Data = pd.read_csv('train.csv', sep=',', index_col=0)
train.head()
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
train.head()
#Embarked to numérical values
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)

train.head()
combine = [train,test]
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
#Map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
#Drop Fare values
train = train.drop(['Fare'], axis = 1)
#Drop Cabin values
train = train.drop(['Cabin'], axis = 1)
#Drop Ticket values
train = train.drop(['Ticket'], axis = 1)

#Age feature has ~ 20% missing Data
missing_ages = train[train['Age'].isnull()]
mean_ages = train.groupby(['Sex','Pclass'])['Age'].mean()

def remove_na_ages(row):
    if pd.isnull(row['Age']):
        return mean_ages[row['Sex'],row['Pclass']]
    else:
        return row['Age']

train['Age'] =train.apply(remove_na_ages, axis=1)

#==============================================================================

predictors = train.drop(['Survived'], axis=1).dropna()
target = train["Survived"].drop([889,891])
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
print(models.sort_values(by='Score', ascending=False))
