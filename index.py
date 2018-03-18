import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.model_selection import train_test_split

#import train and test CSV files
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

#training data description
train.describe(include="all")

				##########################
################## 	DATA ANALYSIS ##################
				##########################

# to get a list of the features within the dataset
train.columns

# to get first five rows of dataset
train.head()

# to get any five rows of dataset
train.sample(5)

#check for null data
pd.isnull(train).sum()

				##########################
################## 	DATA VISAULIZATION ##################
				##########################

############## SEX FEATURE

#bar plot of survival by sex
sns.barplot(x='Sex', y='Survived', data=train)
# plt.show()


#percentage of people by Gender that survived
print "% of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100

print "% of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100

############# PCLASS FEATURE

#bar plot of survival by Pclass
plt.clf()
sns.barplot(x="Pclass", y="Survived", data=train)
plt.xticks([0,1,2], ["upper","middle","lower"])
# plt.show()

#percentage of people by Pclass that survived
print "% of Pclass = upper who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100

print "% of Pclass = middle who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100

print "% of Pclass = lower who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100

############# SIB/SP FEATURE

#bar plot of survival by SibSp
plt.clf()
sns.barplot(x="SibSp", y="Survived", data=train)
# plt.show()

#percentage of people by Pclass that survived
print "% of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100

print "% of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100

print "% of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100

############# PAR/CH FEATURE

#bar plot of survival by Parch
# plt.clf()
# sns.barplot(x="Parch", y="Survived", data=train)
# plt.show()

#percentage of people by Parch that survived
print "% of Parch = 0 who survived:", train["Survived"][train["Parch"] == 0].value_counts(normalize = True)[1]*100

print "% of Parch = 1 who survived:", train["Survived"][train["Parch"] == 1].value_counts(normalize = True)[1]*100

print "% of Parch = 2 who survived:", train["Survived"][train["Parch"] == 2].value_counts(normalize = True)[1]*100

print "% of Parch = 3 who survived:", train["Survived"][train["Parch"] == 3].value_counts(normalize = True)[1]*100

############# AGE FEATURE

#fill null values
train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

plt.clf()
sns.barplot(x="AgeGroup", y="Survived", data=train)
# plt.show()

############# CABIN FEATURE

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

#calculate percentages of CabinBool who survived
print "Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100

print "Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100

#bar plot of CabinBool vs. survival
sns.barplot(x="CabinBool", y="Survived", data=train)
# plt.show()

				##########################
################## 	DATA CLEANING ##################
				##########################

############# CABIN FEATURE

#drop cabin feature
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

############# TICKET FEATURE

#drop the Ticket feature
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

############# EMBARKED FEATURE

print "Number of people embarking in Southampton (S):"
southampton = train[train['Embarked'] == 'S'].shape[0]
print southampton

print "Number of people embarking in Cherbourg (C):"
cherbourg = train[train['Embarked'] == 'C'].shape[0]
print cherbourg

print "Number of people embarking in Queenstown (Q):"
queenstown = train[train['Embarked'] == 'Q'].shape[0]
print queenstown

train['Embarked'] = train['Embarked'].fillna('S')

#map each Embarked value to a numerical value
embarked_mapping = {'S':1, 'C':2, 'Q': 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

############# FARE FEATURE

#fill in missing Fare value in test set based on mean fare for that Pclass

for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)

############# AGE FEATURE

#combine both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()

# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]


#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

#dropping the Age feature for now, might change
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

############# NAME FEATURE

train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)

############# SEX FEATURE

sex_mapping = { "male": 0, "female": 1 }
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

# check train data
# print train.head()

#check test data
# print test.head()

# Splitting the Training Data

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

guassian = GaussianNB()
guassian.fit(x_train, y_train)
y_pred = guassian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print "Gaussian NB acc: " + str(acc_gaussian)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print "Logistic Regression acc: " + str(acc_logreg)

