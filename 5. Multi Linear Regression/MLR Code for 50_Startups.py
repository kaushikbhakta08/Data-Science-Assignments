# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:32:12 2023

@author: kaush
"""
#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#import dataset
dataset=pd.read_csv('50_Startups.csv')

#Statistical analysis of the dataset
dataset.describe()

#Dimensions of dataset
print('There are ',dataset.shape[0],'rows and ',dataset.shape[1],'columns in the dataset.')
print('There are',dataset.duplicated().sum(),'duplicate values in the dateset.') #using duplicated()

#Schema of dataset
dataset.info()

#Correlation Analysis
c=dataset.corr()

#Correlation matrix
sns.heatmap(c,annot=True,cmap='Blues')
plt.show()

#Outliers detection in the target variable
outliers = ['Profit']
plt.rcParams['figure.figsize'] = [8,8]
sns.boxplot(data=dataset[outliers], orient="h", palette="Set2", width=0.7) #Orient = "v" : vertical boxplot , 
                                                                            #Orient = "h" : hotrizontal boxplot
plt.title("Outliers Variable Distribution")
plt.ylabel("Profit Range")
plt.xlabel("Continuous Variable")
plt.show()

#State-wise outliers detection
sns.boxplot(x = 'State', y = 'Profit', data = dataset)
plt.show()

#Histogram on Profit
sns.distplot(dataset['Profit'],bins=5,kde=True)
plt.show()

#Pair plot
sns.pairplot(dataset)
plt.show()

#Model Development
# spliting Dataset in Dependent & Independent Variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Label Encoder: Encode labels with values between 0 and n_classes-1.
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X1 = pd.DataFrame(X)
X1.head()

#Now we have to split the data into training and testing data
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)
x_train

#Model Fitting
model = LinearRegression()
model.fit(x_train,y_train)

#Testing the model using the predict function
y_pred = model.predict(x_test)
y_pred

#Testing scores
testing_data_model_score = model.score(x_test, y_test)
print("Model Score/Performance on Testing data",testing_data_model_score)
training_data_model_score = model.score(x_train, y_train)
print("Model Score/Performance on Training data",training_data_model_score)

#Comparing the predicted values and actual values
df = pd.DataFrame(data={'Predicted value':y_pred.flatten(),'Actual Value':y_test.flatten()})
df

#Model evaluation
from sklearn.metrics import r2_score
r2Score = r2_score(y_pred, y_test)
print("R2 score of model is :" ,r2Score*100)

#MSE – Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred, y_test)
print("Mean Squarred Error is :" ,mse*100)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Root Mean Squarred Error is : ",rmse*100)

#MAE – Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_pred,y_test)
print("Mean Absolute Error is :" ,mae)