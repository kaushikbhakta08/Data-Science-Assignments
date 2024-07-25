# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:55:44 2023

@author: kaush
"""
#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
#import dataset
df=pd.read_csv("Salary_Data.csv")
df
df.info()

#Correlation Analysis
df.corr()

#Splitting Variables
X=df[['YearsExperience']]
Y=df['Salary']
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Bo=LR.intercept_  #Bo
B1=LR.coef_
# Predictions
Y_pred=LR.predict(X)

#EDA and Data Visualization
plt.scatter(x=df['YearsExperience'],y=df['Salary'],color='blue')
plt.scatter(x=df['YearsExperience'],y=Y_pred,color='red')
plt.plot(df['YearsExperience'],Y_pred,color='black')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

sns.regplot(x=df['YearsExperience'],y=df['Salary'])

Y-Y_pred

model=smf.ols("Salary~YearsExperience",data=df).fit()
model.summary()

#Model Testing
#Finding Coeficient Parameter
model.params

#Finding TValue and PValue
model.tvalues
model.pvalues

#Finding RSquared Values
model.rsquared
model.rsquared_adj

#Model Predictions manually for 3 years Experience
Salary = (25792.200199) + (9449.962321)*(3)             # Coeficient Paramenter Value
Salary

#Automatic Prediction for say Years of Experience 5, 8
new_df=pd.Series([5,8])
new_df

df_pred=pd.DataFrame(new_df,columns=['YearsExperience'])
df_pred

model.predict(df_pred)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("Mean Squared Error: ",mse.round(2))
print("Root Mean Squared Error: ",np.sqrt(mse).round(2))