# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:16:55 2023

@author: kaush
"""
#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
#import dataset
df=pd.read_csv("delivery_time.csv")
df
df.info()
#Renaming Columns
df1=df.rename({'Delivery Time':'del_tm','Sorting Time':'sort_tm'},axis=1)
df1

#Correlation Analysis
df1.corr()

#Splitting Variables
X=df1[['sort_tm']]
Y=df1['del_tm']
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Bo=LR.intercept_  #Bo
B1=LR.coef_
# Predictions
Y_pred=LR.predict(X)

#EDA and Data Visualization
plt.scatter(x=df1['sort_tm'],y=df1['del_tm'],color='blue')
plt.scatter(x=df1['sort_tm'],y=Y_pred,color='red')
plt.plot(df1['sort_tm'],Y_pred,color='black')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

sns.regplot(x=df1['sort_tm'],y=df1['del_tm'])

Y-Y_pred

model=smf.ols("del_tm~sort_tm",data=df1).fit()
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

#Model Predictions manually for sorting time 5
del_tm = (6.582734) + (1.649020)*(5)             # Coeficient Paramenter Value
del_tm

#Automatic Prediction for say sorting time 5, 8
new_df=pd.Series([5,8])
new_df

df_pred=pd.DataFrame(new_df,columns=['sort_tm'])
df_pred

model.predict(df_pred)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("Mean Squared Error: ",mse.round(2))
print("Root Mean Squared Error: ",np.sqrt(mse).round(2))