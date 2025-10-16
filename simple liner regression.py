import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#load the dataset
dataset = pd.read_csv(r"C:\Users\AR ANSARI\NIT\ML\spyder\Salary_Data.csv")

#futuer seectins 
x = dataset.iloc[:,:-1] # year of experience(Independent veriable)
y = dataset.iloc[:,-1] # salary (Dependent veriable) 

#spit the dataset into training and testting setes(80% training and 20% testing)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x,y, test_size=0.20, random_state=0) 

#you do not  need  to resshap y_train as its target variable 
#fit the linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(x_train, y_train) 

y_pred = regressor.predict(x_test)
m.
#Compare predicted and actual salaries from the test set
comparison = pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
print(comparison)

#visulization the Test  set result
plt.scatter(x_test, y_test, color = 'red') #real salary data
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Year of experience')
plt.ylabel('salary')
plt.show()

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)
 
y_12 = m*12+c
print(y_12) 

y_10 = m*10+c
print(y_10) 


# Best fit line here
coef = print(f"Coefficent : {regressor.coef_}")
intercept = print(f"Intercept : {regressor.intercept_}")

#ftuer predicttion code
exp_12_future_pred = 9312 * 100 +26780
exp_12_future_pred 

bias = regressor.score(x_train,y_train)
print(bias)

variance = regressor.score(x_test,y_test)
print(variance)

dataset.mean()  # This will give mean of entire dataframe
dataset['Salary'].mean() #This will give us the mean of that particular column
dataset.median() # This will give median of entire dataframe
dataset['Salary'].median() # This will give us median of that particular column
dataset['Salary'].mode() # This will give us mode of that particular column
dataset.var() # This will give variance of entire dataframe

#Standard Deviation
dataset.std() # This will give you the standard deviation of the entire dataframe

# Coefficent of Variation
# for calculating cv we have to import a library first
from scipy.stats import variation
variation(dataset.values) # This will give cv of entire dataframe

# Correlation
dataset.corr() # This will give correlation of entire dataframe
# This will give us the correlation between these two attributes
dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness
dataset.skew() # This will give the skewness of entire dataframe
dataset['Salary'].skew() # This will give us skewness of that particular column

#Standard Error
dataset.sem() # This well give standard error os entire dataframe
dataset['Salary'].sem() # This will give us standard error of that particular column

# Z Score 
# For calculating Z- Score we have to import a library first
import scipy.stats as stats
dataset.apply(stats.zscore) # This will give Z-Score of Entire Dataframe
stats.zscore(dataset['Salary']) # This will give us the Z Score of that particular column

#Skewness
dataset.skew() # This will give the skewness of entire dataframe
dataset['Salary'].skew() # This will give us skewness of that particular column
#Standard Error
dataset.sem() # This well give standard error os entire dataframe
dataset['Salary'].sem() # This will give us standard error of that particular column


# Sum of Sqaure Regressor (SSR)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)


#SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)


#SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)


print(SSR)
print(SSE)
print(SST)

#R2 Square
r_square = 1 - (SSR/SST)
r_square

from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)
print(train_mse)
print(test_mse)

#pickle is the frontend . use to frontend devl.
import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:

    pickle.dump(regressor, file)

print("Model has been pickled and saved as linear_regreeion_model")   

#save file 
import os 
print(os.getcwd())



















