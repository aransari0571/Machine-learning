import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#load the dataset
dataset = pd.read_csv(r"C:\Users\AR ANSARI\NIT\ML\spyder\Investment(MLR).csv")

#futuer seectins 
x = dataset.iloc[:,:-1] # year of experience(Independent veriable)
y = dataset.iloc[:,-1] # salary (Dependent veriable) 

x = pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x,y, test_size=0.20, random_state=0) 


from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(x_train, y_train) 

y_pred = regressor.predict(x_test)


# we build MLR(Mutiple liner Regression) model
m= regressor.coef_
print(regressor.coef_)  

c = regressor.intercept_
print(regressor.intercept_)

x = np.append(arr = np.full((50,1),42467).astype(int), values = x ,axis =1)

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]] #const are x0, x1,x2, x3, x4,x5
#ordinary Least Squres
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,5]]  #const are x0, x1,x2, x3,x5 
  #4 is remove p>|t| is max(highest) value and p value is 0.05
#ordinary Least Squres
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]  #const are x0,x1,x2, x3 
  #5 is remove p>|t| is max(highest) value and p value is 0.05
#ordinary Least Squres
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
 

import statsmodels.api as sm
x_opt = x[:,[0,1,3]]  #const are x0,x1, x3 
  #2 is remove p>|t| is max(highest) value and p value is 0.05
#ordinary Least Squres
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
 

import statsmodels.api as sm
x_opt = x[:,[0,1]]  #const are x0,x1,  
  #3 is remove p>|t| is max(highest) value and p value is 0.05
#ordinary Least Squres
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


bias = regressor.score(x_train,y_train)
print(bias)

# bias and variance jitne nerates ho ge model is best h ga 
#(i.e 0.9501847627493607 is bias and 0.93847627497683607 is varianec )
variance = regressor.score(x_test,y_test)
print(variance)

