import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

## load dataset
dataset = pd.read_csv(r"C:\Users\AR ANSARI\vscode\ML\spyder\emp_sal.csv")

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Linear Regreassion
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

 # Linear Regression visualization
 #plt.scatter(X, Y, color='red')
 #plt.plot(X, lin_reg.predict(X), color='blue')
 #plt.title('Linear Regression graph')
 #plt.xlabel('Position level')
 #plt.ylabel('Salary')
 #plt.show()
 
 # Polynomial regression algorithm
 poly_reg = PolynomialFeatures(degree = 4)  # You can try degree= 2 or 3 also
 X_poly = poly_reg.fit_transform(X)
 
 # again liner model build with 2nd degree 
 lin_reg_2 = LinearRegression()
 lin_reg_2.fit(X_poly, Y)
 
 # Plot Polynomial Regressio (poly model)
 plt.scatter(X, Y, color='red')
 plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
 plt.title('Truth or Bluff (Polynomial Regression)')
 plt.xlabel('Position level')   # Fixed: was plt.Xlabel
 plt.ylabel('Salary')           # Fixed: was plt.Ylabel
 plt.show()
 
 
 
 # ([[6.5]]) yhe new joining ka exprence hai tu is ko salary kise dena hai   
 #  q ki 6 year wala emp remove the job and his salary 200000 hai
 # or 7 year wale i salary 300000 hai tu new(6.5) new joining ko 
 # between 20000 se 300000 me dena hai tu is ke liye yhe prediction kr rahe h
 # with polynomial ,svr,knn,decission tree and randon forest algorithm 
 
 # Polynmial Model algrithm
lin_model_pred = lin_reg.predict([[6.5]])
print("Linear Model Prediction:", lin_model_pred)

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print("Polynomial Model Prediction:", poly_model_pred)
 
# ##### SVR Model Algorithm
from sklearn.svm import SVR
svr_model=SVR()
svr_model.fit(X,Y)

svr_model_pred=svr_model.predict([[6.5]])
print(svr_model_pred)

#### KNN Model Algorithm
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=4,weights='distance',algorithm='brute', p=1)
knn_model.fit(X, Y)

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)

#### decission tree model agrithm
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X,Y)

dt_model_pred = dt_model.predict([[6.5]])
print(dt_model_pred)

# Random Forest Algrithm

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X,Y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)
