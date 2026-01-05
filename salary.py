import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv("Salary_Data.csv")
print(data.head())
X = data[['YearsExperience']]   # Independent variable
Y = data['Salary']              # Dependent variable
model = LinearRegression()
model.fit(X, Y)
y_pred = model.predict(X)
plt.scatter(X, Y)
plt.plot(X, y_pred)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression on Salary Dataset")
plt.show()
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
exp = [[5]]
salary = model.predict(exp)
print("Predicted Salary:", salary)
