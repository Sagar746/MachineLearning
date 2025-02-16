import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('linreg_data.csv')
# print(df.head())

# plt.scatter(df['cgpa'], df['package'])
# plt.xlabel('CGPA')
# plt.ylabel('PACKAGE')

# plt.show()


X = df.iloc[:,0:1 ]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(X_train, y_train)


pr = lr.predict(X_test.values)

m = lr.coef_
b = lr.intercept_
# print(m, b)
# print(m*8.58 +b)
print(pr)

# plt.scatter(df['cgpa'], df['package'])
# plt.plot(X_test, lr.predict(X_test), color='red')
# plt.xlabel('CGPA')
# plt.ylabel('PACKAGE')

# plt.show()
# print(X_test)


# error reducing

print('MAE: ', mean_absolute_error(y_test, pr))
print('MSE: ', mean_squared_error(y_test, pr))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, pr)))
print('R2-SCORE: ', r2_score(y_test, pr))