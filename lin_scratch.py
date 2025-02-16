import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('linreg_data.csv', names=['x','y'])

xpd = df['x']
ypd = df['y']

n = xpd.size

# plt.scatter(xpd,ypd)
# plt.show()

xbar = np.mean(xpd)
ybar = np.mean(ypd)


b = np.sum(xpd * ypd - xpd *ybar)/np.sum(xpd **2 - xpd * xbar)
a = ybar - b * xbar

x = np.linspace(0,2, 100)
y = a + b*x

plt.plot(x,y, color='black')
plt.scatter(xpd, ypd)
plt.scatter(xbar, ybar, color='red')
plt.show()

yhat = a + b * xpd

rmse = np.sqrt(np.sum((ypd - yhat)**2)/n)
print(f'rmse: {rmse}')

rss = np.sum((ypd-yhat)**2)
print(f'rss: {rss}')