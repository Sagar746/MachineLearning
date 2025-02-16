import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv('ridgereg_data.csv')
x = df[['x']]
y = df[['y']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

alphas = np.linspace(0,2, 50)
# print(alphas)

r2_values = []
for alp in alphas:
    rr = Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    r2_test = r2_score(y_test,rr.predict(X_test))
    r2_values.append(r2_test)

plt.plot(alphas, r2_values)
plt.show()

best_r2 = max(r2_values)
idx = r2_values.index(best_r2)
best_alp = alphas[idx]
print(f'\nBest R2: {best_r2}, Best alpha: {best_alp}')