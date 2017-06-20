import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data_new = pd.read_csv('challenge_dataset.txt', header=None, names=['a', 'b'])
print data_new
X = data_new['a'].values.reshape(-1, 1)
print data_new['a']
print data_new[['a']]
y = data_new['b']

lin_reg_model = linear_model.LinearRegression()
lin_reg_model.fit(X, y)

plt.scatter(X, y, marker='x')
plt.scatter(X, lin_reg_model.predict(X), marker='.')
plt.plot(X, lin_reg_model.predict(X))
plt.show()
