import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read old data
data_frame_old = pd.read_fwf('brain_body.txt')
X_old = data_frame_old[['Brain']]
y_old = data_frame_old[['Body']]

# read new challenge data
data_frame_new = pd.read_csv('challenge_dataset.txt', header=None, names=['x2', 'y2'])
print data_frame_new
# X = data_new['a'].values.reshape(-1, 1)
# print data_new['a']
# print data_new[['a']]
X = data_frame_new[['x2']]
y = data_frame_new[['y2']]

lin_reg_model = linear_model.LinearRegression()
lin_reg_model.fit(X, y)

plt.scatter(X, y, marker='x')
# plt.scatter(X_old, y_old, marker='o')
plt.plot(X, lin_reg_model.predict(X))
plt.scatter(X, lin_reg_model.predict(X), marker='o')
plt.show()
