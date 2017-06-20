from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt

lin_reg_model = linear_model.LinearRegression()

# prepare data
data_frame = pd.read_csv('challenge_dataset.txt', header=None, names=['X', 'y'])
print data_frame
X_all = data_frame[['X']]
y_all = data_frame[['y']]

# split data set
X_train = X_all[:-20]
X_test = X_all[-20:]
y_train = y_all[:-20]
y_test = y_all[-20:]

# train
lin_reg_model.fit(X_train, y_train)

# plotting
plt.scatter(X_train, y_train, color='red', marker='.')
plt.scatter(X_test, y_test, color='green', marker='.')
plt.scatter(X_test, lin_reg_model.predict(X_test), color='orange', marker='x')
plt.plot(X_test, lin_reg_model.predict(X_test))
plt.show()
