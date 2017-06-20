import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_fwf('brain_body.txt')
print data
X = data[['Brain']]
y = data[['Body']]
print X
print y

# train data
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X, y)

plt.scatter(X, y)
# predict data
plt.plot(X, lin_reg.predict(X))
# show difference with predict and real data
plt.scatter(X, lin_reg.predict(X), marker='x')
plt.show()
