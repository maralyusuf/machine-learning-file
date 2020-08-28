import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


dataset = pd.read_csv("datset2.csv")



x = dataset.iloc[:,:1].values
y = dataset.iloc[:,1:].values

linear = LinearRegression()

linear.fit(x,y)

pred = linear.predict([[20]])

print(pred)

plt.scatter(x,y,color="red")
plt.plot(x,linear.predict(x))
plt.show()
