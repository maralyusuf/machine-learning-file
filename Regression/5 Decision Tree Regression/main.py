import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


x = np.arange(0,15,0.09).reshape(-1,1)
y = np.cos(x).ravel()

dtr = DecisionTreeRegressor(max_depth=100)
dtr.fit(x,y)


plt.plot(x,dtr.predict(x),color="black")
plt.scatter(x,y,color="darkorange")
plt.show()
