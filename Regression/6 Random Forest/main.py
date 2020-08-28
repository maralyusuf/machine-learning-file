import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


x= np.arange(0,15,0.09).reshape(-1,1)
y = np.cos(x).ravel()

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x,y)


plt.plot(x,rfr.predict(x),color="black")
plt.title("tree 100")
plt.scatter(x,y,color="red")
plt.show()
