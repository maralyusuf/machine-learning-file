import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR


x = np.arange(0,15,0.09).reshape(-1,1)
y = np.cos(x).ravel()

svr_r = SVR(kernel='rbf',gamma=0.22)
svr_r.fit(x,y)

svr_l = SVR(kernel='linear',gamma=0.22)
svr_l.fit(x,y)

svr_p = SVR(kernel='poly',gamma=0.22)
svr_p.fit(x,y)


plt.plot(x,svr_r.predict(x),color="red")
plt.plot(x,svr_l.predict(x),color="blue")
plt.plot(x,svr_p.predict(x),color="green")
plt.legend(labels=("rbf","linear","poly"))
plt.scatter(x,y,color="darkorange")
plt.show()
