from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
x=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int)
clf = LogisticRegression()
clf.fit(x,y)
example=clf.predict(([[2.6]]))
print(example)
Xaxis=np.linspace(0,3,1000).reshape(-1,1)
Yaxis=clf.predict_proba(Xaxis)
plt.plot(Xaxis, Yaxis[:,1], "g-")
plt.show()