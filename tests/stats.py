import linearity as ln
import numpy as np
import time


X = [
    ln.random(200) for i in range(20)
]
Y = [
    np.array(X[i].value) for i in range(len(X))
]


start = time.time()
variance = ln.var(X)
print(time.time() - start)

start = time.time()
variance = np.var(Y, axis=0)
print(time.time() - start)
