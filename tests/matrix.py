import linearity as ln
import numpy as np
import time

start = time.time()
d = list(np.identity(5000))
print(time.time() - start)

start = time.time()
a = ln.matrix(d)
print(time.time() - start)
print(a.shape)
