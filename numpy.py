import numpy as np

ll=[0,1,2]
np.array(ll)
_.shape

ll1=[[0,1,2]]
np.array(ll1)
_.shape

np.array(ll)[None,:]
np.array(ll1).ravel()

np.array([[0,0,0,0]]).shape
np.array([0,0,0,0]).shape

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))

vv.shape
print(vv)

np.empty_like(x)

######
##http://www.labri.fr/perso/nrougier/from-python-to-numpy/
import random
import numpy as np
from timeit import timeit
from itertools import accumulate

def random_walk_faster(n=1000):
    # Only available from Python 3.6
    steps = random.choices([-1,+1], k=n)
    return [0]+list(accumulate(steps))

walk = random_walk_faster(1000)
timeit("random_walk_faster(n=10000)", number=100, globals=globals())

def random_walk_fastest(n=1000):
    # No 's' in NumPy choice (Python offers choice & choices)
    steps = np.random.choice([-1,+1], n)
    return np.cumsum(steps)
timeit("random_walk_fastest(n=10000)", number=100, globals=globals())

Z=np.arange(9).reshape(3,3).astype(np.int16)
Z.itemsize
Z.ndim
Z[::2,::2]

Z = np.random.uniform(0,1,(5,5))
#view
Z1 = Z[:3,:]
#copy
Z2 = Z[[0,1,2], :]
print(np.allclose(Z1,Z2))

print(Z1.base is Z)
print(Z2.base is Z)
print(Z2.base is None)

Z = np.zeros((5,5))
#ravel returns view
Z.ravel().base is Z
#flatten returns copy
Z.flatten().base is Z

Z1 = np.arange(10)
#Z1[start:stop:step]
Z2 = Z1[1:-1:2]
print(Z2.base is Z1)

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))
np.argwhere(A.ravel() > B.ravel())
np.linspace(1, 25, 10, dtype=np.float32)
np.less(.5, A)

np.random.randint(0,9,(3,3))
np.linspace(0, 1, 5)