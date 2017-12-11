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
