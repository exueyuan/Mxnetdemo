from mxnet import nd
import numpy as np

P = np.ones((2,3))
D = nd.array(P)
print(D)

print(D.asnumpy())