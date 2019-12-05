from mxnet import nd

A = nd.arange(3).reshape((3, 1))
B = nd.arange(2).reshape((1, 2))

print(A)
print(B)

print(A+B)