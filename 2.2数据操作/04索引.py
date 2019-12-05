from mxnet import nd

x = nd.arange(12)
# 改
X = x.reshape((3, 4))
Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X)
print(Y)

#左闭右开
print(X[1:3])

X[1,2] = 9
print(X)

X[1:2, :] = 12
print(X)