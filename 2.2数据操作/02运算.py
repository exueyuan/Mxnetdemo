from mxnet import nd

x = nd.arange(12)
print(x)
print(x.shape)
print(x.size)
# 改
X = x.reshape((3, 4))
Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)

print(X + Y)
print(X * Y)
print(X / Y)
print(Y.exp())
print(nd.dot(X, Y.T))

# 0是最外层的，1是最内层的，越来越向里层
print(nd.concat(X, Y, dim=0))
print(nd.concat(X, Y, dim=1))

print(X == Y)

print(X < Y)
print(X > Y)

print(X.sum())

# L2范数
print(X.norm())
print(X.norm().asscalar())
