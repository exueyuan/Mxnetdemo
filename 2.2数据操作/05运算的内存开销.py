from mxnet import nd

x = nd.arange(12)
# 改
X = x.reshape((3, 4))
Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X)
print(Y)

before = id(Y)
Y = Y + X
print(id(Y) == before)

#可以用索引来进行替换操作，节省内存
Z = Y.zeros_like()
before = id(Z)
Z[:] = X+Y
print(id(Z) == before)

#避免临时内存开销
nd.elemwise_add(X, Y, out=Z)
print(id(Z) == before)

before = id(X)
X+=Y
print(id(X) == before)
