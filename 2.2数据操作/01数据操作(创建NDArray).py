from mxnet import nd

# 第一种
x = nd.arange(12)

print(x)

print(x.shape)

print(x.size)
# 改姓张
X = x.reshape((3, 4))

print(X)

# 全0
print(nd.zeros((2, 3, 4)))

# 全1
print(nd.ones((3, 4)))

# 列表转化为nd
Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)

# 正态分布nd
print(nd.random.normal(0, 1, shape=(3, 4)))
