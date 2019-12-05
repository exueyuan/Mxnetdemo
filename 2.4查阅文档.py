from mxnet import autograd, nd
#查找模块中的所有函数和类
print(dir(nd.random))

#查找特定函数和类的使用
help(nd.ones_like)

x = nd.array([[0,0,0],[2,2,2]])
y = x.ones_like()
print(y)

