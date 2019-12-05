from mxnet import autograd, nd

x = nd.arange(4).reshape((4, 1))
# 调用attach_grad来申请求梯度所需的内存
x.attach_grad()
# 调用record函数来要求MXNet记录与求梯度有关的计算
with autograd.record():
    y = 2 * nd.dot(x.T, x)
y.backward()

# 验证是否正确
assert (x.grad - 4 * x).norm().asscalar() == 0
print((x.grad - 4 * x).norm().asscalar() == 0)
print(x.grad)

# 训练模型还是预测模型
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())


# 对python的控制流求梯度

def f(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = nd.random.normal(shape=4)
print(a)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()

print(a.grad == c / a)
