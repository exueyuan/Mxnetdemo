from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

num_inputs = 2
num_examples = 1000
# 特征真实值
true_w = [2, -3.4]
true_b = 4.2

# 特征服从正太分布
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))

# 特征真实值
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

print(features[0], labels[0])


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# set_figsize()

print(features[:, 1].asnumpy()[0:5])
print(labels.asnumpy()[0:5])


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(features[:, 1].asnumpy(), labels.asnumpy())
plt.show()


# plt.scatter(features[:, 1].asnumpy(), labels.asnumpy());  # 加分号只显示图
