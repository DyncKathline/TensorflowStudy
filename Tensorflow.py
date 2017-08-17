import tensorflow as tf
import numpy as np

## 数据准备阶段
# 管道中的液体：训练数据
# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 管道的阀门：权重和偏移
# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# 训练的方法设置
# Minimize the mean squared errors.
# 损失函数设置：平方损失函数
loss = tf.reduce_mean(tf.square(y - y_data))
# 优化方法设置：梯度下降方法，学习速率为0.5，一般在0~1之间
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练方式：最小化损失函数
train = optimizer.minimize(loss)

# 阀门状态的初始化
# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()


## 执行阶段
# Launch the graph.
# 创建会话
sess = tf.Session()
# 执行初始化操作
sess.run(init)

# Fit the line.
for step in range(201):
    # 执行训练过程
    sess.run(train)
    if step % 20 == 0:
        # 执行取出权重和偏移值操作
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]