#结果可视化
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size])) #矩阵
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases #矩阵乘法
	if activation_function is None: #线性方程
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#添加一个隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
#添加一个输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.square(ys - prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	#图形化 start
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x_data, y_data)
	plt.ion()
	plt.show()
	#图形化 end

	for i in range(1000):
		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
		if i % 50 == 0:
			# print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
			try:
				ax.lines.remove(lines[0]) #移除lines第一个的图像
			except Exception:
				pass
			prediction_value = sess.run(prediction, feed_dict={xs: x_data})
			lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
			plt.pause(0.5)
