import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt

# 随机产生1000个点  线性模型y=0.1x+0.3

num_points = 1000
vectors_sets = []
for i in range(num_points):
    x1 = np.random.normal(0.0,0.55)#生成一个均值事0.0，方差事0.55的高斯分布
    y1 = x1*0.1+0.3+np.random.normal(0.0,0.05)
    vectors_sets.append([x1,y1])

x_data = [v[0] for v in vectors_sets]
y_data = [v[1] for v in vectors_sets]

#画图
plt.scatter(x_data,y_data,c='r')
plt.show()

#开始使用TensorFlow
w = tf.Variable(tf.random_uniform([1],-1.0,1.0),name='w')
b = tf.Variable(tf.zeros([1],name='b'))
#使用模型得到y
y = w * x_data + b
#实际值和预测值y之间的均方差做loss参数
loss = tf.reduce_mean(tf.square(y - y_data),name='loss')
#梯度下降优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
#训练模型
train = optimizer.minimize(loss,name='train')

#启动TensorFlow
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#最开始的wbloss等值 用sess.run获取展示
print("w=",sess.run(w))
print("b=",sess.run(b))
print("loss=",sess.run(loss))

print("========以下为训练数据=========")
#训练20次观察参数
for step in range(20):
    sess.run(train)
    print ("w=",sess.run(w),"b=",sess.run(b),"loss=",sess.run(loss))

#实际数据和训练对比
plt.scatter(x_data,y_data,c='r')
plt.plot(x_data,sess.run(w)*x_data+sess.run(b),c='g')
plt.show()

