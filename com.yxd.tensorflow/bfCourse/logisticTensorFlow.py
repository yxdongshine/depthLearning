import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

#获取数据
mnist = read_data_sets('../data/',one_hot=True)

#下载下来的数据集被分三个子集：
#5.5W行的训练数据集（mnist.train），
#5千行的验证数据集（mnist.validation)
#1W行的测试数据集（mnist.test）。
#因为每张图片为28x28的黑白图片，所以每行为784维的向量。
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("MNIST loaded")

#查看训练集和数据集合的具体数据格式
print (trainimg.shape)
print (trainlabel.shape)
print (testimg.shape)
print (testlabel.shape)
print(mnist.train.next_batch)
#print (trainimg)
print (trainlabel[0])

x=tf.placeholder("float",[None,784])#784是维度，none表示的是无限多
y=tf.placeholder("float",[None,10])
W=tf.Variable(tf.zeros([784,10]))#每个数字是784像素点的，所以w与x相乘的话也要有784个，b-10表示这个10分类的
b=tf.Variable(tf.zeros([10]))
#回归模型  w*x+b
actv=tf.nn.softmax(tf.matmul(x,W)+b)
#cost function 均值
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),reduction_indices=1))
#优化
learning_rate=0.01
#使用梯度下降，最小化误差
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


pred=tf.equal(tf.argmax(actv,1),tf.argmax(y,1))
#正确率
accr=tf.reduce_mean(tf.cast(pred,"float"))
#初始化
init=tf.global_variables_initializer()

# 每多少次迭代显示一次损失
training_epochs=50
#批尺寸
batch_size=100
# 训练迭代次数
display_step=5
#session
sess=tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost=0.
    #55000/100
    num_batch=int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        #获取数据集 next_batch获取下一批的数据
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        #模型训练
        sess.run(optm,feed_dict={x:batch_xs,y:batch_ys})
        feeds={x:batch_xs,y:batch_ys}
        avg_cost+=sess.run(cost,feed_dict=feeds)/num_batch
    #满足5次的一个迭代
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print ("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
               % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print ("DONE")
