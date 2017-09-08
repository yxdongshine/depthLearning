import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

#获取数据
mnist = read_data_sets('../data/',one_hot=True)
print("type: %s: " %(type(mnist)))
print("count of train data: %d"%(mnist.train.num_examples))
print("count of test data: %d"%(mnist.test.num_examples))

#解析训练集和测试集
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print (" shape of 'trainimg' is %s"   % (trainimg.shape,))#28*28,总共784个像素
print (" shape of 'trainlabel' is %s" % (trainlabel.shape,))#0-9十个标签
print (" shape of 'testimg' is %s"    % (testimg.shape,))
print (" shape of 'testlabel' is %s"  % (testlabel.shape,))

# NETWORK TOPOLOGIES
n_hidden_1 = 256 #第一隐藏层神经元数
n_hidden_2 = 128 #第二隐藏层神经元数
n_input = 784 #每个数据输入的长度数
n_classes = 10 #输出的label数
#先定义输入输出层数据 格式
x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

#神经网络，设立2个隐藏，填充数据
stddev=0.1
weights={
    "w1":tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),#产生n_input到n_hidden_1的均方差为stddev的连续数组
    "w2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
    "out":tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
}
biases={
    "b1":tf.Variable(tf.random_normal([n_hidden_1])),
    "b2":tf.Variable(tf.random_normal([n_hidden_2])),
    "out":tf.Variable(tf.random_normal([n_classes]))
}
print ("NETWORK READ")

#s激活函数，进行数据的操作 （BP算法）
def multilayer_perceotron(_X, _weights, _biases):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']),_biases['b1']))#每次wx+b之后经过sigmoid函数激活
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']),_biases['b2']))
    return (tf.matmul(layer_2, _weights['out'])+_biases['out'])#返回的是10个输出
pred=multilayer_perceotron(x,weights,biases)

#损失函数 softmax_cross_entropy_with_logits中0.x版本和1.x不同的是1.x要加logits和labels
#得到损失值
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
#执行GradientDescentOptimizer梯度下降以及minimize优化
optm=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
#如果相等返回true，不等返回flase
corr=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
#把true转为1，false为0，再相加，计算总的（正确率）
accr=tf.reduce_mean(tf.cast(corr,"float"))
#初始化
init=tf.global_variables_initializer()
print ("FUNCTIONS READY")
#以上是线性模型的神经模型 下面开始训练这个个数据

#执行模型的训练
training_epochs = 50#迭代的次数
batch_size      = 100#每次处理的图片数
display_step    = 4#每次次打印一次
# LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)
# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.   #平均误差
    total_batch = int(mnist.train.num_examples/batch_size)
    # ITERATION  训练集循环次数确定
    for i in range(total_batch):
        #去除x和y的值
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    # DISPLAY  显示误差率和训练集的正确率以此测试集的正确率
    if (epoch+1) % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print ("TRAIN ACCURACY: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print ("TEST ACCURACY: %.3f" % (test_acc))
print ("OPTIMIZATION FINISHED")
#最后的损失函数下降的不是很多，正确率也不高，需要加大迭代次数


