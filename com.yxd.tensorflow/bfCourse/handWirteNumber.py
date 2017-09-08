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

#随机从样本训练集合中拿出5组数据 画图出来看
nsample = 3
randidx = np.random.randint(trainimg.shape[0],size=nsample)
for i in randidx:
    curr_img = np.reshape(trainimg[i,:],(28,28))#将该数据的第二位数据转换成28*28的二维数组
    curr_label = np.argmax(trainlabel[i,:])
    #画图展示
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("" + str(i) + "th Training Data "
              + "Label is " + str(curr_label))
    print("" + str(i) + "th Training Data "
          + "Label is " + str(curr_label))
    plt.show()

#选一批次数据详细信息展示
# Batch Learning?
print ("Batch Learning? ")
batch_size = 100
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
print ("type of 'batch_xs' is %s" % (type(batch_xs)))
print ("type of 'batch_ys' is %s" % (type(batch_ys)))
print ("shape of 'batch_xs' is %s" % (batch_xs.shape,))
print ("shape of 'batch_ys' is %s" % (batch_ys.shape,))