import numpy as np
import scipy.io as sio
#import input_data
import pickle as pk
import random
import matplotlib.pyplot as plt
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()
Sess = tf.Session()
net_size = 90
x = tf.placeholder("float", [None, net_size*net_size])
y_ = tf.placeholder("float", [None, 2])
data_dir = r'Alltfdata.pkl'
#data_dir = 'Huaxim.mat'
emptyTrain = 1
'''
data1 = sio.loadmat(data_dir)
label1 = data1['label2']
data2 = data1['r_vec_net']
data_dic = {'label':label1,'data':data2}
'''
with open(data_dir,'rb') as f:
    data_dic = pk.load(f)

def scale_to_01(data_x):
    max_num = float(np.max(data_x))
    return np.multiply(data_x , 1.0/max_num)

def data_next_batch(batch_size,batch_num,data_x,data_y):
    if(len(data_x)!= len(data_y)):
        print 'size not ok!'
        return 
    start = batch_size * batch_num
    end = start + batch_size
    if(end>=len(data_x)):
        return data_x[start:],data_y[start:]
    return data_x[start:end],data_y[start:end]
'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.initialize_all_variables())

y = tf.softmax(tf.matmul(x,W)+b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
'''

def weight_variable(shape,lam):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lam)(var))
    return var

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME')

#First layer of convolution, activation function and max pooling.
fm_size = 75
W_conv1 = weight_variable([1, net_size, 1, fm_size],lam=0.5)
b_conv1 = bias_variable([fm_size])

x_image = tf.reshape(x, [-1, net_size, net_size, 1])
#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

'''
#h_pool1 = max_pool_2x2(h_conv1)
'''
#Second layer
h_conv1_rsp = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
h_conv1 = tf.reshape(h_conv1_rsp,[-1, net_size,fm_size])
x_image_mat = tf.reshape(x_image, [-1, net_size,net_size])

W_conv2 = weight_variable([1, fm_size, 1, fm_size],lam=0.3)
b_conv2 = bias_variable([fm_size])
order_1 = tf.reshape(tf.matmul(x_image_mat, h_conv1),[-1,net_size,fm_size])
h_conv2_rsp = tf.nn.sigmoid(conv2d(tf.reshape(h_conv1 + order_1,[-1,net_size,fm_size,1]), W_conv2) + b_conv2)

#Third layer
h_conv2 = tf.reshape(h_conv2_rsp,[-1, net_size,fm_size])

W_conv3 = weight_variable([1, fm_size, 1, fm_size],lam=0.1)
b_conv3 = bias_variable([fm_size])
order_2 = tf.reshape(tf.matmul(x_image_mat, h_conv2),[-1,net_size,fm_size])
h_conv3 = tf.nn.sigmoid(conv2d(tf.reshape(h_conv2+order_2+order_1,[-1,net_size,fm_size,1]), W_conv3) + b_conv3)

#Fourth layer
h_conv3 = tf.reshape(h_conv2_rsp,[-1, net_size,fm_size])

W_conv4 = weight_variable([1, fm_size, 1, fm_size],lam=0.05)
b_conv4 = bias_variable([fm_size])
order_3 = tf.reshape(tf.matmul(x_image_mat, h_conv3),[-1,net_size,fm_size])
h_conv4 = tf.nn.sigmoid(conv2d(tf.reshape(h_conv3+order_3+order_2+order_1,[-1,net_size,fm_size,1]), W_conv4) + b_conv4)

'''
h_conv2_rsp = tf.nn.relu(conv2d(x_image, W_conv2) + b_conv1)
h_conv2 = tf.reshape(h_conv2_rsp,[-1, 116, fm_size])

W_conv3 = weight_variable([1, fm_size, 1, fm_size])
b_conv3 = bias_variable([fm_size])
order_2 = tf.reshape(tf.matmul(x_image_mat, h_conv2),[-1,116,fm_size])
h_conv3 = tf.nn.sigmoid(conv2d(tf.reshape(h_conv2 + order_2,[-1,116,fm_size,1]), W_conv3) + b_conv3)

h_conv2_rsp = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv2 = tf.reshape(h_conv2_rsp,[-1,116,116,1])
h_pool2 = max_pool_2x2(h_conv2)
'''
fc_size = 90
fc_size2 = 2

#FC layer
W_fc1 = weight_variable([net_size*fm_size,fc_size],lam=0.1)
b_fc1 = bias_variable([fc_size])

h_pool1_flat = tf.reshape(h_conv3, [-1, net_size*fm_size])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Output layer
W_fc2 = weight_variable([fc_size, fc_size2],lam=0.1)
b_fc2 = bias_variable([fc_size2])
'''
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([fc_size2, 2])
b_fc3 = bias_variable([2])
'''
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Train and evaluate
saver = tf.train.Saver()
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#Weight reg 
tf.add_to_collection("losses",cross_entropy)
loss = tf.add_n(tf.get_collection("losses"))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
if emptyTrain:
    sess.run(tf.initialize_all_variables())
else:
    saver.restore(sess, "./mnistnnsave/model.ckpt")
Iteration = 20000
data_dic
#train_x = scale_to_01(np.array(data_dic['data']))
train_x = np.array(data_dic['data'])
train_y = np.array(data_dic['label'])
#shuffle the train data
train = np.hstack((train_x,train_y))
train_list = train.tolist()
#random.shuffle(train_list)
train = np.array(train_list)
train_acc_fold = []
test_acc_fold = []
val_acc_fold = []
init_op = tf.global_variables_initializer()
fsp = 70
#train = np.delete(train,[k for k in range(210,350)],axis = 0)
#train = np.delete(train,[k for k in range(280,350)],axis = 0)
all_num = len(train)
print len(train)
for j in range(10):
    sess.run(init_op)
    tr_idx = [k for k in range(all_num)]
    te_idx = [k for k in range(j*fsp,(j+1)*fsp)]
    va_idx = [k for k in range((j+1)%10*fsp,(j+2)%10*fsp)]
    tr_idx[j*fsp:(j+1)*fsp]=[]
    tr_idx[(j)%10*fsp:(j+1)%10*fsp]=[]
    if j == 8:
        va_idx = [k for k in range(9*fsp,10*fsp)]
    if j ==9:
        tr_idx[(j+1)%10*fsp:(j+2)%10*fsp]=[]
    train_x = train[tr_idx,:-2]
    print len(train_x)
    train_y = train[tr_idx,-2:]
    test_x = train[te_idx,:-2]
    test_y = train[te_idx,-2:]
    val_x = train[va_idx,:-2]
    val_y = train[va_idx,-2:]
    data_num = len(train_x)
    batch_size = 100
    batch_num = data_num/batch_size
    train_acc_list=[]
    test_acc_list = []
    val_acc_list = []
    x_iter_list = []
    #merged_summary_op = tf.merge_all_summaries()
    #summary_writer = tf.train.SummaryWriter('/tmp/lung_image_logs', sess.graph)
    for i in range(Iteration):
        batch = data_next_batch(batch_size,i%batch_num,train_x,train_y)
        #batch1 = mnist.train.next_batch(50)
        if i%100 ==0:
            x_iter_list.append(i)
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_:batch[1],keep_prob:1.0})
            print "step %d, fold %d,trainning accuracy %g"%(i, j, train_accuracy)
            train_acc_list.append(train_accuracy)
            test_accuracy = accuracy.eval(feed_dict={
                x:test_x, y_:test_y,keep_prob:1.0})
            #print "step %d, validation accuracy %g"%(i, test_accuracy)
            test_acc_list.append(test_accuracy)
            val_accuracy = accuracy.eval(feed_dict={
                x:val_x, y_:val_y,keep_prob:1.0})
            print "step %d, validation accuracy %g"%(i, val_accuracy)
            val_acc_list.append(val_accuracy)
            #summary_str = sess.run(merged_summary_op)
            #summary_writer.add_summary(summary_str, total_step)
            #saver.save(sess, "mnistnnsave1/model.ckpt")
        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    train_acc_fold.append(train_acc_list)
    test_acc_fold.append(test_acc_list)
    val_acc_fold.append(val_acc_list)

train_acc_list = []
test_acc_list = []
val_acc_list = []
dic = {'train':train_acc_fold,'test':test_acc_fold,'val':val_acc_fold}
ff = open('Alltfacc.pkl','wb')
pk.dump(dic,ff)
ff.close()
tr_acc = np.array(train_acc_fold[0])
te_acc = np.array(test_acc_fold[0])
for k in range(1,10):
    tr_acc = tr_acc + train_acc_fold[k]
    te_acc = te_acc + test_acc_fold[k]
tr_acc = tr_acc/10.0
te_acc = te_acc/10.0
train_acc_list = tr_acc.tolist()
test_acc_list = te_acc.tolist()
y1 = train_acc_list
y2 = test_acc_list
x1 = x_iter_list
plt.plot(x1,y1,label = 'train acc',color = 'r')
plt.plot(x1,y2,label = 'val acc')
plt.xlabel('iteration num')
plt.ylabel('acc')
plt.show()

'''
output = open('acc_point_1.pkl', 'wb')
pk.dump([train_acc_list,test_acc_list],output)
output.close()
'''
'''
for i in range(Iteration):
    batch = mnist.train.next_batch(50)
    if i%100 ==0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_:batch[1],keep_prob:1.0})
        print "step %d, trainning accuracy %g"%(i, train_accuracy)
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
'''
