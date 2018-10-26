from data_loader import Dataset
import numpy as np
from collections import defaultdict
import tensorflow as tf
import matplotlib.pyplot as plt

train_debate = np.load('array/2train_debate.npy')
train_reason = np.load('array/2train_reason.npy')
train_claim = np.load('array/2train_claim.npy')
train_warrant = np.load('array/2train_warrant.npy')
train_label = np.load('array/2train_label.npy')

dev_debate = np.load('array/2dev_debate.npy')
dev_reason = np.load('array/2dev_reason.npy')
dev_claim = np.load('array/2dev_claim.npy')
dev_warrant = np.load('array/2dev_warrant.npy')
dev_label = np.load('array/2dev_label.npy')

test_debate = np.load('array/2test_debate.npy')
test_reason = np.load('array/2test_reason.npy')
test_claim = np.load('array/2test_claim.npy')
test_warrant = np.load('array/2test_warrant.npy')
test_label = np.load('array/2test_label.npy')

#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷基层
def conv1d(x,W):
    return tf.nn.conv1d(x,W,1,padding='SAME')

with tf.name_scope('Input'):
    debate = tf.placeholder(tf.float32,[None,48,300],name='debate')
    reason = tf.placeholder(tf.float32,[None,48,300],name='reason')
    claim = tf.placeholder(tf.float32,[None,48,300],name='claim')
    warrant = tf.placeholder(tf.float32,[None,48,300],name='warrant')
    y = tf.placeholder(tf.float32,[None,2],name='y')
with tf.name_scope('Layer1-CNN'):
    W_conv_debate0 = weight_variable([5,300,50])
    W_conv_reason0 = weight_variable([5,300,50])
    W_conv_claim0 = weight_variable([5,300,50])
    W_conv_warrant0 = weight_variable([5,300,50])

    b_conv_debate0 = bias_variable([50])
    b_conv_reason0 = bias_variable([50])
    b_conv_claim0 = bias_variable([50])
    b_conv_warrantn0 = bias_variable([50])

    h_conv_debate0 = tf.nn.relu(conv1d(debate,W_conv_debate0)+b_conv_debate0)#48*50
    h_conv_reason0 = tf.nn.relu(conv1d(reason,W_conv_reason0)+b_conv_reason0)
    h_conv_claim0 = tf.nn.relu(conv1d(claim,W_conv_claim0)+b_conv_claim0)
    h_conv_warrant0 = tf.nn.relu(conv1d(warrant,W_conv_warrant0)+b_conv_warrantn0)

attention = tf.matmul(h_conv_claim0,tf.transpose(h_conv_warrant0,[0,2,1]))
wei = tf.reduce_sum(attention,axis=1)
mwei = tf.reshape(tf.tile(wei,multiples=[1,50]),[-1,48,50])
h_conv_claim0 = tf.multiply(h_conv_claim0,h_conv_warrant0)

# with tf.Graph.control_dependencies([h_conv_debate0,h_conv_warrant0]):
#     result = []
    
#     for i in range(h_conv_debate0.shape()[0]):
#         result.append([sum(line,axis=1) for line in tf.matmul(h_conv_debate0[i],h_conv_warrant0[j],transpose_b=True)])
#     weight = tf.nn.softmax(np.array(result))
#     for m in h_conv_debate0:
#         for line,w in zip(m,weight):
#             line*=w
final_representation = tf.stack([h_conv_debate0,h_conv_reason0,h_conv_claim0,h_conv_warrant0],axis=2)

with tf.name_scope('Layer2-DNN'):
    W_fcl = weight_variable([48*4*50,100])
    b_fcl = bias_variable([100])

    h_flat = tf.reshape(final_representation,[-1,48*4*50])
    h_fcl = tf.tanh(tf.matmul(h_flat,W_fcl)+b_fcl)

    keep_prob = tf.placeholder(tf.float32)
    h_fcl_drop = tf.nn.dropout(h_fcl,keep_prob)

with tf.name_scope('Layer3-DNN'):
    W_fcl2 = weight_variable([100,50])
    b_fcl2 = bias_variable([50])

    h_fcl2 = tf.tanh(tf.matmul(h_fcl_drop,W_fcl2)+b_fcl2)
    h_fcl_drop2 = tf.nn.dropout(h_fcl2,keep_prob)

with tf.name_scope('Layer4-DNN'):
    W_fcl3 = weight_variable([50,30])
    b_fcl3 = bias_variable([30])

    h_fcl3 = tf.tanh(tf.matmul(h_fcl_drop2,W_fcl3)+b_fcl3)
    h_fcl_drop3 = tf.nn.dropout(h_fcl3,keep_prob)

with tf.name_scope('Layer5-DNN'):
    W_fcl4 = weight_variable([30,2])
    b_fcl4 = bias_variable([2])
    prediction = tf.nn.softmax(tf.matmul(h_fcl_drop3,W_fcl4)+b_fcl4)

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('Train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

def get_result(x,y):
	index = 0
	pred_list = []
	gold_list = []
	while index<len(x):
		if x[index][1] > x[index+1][1]:
			pred_list.append(0)
		else:
			pred_list.append(1)
		index+=2
	index = 0
	while index<len(y):
		if y[index][1] > y[index+1][1]:
			gold_list.append(0)
		else:
			gold_list.append(1)
		index+=2
	count = 0
	for p,g in zip(pred_list,gold_list):
		if p==g:
			count+=1
	print('x length',len(x),'count num',count)
    
	return count/len(x)*2

merged = tf.summary.merge_all()
#saver = tf.train.Saver()
with tf.Session() as sess:
    train_a = []
    dev_a = []
    sess.run(tf.global_variables_initializer())
    #writer = tf.summary.FileWriter('logs/',sess.graph)
    batch_size = 40
    n_batch = 2422 // batch_size
    for epoch in range(100):
        for batch in range(n_batch):
            train = sess.run(train_step,feed_dict={debate:train_debate[batch*n_batch:(batch+1)*n_batch]
            	,reason:train_reason[batch*n_batch:(batch+1)*n_batch]
            	,claim:train_claim[batch*n_batch:(batch+1)*n_batch]
            	,warrant:train_warrant[batch*n_batch:(batch+1)*n_batch]
            	,y:train_label[batch*n_batch:(batch+1)*n_batch],keep_prob:0.7})
        train = sess.run(train_step,feed_dict={debate:train_debate[2400:2422]
            	,reason:train_reason[2400:2422]
            	,claim:train_claim[2400:2422]
            	,warrant:train_warrant[2400:2422]
            	,y:train_label[2400:2422],keep_prob:0.7})

        train_pred = sess.run(prediction,feed_dict={debate:train_debate
        	,reason:train_reason
        	,claim:train_claim
        	,warrant:train_warrant
        	,y:train_label,keep_prob:1.0})
        
        dev_pred = sess.run(prediction,feed_dict={debate:dev_debate
        	,reason:dev_reason
        	,claim:dev_claim
        	,warrant:dev_warrant
        	,y:dev_label,keep_prob:1.0})
        
        train_acc = get_result(train_pred,train_label)
        #tf.summary.scalar('train accuracy',train_pred)
        dev_acc = get_result(dev_pred,dev_label)
        #tf.summary.scalar('develop accuracy',dev_pred)

        summary = sess.run(merged,feed_dict={debate:train_debate[batch*n_batch:(batch+1)*n_batch]
                ,reason:train_reason[batch*n_batch:(batch+1)*n_batch]
                ,claim:train_claim[batch*n_batch:(batch+1)*n_batch]
                ,warrant:train_warrant[batch*n_batch:(batch+1)*n_batch]
                ,y:train_label[batch*n_batch:(batch+1)*n_batch],keep_prob:0.7})

        #writer.add_summary(summary,epoch)
        print('Iterate ',epoch,', train accuracy: ',train_acc)
        print('Iterate ',epoch,', dev accuracy: ',dev_acc)
        train_a.append(train_acc)
        dev_a.append(dev_acc)
        print()
    # itera = [i for i in range(1,201)]
    # plt.plot(itera,train_a)
    # plt.xlabel('iteration times')
    # plt.show()
    # plt.plot(itera,dev_a)
    # plt.xlabel('iteration times')
    # plt.show()
    #saver.save(sess,save_path='/Users/zhoumeng/Documents/Sheffield/project/myCode/Deepsolution/parameter/-100iteration_reasoning_net.ckpt')




