# Use MNIST hand-written digits for this practice
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Specify a subset of MNIST as training set and test set
test_size = 400
train_size = 20000
# Since MNIST contains 10 categories, use a bigger K could possibly reach
# a better output.
K = 13

# Load data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
Xtr, Ytr = mnist.train.next_batch(train_size)
Xte, Yte = mnist.test.next_batch(test_size)

# Each time compute Euclidean distance across the training set
xtr = tf.placeholder(tf.float32, shape=[None, 784])
xte = tf.placeholder(tf.float32, shape=[784])  # Feed one test sample each time
ytr = tf.placeholder(tf.float32, shape=[None, 10])

# Calculate Euclidean distance for each dimension, then sum up
distance = tf.reduce_sum(tf.square(xtr - xte), reduction_indices=1)
_, ind = tf.nn.top_k(-distance, k=K)  # Fine indices of top K neighbors

nearest_neighbor = []
for i in range(K):
    nearest_neighbor.append(ytr[ind[i], :])
# Voting: Each neighbor vote for a class. Summing their ground truths
# therefore helps predict the test's label.
# Gather the y-label of these neighbors and sum up. Find argmax.
kneighbors = tf.transpose(tf.pack(nearest_neighbor, axis=1))
pred = tf.argmax(tf.reduce_sum(kneighbors, reduction_indices=0), axis=0)

count = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for index in range(len(Xte)):
        inpt = Xte[index, :]
        label = sess.run(pred, feed_dict={xtr: Xtr, ytr: Ytr, xte: inpt})
        print('{}-th test, pred={}, truth={}'.format(index,
                                                     label, np.argmax(Yte[index])))
        if label == np.argmax(Yte[index]):
            count += 1
    print('Total accuracy: {:f}%.'.format(count / test_size * 100))
