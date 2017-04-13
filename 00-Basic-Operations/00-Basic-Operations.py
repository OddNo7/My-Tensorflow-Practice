# My-Tensorflow-Practice
# 00-Basic-Operations
# Author: OddNo7
# Last modified: 2017/04/13

# I wrote this for getting a hold of Tensorflow and for fun. For official
# documentations, please go to https://www.tensorflow.org

import tensorflow as tf
import numpy as np

# Constant addition, reduction, multiplication and division
a = tf.constant(3, dtype=tf.float32, name='a')
b = tf.constant(4, dtype=tf.float32, name='b')
c = a + b
d = a - b
e = a * b
f = a / b

# Run the computational graph
sess = tf.Session()
print(sess.run([c, d, e, f]))

# A placeholder must be fed with data. On definition of a place holder dtype must be specified. One can also specify
# name and shape.
# The feeding usually goes with 'feed_dict' argument.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
with tf.Session() as sess:
    print(sess.run(a + b, feed_dict={a: 3, b: 4}))

# A variable is a value that can be trained during the running of a graph.
# A variable needs to be designated an initial value when created. On
# computation, this initial value need to be assigned as a initialization
# process.
W = tf.Variable(2.0, tf.float32, name='W')
b = tf.Variable(3.5, tf.float32, name='b')
x = tf.placeholder(tf.float32, name='x')
line = W * x + b
init = tf.global_variables_initializer()
# For initialization, one can call a variable's method 'initializer' to
# initiate the value, or use global initializer.

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(line, feed_dict={x: [1, 2, 3, 4]}))

# Loss function. To evaluate how good a model is, we need to find a loss function,
# With loss function we can try to train the variables so that the model
# fits the fed data.
y = tf.placeholder(tf.float32, name='y')
with tf.name_scope('Loss'):
    MSE = tf.reduce_mean(tf.square(line - y), name='MSE')
with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(MSE)

xtr = np.array([1.0, 2.0, 3.0, 4.0])
ytr = np.array([2.0, 3.0, 4.0, 5.0])
# Now one needs to minimize the loss function. To do that, we need to call some optimizer.
# tf.train provides several optimizers. We can use the basic one and build
# an instance, and call its minimize method.
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(2000):
        if epoch % 100 == 0:
            print('Reaching {}00-th epoch'.format(epoch // 100))
            print(sess.run(MSE, feed_dict={x: xtr, y: ytr}))
        sess.run(optimizer, feed_dict={x: xtr, y: ytr})
    print(sess.run([W, b]))
