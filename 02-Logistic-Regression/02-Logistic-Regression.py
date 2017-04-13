# My-Tensorflow-Practice
# 02-Logistic-Regression-GPU
# Author: OddNo7
# Date created: 2017/3/24
# Reference: aymericdamien/TensorFlow-Examples
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

# I wrote this for getting a hold of Tensorflow and for fun. For official
# documentations, please go to tensorflow.org
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# The idea of logistic regression is to let the log odds of y given x can
# be described by Wx+b. That is, ln(p(y=1|x)/p(y=-1|x))=(x.T)*w+b.
# This would yield a sigmoid function with W and b unknown.
# For multiclass problems, we can use softmax in replace of sigmoid.


mnist = input_data.read_data_sets("./tmp/data", one_hot=True)
num_train = 55000

batch_size = 50
epochs = 50
lr = 0.01
Xte = mnist.test.images
Yte = mnist.test.labels

with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.random_uniform([784, 10]), dtype=tf.float32)
    b = tf.Variable(tf.random_uniform([10]), dtype=tf.float32)
    pred = tf.matmul(x, W) + b  # Use cross entropy for this multiclass classification problem
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    test_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))

num_batch = int(num_train / batch_size)
# Train the model
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        cost_this_epoch = 0
        for i in range(num_batch):
            xtr, ytr = mnist.train.next_batch(batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={x: xtr, y: ytr})
            cost_this_epoch += l * batch_size  # Calculate total loss
        print('Epoch {} done. Loss: {:5f}'.format(epoch, cost_this_epoch))
    acc = sess.run(accuracy, feed_dict={x: Xte, y: Yte})
    print('Accuracy is {:2f}%'.format(acc * 100))
