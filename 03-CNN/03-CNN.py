# My-Tensorflow-Practice
# 03-CNN
# Author: OddNo7
# Last modified: 2017/04/13
# Reference: aymericdamien/TensorFlow-Examples
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
# I wrote this for getting a hold of Tensorflow and for fun. For official
# documentations, please go to https://www.tensorflow.org

import tensorflow as tf
from CNNutils import load_data, preprocess_image, BatchGenerator

# Load cifar-10 dataset, use the first 1000 test samples for testing
train_set = load_data(mode='train')
test_set = load_data(mode='test')
Xtrain, Ytrain = preprocess_image(train_set)
Xtest, Ytest = preprocess_image(test_set)
Xtest = Xtest[45000:, :, :, :]
Ytest = Ytest[45000:]

# Create batch generator. Set mini-batch size.
gen = BatchGenerator(Xtrain, Ytrain, batch_size=128).initialize()
batch_size = 128
num_samples = Xtrain.shape[0]
image_size = Xtrain.shape[1]
image_channel = Xtrain.shape[3]

# Begin the construction of neural network
# Define input, weights, biases.
X = tf.placeholder(tf.float32, [None, image_size, image_size, image_channel])
Y = tf.placeholder(tf.int64, [None])
Y_hot = tf.one_hot(Y, depth=10, on_value=1, off_value=0)

weights = {
    'conv1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    'conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'conv3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'fc1': tf.Variable(tf.random_normal([8 * 8 * 64, 1024])),
    'fc2': tf.Variable(tf.random_normal([1024, 10]))
}

bias = {
    'fc1': tf.Variable(tf.random_normal([1024])),
    'fc2': tf.Variable(tf.random_normal([10]))
}

# Network strcutre:
# Convolution -> Convolution -> flatten -> fully connect -> fully connect
conv1 = tf.nn.conv2d(X, weights['conv1'], strides=[1, 1, 1, 1],
                     padding='SAME', use_cudnn_on_gpu=True)
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
conv2 = tf.nn.conv2d(conv1, weights['conv2'], strides=[1, 1, 1, 1],
                     padding='SAME', use_cudnn_on_gpu=True)
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
flat1 = tf.contrib.layers.flatten(conv2)
fc1 = tf.add(tf.matmul(flat1, weights['fc1']), bias['fc1'])
fc1 = tf.nn.relu(fc1)
fc2 = tf.add(tf.matmul(fc1, weights['fc2']), bias['fc2'])


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_hot, logits=fc2))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.equal(tf.argmax(fc2, axis=1), Y)
accr = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        for batch_idx in range(num_samples // batch_size):
            xbatch, ybatch = next(gen)
            test, _ = sess.run([loss, optimizer], feed_dict={X: xbatch, Y: ybatch})
            if batch_idx % 50 == 0:
                print('Batch {}, batch loss: {:.5f}'.format(batch_idx, test))
    accuracy = sess.run([accr], feed_dict={X: xbatch, Y: ybatch})
    print('Accuracy is {}'.format(accuracy))
