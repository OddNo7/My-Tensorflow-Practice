# My-Tensorflow-Practice
# 05-LSTM classification
# Author: OddNo7
# Last modified: 2017/06/12
# Reference:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
# I wrote this for getting a hold of Tensorflow and for fun. For official
# documentations, please go to https://www.tensorflow.org
# Import modules
import tensorflow as tf
import sys

sys.path.insert(0, './utils/')
from tensorflow.contrib.rnn import BasicLSTMCell, static_rnn
from CIFARutils import BatchGenerator, preprocess_image, load_data

# Load cifar-10 dataset, use the first 1000 test samples for testing
train_set = load_data(mode='train')
test_set = load_data(mode='test')
Xtrain, Ytrain = preprocess_image(train_set)
Xtest, Ytest = preprocess_image(test_set)
Xtest = Xtest[0:1000, :, :]
Ytest = Ytest[0:1000]

# Reshape data to target size. This essentially 'stretches' the data so each pixel's RGB value is lined up in the
#  same row.
Xtrain_seq = Xtrain.reshape([-1, 32, 96])
Xtest_seq = Xtest.reshape([-1, 32, 96])
print(Xtrain_seq.shape)

# Create batch generator. Set mini-batch size.
gen = BatchGenerator(Xtrain_seq, Ytrain, batch_size=128).initialize()
batch_size = 128
num_samples = Xtrain.shape[0]
sequence_length = 32
input_dim = 96
n_hidden = 512
num_class = 10

# Define TensorFlow variables
X_init = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
y = tf.placeholder(tf.int32, [None, ])
out_w = tf.Variable(tf.random_normal([n_hidden, num_class], dtype=tf.float32))
out_b = tf.Variable(tf.random_normal([num_class]), dtype=tf.float32)
y_hot = tf.one_hot(y, 10, on_value=1, off_value=0)

# Split input tensor to a list, so that it fits into RNN.
X = tf.split(X_init, sequence_length, axis=1)
X = [tf.squeeze(i, axis=1) for i in X]

# Construct LSTM network
lstm = BasicLSTMCell(n_hidden)
hidden_initial = tf.zeros([n_hidden,])
lstm_output, _ = static_rnn(lstm, X, dtype=tf.float32)

# Set loss, accuracy, etc.
predictions = tf.matmul(lstm_output[-1], out_w) + out_b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_hot))
optimizer = tf.train.AdadeltaOptimizer().minimize(loss)
test_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_hot, 1))
accuracy = tf.reduce_mean(tf.cast(test_pred, tf.float32))

# Start training the computation graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(40):
    for _ in range(50000 // 128):
        xbatch, ybatch = next(gen)
        sess.run([optimizer], feed_dict={X_init: xbatch, y: ybatch})
    l, acc = sess.run([loss, accuracy], feed_dict={X_init: Xtrain_seq, y: Ytrain})
    print('Epoch {}, loss {}, accuracy {}%'.format(epoch, l, acc * 100))

print('Optimization compmlete.')

# Do a simple test
acc = sess.run([loss], feed_dict={X_init: Xtest_seq, y:Ytest})
print('Accuracy on test set: {}'.format(acc[0]))
