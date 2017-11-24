import tensorflow as tf
import numpy as np
import random

input_length = 10000
input_max = 100
input_dim = 10
hidden_dim = 20
output_dim = 1
output_activation = tf.tanh
epoch_number = 501
display_step = 100
learning_rate = 0.001

# Just generate random data, and normalize the vectors for future use
X = np.random.randint(input_max, size=(input_length, input_dim)) * 1.0 / (input_max * input_dim)
Y = np.sum(X, axis=1)

inputs = tf.placeholder(tf.float32, shape=[1, input_dim], name="inputs")
target = tf.placeholder(tf.float32, shape=[1], name="target")
weights = tf.random_normal([output_dim, hidden_dim])
bias = tf.random_normal([output_dim])

# Note: you can use any BasicRRN cell, GRU, LSTM... 
lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
state = lstm.zero_state(1, tf.float32)

# Forward pass with non-linear activation
h, state = lstm(inputs, state)
Wo_s = tf.matmul(weights, tf.reshape(h, (hidden_dim, 1)))
Wo_s = tf.add(bias, Wo_s, name = 'Wo_s')
output = output_activation(Wo_s, name = 'output')

errors = output - target
loss = tf.reduce_mean(errors, name = 'loss')

# Some other optimizers like RMSProp might be better
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch_number):
        i = random.randint(0, input_length - 1)
        data_dict = {
            inputs: np.reshape(X[i], (1, -1)),
            target: np.array([Y[i]])
        }
        t, l, o = sess.run([train_op, loss, output], feed_dict =  data_dict)
        if epoch % display_step == 0 or epoch == 1:
            print("Step " + str(epoch) + ", Minibatch Loss= " + "{:.4f}".format(l))

