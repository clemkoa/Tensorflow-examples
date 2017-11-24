batch_size = 50
trunctated_length = 5

inputs = tf.placeholder(tf.float32, shape = [batch_size, input_dim], name = "inputs")
target = tf.placeholder(tf.float32, shape = [batch_size], name = "target")

weights = tf.random_normal([output_dim, hidden_dim])
bias = tf.random_normal([output_dim])

lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
state = lstm.zero_state(batch_size, tf.float32)


for i in range(trunctated_length):
    h, state = lstm(inputs, state)
    Wo_s = tf.matmul(weights, tf.transpose(h))
    Wo_s = tf.add(bias, Wo_s, name = 'Wo_s')
    output = output_activation(Wo_s, name = 'output')

errors = tf.abs(output - target)
loss = tf.reduce_mean(errors, name = 'loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch_number):
        indices = np.random.choice(input_length, batch_size)
        data_dict = {
            inputs: X[indices],
            target: Y[indices]
        }
        t, l, o = sess.run([train_op, loss, output], feed_dict =  data_dict)
        if epoch % display_step == 0 or epoch == 1:
            print("Step " + str(epoch) + ", Minibatch Loss= " + "{:.4f}".format(l))
