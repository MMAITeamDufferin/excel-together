import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
import tensorflow as tf

# Extract http://archive.ics.uci.edu/ml/machine-learning-databases/00447/ to same directory as this script

# List files
one = ['.\\data\\TS1.txt', '.\\data\\TS2.txt', '.\\data\\TS3.txt', '.\\data\\TS4.txt', '.\\data\\VS1.txt', '.\\data\\CE.txt', '.\\data\\CP.txt', '.\\data\\SE.txt']
ten = ['.\\data\\FS1.txt', '.\\data\\FS2.txt']
hundred = ['.\\data\\PS1.txt', '.\\data\\PS2.txt', '.\\data\\PS3.txt', '.\\data\\PS4.txt', '.\\data\\PS5.txt', '.\\data\\PS6.txt', '.\\data\\EPS1.txt']

# Parse condition profiles
df_profile = pd.read_table('.\\data\\profile.txt', header=None)
df_profile = df_profile.values.reshape(2205, 1, 5)
df_profile = zoom(df_profile, (1,6000,1))

# Parse 1 Hz measurements
df_one =  np.stack([pd.read_table(x, header=None) for x in one], axis=2)
df_one = zoom(df_one, (1, 100, 1))

# Parse 10 Hz measurements
df_ten =  np.stack([pd.read_table(x, header=None) for x in ten], axis=2)
df_ten = zoom(df_ten, (1, 10, 1))

# Parse 100 Hz measurements
df_hundred = np.stack([pd.read_table(x, header=None) for x in hundred], axis=2)

# Concatenate all data
df = np.concatenate([df_profile, df_one, df_ten, df_hundred], axis=2)

# Split data into training, validation, and test sets
val = 0.2
test = 0.1
train = 1 - val - test

X_train = df[:int(train*df.shape[0])+1:,::,[i not in [1] for i in range(df.shape[2])]]
X_val = df[int(train*df.shape[0])+1:int(train*df.shape[0])+int(val*df.shape[0])+1:,::,[i not in [1] for i in range(df.shape[2])]]
X_test = df[int(train*df.shape[0])+int(val*df.shape[0])+1::,::,[i not in [1] for i in range(df.shape[2])]]

oh_target = (np.arange(df[:,0,1].max()+1) == df[:,0,1][...,None]).astype(int)
oh_target = np.delete(oh_target,np.where(~oh_target.any(axis=0))[0], axis=1)

y_train = oh_target[:int(train*oh_target.shape[0])+1:,]
y_val = oh_target[int(train*oh_target.shape[0])+1:int(train*oh_target.shape[0])+int(val*oh_target.shape[0])+1:,]
y_test = oh_target[int(train*oh_target.shape[0])+int(val*oh_target.shape[0])+1::,]

def sample_batch(X, y, batch_size):
    for b in range(0, len(X)-(len(X)%batch_size)-batch_size, batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]

# Constants
samples, seq_len, features = X_train.shape
n_classes = y_train.shape[1]

# Hyperparameters
lstm_size = 3*features
lstm_layers = 2
dropout = 0.8
batch_size = 50
learning_rate = 0.0001  # default is 0.001
epochs = 1

graph = tf.Graph()

with graph.as_default():
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, seq_len, features], name='inputs')
    with tf.name_scope("Target"):
        target = tf.placeholder(dtype=tf.int32, shape=[None, n_classes], name='target')
    keep_prob = tf.placeholder(tf.float32, name = 'keep')

with graph.as_default():
    lstm_in = tf.transpose(inputs, [1, 0, 2])  # reshape into (seq_len, samples, features)
    lstm_in = tf.reshape(lstm_in, [-1, features])  # Now (seq_len*samples, features)

    # To cells
    lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None)

    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, seq_len, 0)

    # Add LSTM layers
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32, initial_state=initial_state)

    # We only need the last output tensor to pass into a classifier
    logits = tf.layers.dense(outputs[-1], n_classes, name='logits')

    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target))

    # No grad clipping
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Grad clipping
    train_op = tf.train.AdamOptimizer(learning_rate)

    gradients = train_op.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    optimizer = train_op.apply_gradients(capped_gradients)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

if (os.path.exists('checkpoints') == False):
    os.system('mkdir checkpoints')

train_acc = []
train_loss = []

validation_acc = []
validation_loss = []

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    for e in range(epochs):
        # Initialize
        state = sess.run(initial_state)

        # Loop over batches
        for x, y in sample_batch(X_train, y_train, batch_size):

            # Feed dictionary
            feed = {inputs: x, target: y, keep_prob: dropout, initial_state: state}

            loss, _, state, acc = sess.run([cost, optimizer, final_state, accuracy],
                                           feed_dict=feed)
            train_acc.append(acc)
            train_loss.append(loss)

            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e+1, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))

            # Compute validation loss at every 25 iterations
            if (iteration % 25 == 0):

                # Initiate for validation set
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))

                val_acc_ = []
                val_loss_ = []
                for x_v, y_v in sample_batch(X_val, y_val, batch_size):
                    # Feed
                    feed = {inputs: x_v, target: y_v, keep_prob: 1.0, initial_state: val_state}

                    # Loss
                    loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict=feed)

                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)

                # Print info
                print("Epoch: {}/{}".format(e+1, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))

            # Iterate
            iteration += 1

    saver.save(sess, "checkpoints/lstm.ckpt")

"""
references:
- https://burakhimmetoglu.com/2017/08/22/time-series-classification-with-tensorflow/
- https://github.com/healthDataScience/deep-learning-HAR/blob/master/HAR-LSTM.ipynb
"""
