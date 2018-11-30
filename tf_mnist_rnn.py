import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph() # in the begining of the code to avoid namespace error

# to be able to compare methods it should be set after tf.reset_default_graph()
tf.set_random_seed(1)
np.random.seed(1)

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/meysam/github/ML-with-TensorFlow/Datasets/mnist", one_hot=True)


# Load data
X_train = mnist.train.images.reshape([-1,28,28])
Y_train = mnist.train.labels
X_validation = mnist.validation.images.reshape([-1,28,28])
Y_validation = mnist.validation.labels
X_test = mnist.test.images.reshape([-1,28,28])
Y_test = mnist.test.labels

# see how data looks like
print('Check the dimenssions')
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_validation.shape:', X_validation.shape)
print('X_validation.shape:', X_validation.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)


# set initial parameters
num_hidden = 64
num_steps = 28  # as we want to read the images column by column
num_input = 28 # as one column of image is 28 pixels
num_classes = 10
num_epochs = 1
batch_size = 32
num_batches = int(len(Y_train) / batch_size)


# PLaceholders
X = tf.placeholder(tf.float32, [None, num_steps, num_input])
Y = tf.placeholder(tf.float32,[None,num_classes])


# Create the RNN
def RNN_model(x,num_classes): 
    # Define an lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = num_hidden,state_is_tuple=True) #lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    #outputs: is the RNN output tensor. If time_major == False (default), this will be a Tensor shaped: [batch_size, max_time, cell.output_size] = [batch_size, num_time_steps, num_hidden_units]
    #last_states:is the final state of RNN.  cell.state_size is an int, this will be shaped [batch_size, cell.state_size]. If it is a TensorShape, this will be shaped [batch_size] + cell.state_size
    
    outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell,  # an instance of RNN cell
                                             inputs=x,        # The RNN inputs. If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...], or a nested tuple of such elements
                                             dtype=tf.float32 # It is NOT optional, if we do not provide 
                                             # sequence_length = sequence_length # this one is optional (read the note above on sequence_length). When all our input data points have the same number of time steps
                                             # time_major = False # It is optional. time_major determines the shape format of the inputs and outputs Tensors. If true, these Tensors must be shaped [max_time, batch_size, depth]. If false, these Tensors must be shaped [batch_size, max_time, depth].
                                            )
  
    # If you do not need batch normalization, comment next line and change the return
    batch_normzd = tf.layers.batch_normalization(outputs[:, -1, :])
    
    y_hat = tf.layers.dense(batch_normzd,num_classes, activation=None, kernel_initializer=tf.orthogonal_initializer())
    return y_hat
    






logits = RNN_model(X,num_classes)
predictions = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(cost)

accuracy = tf.reduce_mean( (tf.cast(tf.equal(tf.argmax(predictions,1), tf.argmax(Y,1)),dtype=tf.float32)) )

#saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch_cntr in range(num_epochs):
        for batch_cntr in range(num_batches):
            x_train_batch, y_train_batch = mnist.train.next_batch(batch_size)
            x_train_batch = x_train_batch.reshape([-1,28,28])
            sess.run(optimizer, feed_dict={X:x_train_batch, Y:y_train_batch})
            batch_train_cost,batch_train_acc= sess.run([cost,accuracy], feed_dict={X:x_train_batch, Y:y_train_batch})
            
            if batch_cntr % 250 == 0:
                x_test_batch, y_test_batch = mnist.test.next_batch(batch_size)
                x_test_batch = x_test_batch.reshape([-1,28,28])
                batch_test_cost,batch_test_acc= sess.run([cost,accuracy], feed_dict={X:x_test_batch, Y:y_test_batch})
                
                print('\n Train Acc:{}   Test Acc:{}    Train Cost:{}   Test Cost:{}'.format(batch_train_acc, batch_test_acc, batch_train_cost, batch_test_cost))
            











































