import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import sys
from aux import convert_string_to_ints, clean_string

stack_dimension = 2

class NameClassifier(object):

    def __init__(self, seq_length, vocab_size, memory_dim, batch_size):

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        self.enc_inp = [tf.placeholder(tf.float32, shape=(batch_size,vocab_size)) for t in range(seq_length)]
        single_cell = rnn_cell.LSTMCell(memory_dim, state_is_tuple=False)
        cell = rnn_cell.MultiRNNCell([single_cell]*stack_dimension, state_is_tuple=True)
        _, encoder_state = rnn.rnn(cell, self.enc_inp, dtype=tf.float32)
        self.encoder_state= encoder_state

        # First Feedforward layer (fully connected)
        # The size of the encoder state for an LSTM is 2*memory_dim
        nodes_size = 2*memory_dim * stack_dimension
        W1 = tf.Variable(tf.random_uniform([nodes_size, nodes_size],-0.1,0.1))
        b1 = tf.Variable(tf.random_uniform([nodes_size],-0.1,0.1))
        state = tf.reshape(encoder_state, [1,nodes_size])
        h1 = tf.nn.sigmoid(tf.matmul(state, W1) + b1)

        # Second Feedforward layer (fully connected)
        W2 = tf.Variable(tf.random_uniform([nodes_size, 2*nodes_size],-0.1,0.1))
        b2 = tf.Variable(tf.random_uniform([2*nodes_size],-0.1,0.1))        
        h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)

        # Softmax layer. Only two states for the output (true/false)
        Ws = tf.Variable(tf.zeros([2*nodes_size, 2]))
        bs = tf.Variable(tf.zeros([2]))
        self.y = tf.nn.softmax(tf.matmul(h2, Ws) + bs)

        # Loss function and training
        self.y_ = tf.placeholder(tf.float32, [batch_size, 2])
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        # Clipping the gradient
        optimizer = tf.train.AdamOptimizer(1e-5)
        gvs = optimizer.compute_gradients(self.cross_entropy)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_step = optimizer.apply_gradients(capped_gvs)

        self.sess.run(tf.initialize_all_variables())


    def train(self, data):    

        for epoch in range(2000):
            epoch_losses = []
            for i in range(len(data)):
                X = data[i][0]
                y = data[i][1]
                inp = [X[self.seq_length-1-t] for t in range(self.seq_length)]
                out = y
                feed_dict = {self.enc_inp[t]: [inp[t]] for t in range(self.seq_length)}
                feed_dict.update( {self.y_: out} )
                loss, _ = self.sess.run([self.cross_entropy, self.train_step], feed_dict)
                print loss
                epoch_losses.append(loss)
            print max(epoch_losses)
            sys.stdout.flush()
            if max(epoch_losses) < 1e-1:
                return

            
# Functions related to classification
            
    def __predict(self, X):        
        X = [X[self.seq_length-1-t] for t in range(self.seq_length)]
        feed_dict = {self.enc_inp[t]: [X[t]] for t in range(self.seq_length)}
        y_batch = self.sess.run(self.y, feed_dict)
        return y_batch

    def classify(self, string):
        if len(string) > self.seq_length:
            return False
        line = clean_string(string)
        X = convert_string_to_ints(line, self.vocab_size, self.seq_length)
        prediction = self.__predict(X)[0]
        return prediction[0] > prediction[1]


    
# Loading and saving functions

    def save (self, filename):
        saver = tf.train.Saver()
        saver.save (self.sess, filename)

    def __load_tensorflow (self, filename):
        saver = tf.train.Saver()
        saver.restore (self.sess, filename)

    @classmethod
    def load(self, filename, seq_length = 128):
        vocab_size = 256        
        model = NameClassifier(seq_length = seq_length, vocab_size = vocab_size, memory_dim = 512, batch_size = 1)
        model.__load_tensorflow(filename)
        return model

