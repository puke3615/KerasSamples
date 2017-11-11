import data
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.seq2seq import embedding_attention_seq2seq


inputs, outputs = data.get_data()
print inputs, outputs



n_hidden = 10
cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
state = embedding_attention_seq2seq()
