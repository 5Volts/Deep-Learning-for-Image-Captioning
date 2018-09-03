'''
Here lies the brain of our little caption generating AI.
'''

import tensorflow as tf
import numpy as np

from data.data_loader import vocabulary, idx2word
from data import data_loader


class ImageCaptioningTest:
  def __init__(self, n_word, max_length, batch=1, embed_dim=256, img_vect_size=256, dim_hidden=256, save_path='model/ImageCap'):
    '''
    Initializing our local variables for this class
    :param n_word: Number of words
    :param max_length: Longest sentence in dataset
    :param batch: Number of vectors to process at once
    :param embed_dim: Length of the vector embedding.
    :param img_vect_size: Length of the probability vector from image classifier.
    :param dim_hidden: Size of hidden layer
    :param save_path: Path to checkpoint
    '''
    self.batch_size = batch
    self.n_words = n_word
    self.max_length = max_length
    self.path = save_path
    self.img_vect_size = img_vect_size

    inference = tf.Graph()
    with inference.as_default():
      self.input_image_vector_infer = tf.placeholder(tf.float32, [None, self.img_vect_size])
      self.input_word_vector_infer = tf.placeholder(tf.int32, [None, self.max_length])

      # Hashing word index to embeddings
      self.word2embeddings_infer = tf.Variable(tf.random_uniform([self.n_words, embed_dim]))
      self.word2embeddings_bias_infer = tf.Variable(tf.random_uniform([embed_dim]))

      # Fully Connected Layer for Converting vectors from LSTM output to one hot encoding
      self.embeddings2words_infer = tf.Variable(tf.random_uniform([dim_hidden, self.n_words]))
      self.embeddings2words_bias_infer = tf.Variable(tf.random_uniform([self.n_words]))      

      self.single_lstm_cell_infer = tf.contrib.rnn.BasicLSTMCell(dim_hidden)
      self.out = self.inference_graph()

      self.saver_infer = tf.train.Saver()

      self.sess_eval = tf.Session()
      self.sess_eval.run(tf.global_variables_initializer())

  def inference_graph(self):
    '''
    This function will build the LSTM graph.
    :return: List of idx corresponding to the words in vocabulary
    Ex : [74,336,2934,2,465,324,12,864]
    '''
    hidden_state = self.single_lstm_cell_infer.zero_state(batch_size=self.batch_size, dtype=tf.float32)
    sentence = []
    with tf.variable_scope("RNN"):
      for cell in range(self.max_length):
        if cell == 0:
          curr_vect = self.input_image_vector_infer
          output, hidden_state = self.single_lstm_cell_infer(curr_vect, hidden_state)
        else:
          if cell == 1:
            '''
            See explanation in main code to understand why the index
             of <<start>> is vocabulary['<<start>>'] minus 1
            '''
            curr_vect = tf.nn.embedding_lookup(self.word2embeddings_infer,[vocabulary['<<start>>']-1]) + self.word2embeddings_bias_infer
            tf.get_variable_scope().reuse_variables()            
          else:
            tf.get_variable_scope().reuse_variables()
          output, hidden_state = self.single_lstm_cell_infer(curr_vect, hidden_state)
          word_one_hot = tf.nn.softmax(tf.add(tf.matmul(output, self.embeddings2words_infer), self.embeddings2words_bias_infer))
          word = tf.argmax(word_one_hot,1)
          curr_vect = tf.nn.embedding_lookup(self.word2embeddings_infer, word) + self.word2embeddings_bias_infer
          sentence.append(word)

    return sentence

  def inference(self,img_vector, return_english=False):
    '''
    Inference code that get called from outside
    :param img_vector: Probability vectors from Image Classifier
    Ex : [0.342,0.1,0.6,...,0.2.0.1]

    :param return_english: Set to True if you want it to return sentence of word
    Ex : True

    :return: Either a list of words or list of word indices
    Ex : ['I','love','to','make','personal','projects']
    or   [154, 543, 5432, 1234, 8765, 521]
    '''
    sentence = self.sess_eval.run([self.out], feed_dict={self.input_image_vector_infer:img_vector})
    sentence= np.array(sentence)
    sentence = np.reshape(sentence,(50,))
    if return_english:
      return word_idx2english(sentence)
    return sentence

  def init_inference(self):
    '''
    Restoring Graph with pretrained model
    :return: None
    '''
    if self.path[-1].isdigit():
      ckp = self.path
    else:
      # tf.train.latest_checkpoint only want the model directory, so we give it 'models/'
      ckp = tf.train.latest_checkpoint(self.path[:6])
    print('Loading from',ckp)
    self.saver_infer.restore(self.sess_eval, ckp)

def word_idx2english(word_idxs):
  '''
  Turn word index to english sentence
  :param word_idxs: List of word indices
  Ex : [154, 543, 5432, 1234, 8765, 521]

  :return: English sentence
  Ex: ['I','love','to','make','personal','projects']
  '''
  english = []
  for idx in word_idxs:
    if idx == 0:
      break
    english.append(idx2word[str(idx)])
  return english