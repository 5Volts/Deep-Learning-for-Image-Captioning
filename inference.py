'''
This is the main code for going from a jpg image to English captions.
Your image will be fed into a pre-trained image classifier. The output
of the probability vectors are then fed into the start of an LSTM recurrent
neural network that will generate the captions.
'''
import cv2
import argparse
import numpy as np
from sklearn.externals import joblib

from data import data_loader
from my_class import ImageCaptioningTest
from image_embedding.eval import Transferred_Learning

def print_sentence(li):
  '''
  This function is for printing out the list of words
  in a more proper fahsion.

  :param li: List of words from neural network.
  Ex: ['I', 'love', 'to', 'code']

  :return: None
  '''
  for word_idx in range(len(li)):
    if word_idx == 0:
      word = li[word_idx].capitalize()
    else:
      word = li[word_idx]
    print(word, end=' ')
  print('.')

def main(args):
  '''
  This would be our main code.
  :param args: Dictionary of Arguments from parser
  :return: None
  '''
  dict_param = vars(args)
  longest_sentence = 51
  TL = Transferred_Learning()
  image_file = dict_param['image_file']
  model = dict_param['restore_from']
  pca = joblib.load('reduction/pca.pkl')
  '''
  Warning! The vocabulary is not zero-indexed as 0 is used to 
  pad the end of each sentence so we need to add 1 to our vocabulary size.
  <<start>> is index 11479 (second last) in the vocabulary and since the embeddings
  is zero indexed. We will need to deduct 1 from the index when we use
  it to initialize inference
  '''
  n_words = data_loader.vocab_size + 1

  image_captioning_test = ImageCaptioningTest(n_word=n_words, max_length=longest_sentence, save_path=model)

  image_captioning_test.init_inference()

  img = cv2.imread(image_file)
  image_vector = TL.get_result(img)
  reduced = pca.transform(image_vector)
  reduced_reshaped = reduced[0]
  sentence = image_captioning_test.inference([reduced_reshaped],return_english=True)
  print_sentence(sentence)

if __name__ == '__main__':
  parser = argparse.ArgumentParser('Image_Captioning_Project')

  parser.add_argument('--image_file',
                      type=str,
                      default='personal_test_images/pup.png')

  parser.add_argument('--restore_from',
                      type=str,
                      default='model/ImageCap')

  parameters = parser.parse_args()
  main(parameters)