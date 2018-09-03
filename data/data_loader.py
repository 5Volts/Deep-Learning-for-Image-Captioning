import json

vocab = open('data/bag_of_word_small.json')
vocabulary = json.load(vocab)
vocab_size = len(vocabulary)

reverse = open('data/bag_of_word_reverse.json')
idx2word = json.load(reverse)
