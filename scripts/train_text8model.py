import sys, os, time, glob, pdb
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm


def tokenize_sentences(text):
  import nltk.data
  sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  return sent_tokenizer.tokenize(text)

def split_sentences(text):
  avg_sent_length = 26
  split_text = np.array(text.split())

  remainder = len(split_text)%avg_sent_length
  sents = split_text[:-remainder].reshape((-1, avg_sent_length)).tolist()
  remainder_sents = split_text[-remainder:]
  sents.append(remainder_sents.tolist())
  return sents
  

def get_words(sents = []):
  from nltk.tokenize import wordpunct_tokenize
  words = []
  for sent in sents:
    words.append(wordpunct_tokenize(sent))
  return words

# file_name = sys.argv[1]
for file_name in ['data/text8']:#glob.glob('/Users/jayant/Downloads/Books/gutenberg/txt/*.txt'):
  sent_length = 22
  with open(file_name,'U') as f:
    sents = split_sentences(f.read())

  print "Loaded words for file " + file_name
  for epochs in [1]:
    for model_size in [100]:
      # if epochs is 1 and model_size is 100:
        # next
      model = Word2Vec(sents, iter=epochs, size=model_size)
      print "Trained model for file " + file_name + "epochs: " + str(epochs) + " model size: " + str(model_size)
      model.save('models/' + os.path.splitext(os.path.basename(file_name))[0] + '-preprocessed-model-epochs-'+str(epochs)+"-size-"+str(model_size)) 

  # if batch_train: 
  #   model = Word2Vec(sents)
  #   print "Trained model for corpus in batch"
  #   model.save('models/batch-model-' + str(time.time())) 