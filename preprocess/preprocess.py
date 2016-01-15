from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer
from train import train
import nltk.data

def split_by_char(str, delimiting_char=' '):
  return str.split(delimiting_char)

def reshape_into_sents(text_arr, avg_sent_length = 26):
  sents = []
  for i in range(len(text_arr)/avg_sent_length):
    sents.append(text_arr[i:i+avg_sent_length])
  sents.append(text_arr[i:])
  return sents

def tokenize_into_sentences(text):
  sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  return sent_tokenizer.tokenize(text)

def tokenize_into_words(sents = []):
  words = []
  for sent in sents:
    words.append(wordpunct_tokenize(sent))
  return words

def lowercase_sents(sents = []):
  lower_sents = []
  for sent in sents:
    lower_sents.append(' '.join(sent).lower().split(' '))
  return lower_sents

def remove_punctuation_sents(sents = []):
  processed_sents = []
  for sent in sents:
    processed_sents.append(remove_punctuation(' '.join(sent)))
  return processed_sents

def remove_punctuation(str):
  tokenizer = RegexpTokenizer(r'\w+')
  return tokenizer.tokenize(str)