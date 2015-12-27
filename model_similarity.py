from gensim.models import Word2Vec
from tqdm import tqdm
import itertools, random, pdb, sys

def model_similarity_word(given_word, given_model, target_model, topn = 10):
  divergence = 0.0
  if not given_word in target_model:
    divergence += 2 * topn
  else:
    given_most_similar_dict = {obj[0]:obj[1] for obj in given_model.most_similar(given_word, topn=topn)}
    target_most_similar_dict = {obj[0]:obj[1] for obj in target_model.most_similar(given_word, topn=topn)}
    for word, similarity in given_most_similar_dict.iteritems():
      if word in target_most_similar_dict:
        divergence += abs(similarity - target_most_similar_dict[word])
        del target_most_similar_dict[word]
      elif word in target_model.vocab:
        divergence += abs(similarity - target_model.similarity(given_word, word))
      else:
        divergence += similarity
    for word, similarity in target_most_similar_dict.iteritems():
      if word in given_model.vocab:
        divergence += abs(similarity - given_model.similarity(given_word, word))
      else:
        divergence += similarity
  return divergence/(2 * topn)

def model_similarity(m1, m2):
  divergence = 0.0
  n = 100
  topn = 10
  considered_words = {}
  combined_vocab = {}
  for w in m1.vocab:
    combined_vocab[w] = 1
  for w in m2.vocab:
    combined_vocab[w] = 1
  for m1_word in tqdm(itertools.islice(m1.vocab, n)):
    considered_words[m1_word] = 1
    word_divergence = model_similarity_word(m1_word, m1, m2, topn)
    divergence += word_divergence
    if word_divergence > 10:
      print word_divergence, m1_word

  for m2_word in tqdm(itertools.islice(m2.vocab, n)):
    if len(considered_words) >= len(combined_vocab) - len(m2.vocab):
      break
    while m2_word in considered_words:
      m2_word = m2.vocab.keys()[random.randint(0, len(m2.vocab)-1)]
    considered_words[m2_word] = 1
    word_divergence = model_similarity_word(m2_word, m2, m1, topn)
    divergence += word_divergence
  divergence /= 2 * n
  return divergence


if len(sys.argv) > 1:
  m1 = Word2Vec.load(sys.argv[1])
  m2 = Word2Vec.load(sys.argv[2])
  print model_similarity(m1, m2)
else:
  m1 = Word2Vec.load('models/text8-model-epochs-1-size-100')
  for epochs in range(1, 10):
    for size in range(100,500,100):
      model = Word2Vec.load('models/text8-model-epochs-%d-size-%d' % (epochs,size))
      print "Model similarity for model of size - %d and epochs - %d" % (size, epochs)
      print model_similarity(m1, model)