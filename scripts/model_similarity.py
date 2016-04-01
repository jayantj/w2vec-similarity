from gensim.models import Word2Vec
from tqdm import tqdm
import itertools, random, pdb, sys, cProfile, random

def model_similarity_word(given_word, given_model, target_model, topn = 10):
  divergence = 0.0
  given_most_similar_dict = {obj[0]:obj[1] for obj in given_model.most_similar(given_word, topn=topn)}
  if not given_word in target_model:
    target_most_similar_dict = {obj[0]:obj[1] for obj in target_model.most_similar(target_model.index2word[0], topn=topn)}
    divergence += sum(given_most_similar_dict.values())
  else:
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
  n = 20000
  topn = 10
  considered_words = {}
  combined_vocab = {}
  for w in m1.vocab:
    combined_vocab[w] = 1
  for w in m2.vocab:
    combined_vocab[w] = 1
  for i in range(n/2):
    m1_word = m1.index2word[random.randint(0,len(m1.vocab)-1)]
    word_divergence = model_similarity_word(m1_word, m1, m2, topn)
    divergence += word_divergence

    m2_word = m2.index2word[random.randint(0,len(m2.vocab)-1)]
    word_divergence = model_similarity_word(m2_word, m2, m1, topn)
    divergence += word_divergence
  divergence /= n
  return divergence

if len(sys.argv) > 1:
  m1 = Word2Vec.load(sys.argv[1])
  m2 = Word2Vec.load(sys.argv[2])
  method_name = sys.argv[3]
  iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 10
  for i in range(iterations):
    cProfile.run('print evaluate.%s(m1, m2)' % method_name, '../plots/stats-%d.stats' % i)
  stats = pstats.Stats('../plots/stats-0.stats')
  for i in range(1, iterations):
    stats.add('../plots/stats-%d.stats' % i)
  # Clean up filenames for the report
  stats.strip_dirs()
  # Sort the statistics by the cumulative time spent in the function
  stats.sort_stats('tottime')
  stats.print_stats(0.5)
  pdb.set_trace()
else:
  m1 = Word2Vec.load('models/text8-model-epochs-1-size-100')
  for epochs in range(1, 10):
    for size in range(100,500,100):
      model = Word2Vec.load('models/text8-model-epochs-%d-size-%d' % (epochs,size))
      print "Model similarity for model of size - %d and epochs - %d" % (size, epochs)
      print model_similarity(m1, model)