import numpy as np
from gensim.models import Word2Vec
import random, sys, pdb, cProfile, time

def model_similarity(m1, m2):
  m1.init_sims()
  m2.init_sims()
  num_chosen_words = 20000
  topn = 10
  m1_embedding_size = m1[m1.vocab.keys()[0]].shape[0]
  m2_embedding_size = m2[m2.vocab.keys()[0]].shape[0]
  m1_words = m1.vocab.keys()
  m2_words = m2.vocab.keys()
  chosen_words = []
  for i in range(num_chosen_words/2):
    chosen_words.append(m1_words[random.randint(0, len(m1_words)-1)])
    chosen_words.append(m2_words[random.randint(0, len(m2_words)-1)])
  chosen_words_embeddings_m1 = []
  chosen_words_embeddings_m2 = []
  for chosen_word in chosen_words:
    if chosen_word in m1.vocab:
      chosen_words_embeddings_m1.append(m1.syn0norm[m1.vocab[chosen_word].index])
    else:
      chosen_words_embeddings_m1.append(np.zeros(m1_embedding_size))
    if chosen_word in m2.vocab:
      chosen_words_embeddings_m2.append(m2.syn0norm[m2.vocab[chosen_word].index])
    else:
      chosen_words_embeddings_m2.append(np.zeros(m2_embedding_size))
  chosen_words_embeddings_m1 = np.array(chosen_words_embeddings_m1)
  chosen_words_embeddings_m2 = np.array(chosen_words_embeddings_m2)

  now = time.time()
  similar_scores_m1 = np.matmul(chosen_words_embeddings_m1, m1.syn0norm.T)
  similar_scores_m2 = np.matmul(chosen_words_embeddings_m2, m2.syn0norm.T)

  top_similar_scores_m1 = [[0 for i in range(2*topn)] for i in range(num_chosen_words)]
  top_similar_scores_m2 = [[0 for i in range(2*topn)] for i in range(num_chosen_words)]
  for i in range(similar_scores_m1.shape[0]):
    top_similar_indices_m1 = np.argpartition(similar_scores_m1[i], -topn)[-topn-1:]
    top_similar_indices_m2 = np.argpartition(similar_scores_m2[i], -topn)[-topn-1:]

    max_m1_index = similar_scores_m1[i][top_similar_indices_m1].argmax()
    max_m2_index = similar_scores_m2[i][top_similar_indices_m2].argmax()
    top_similar_indices_m1 = np.delete(top_similar_indices_m1, max_m1_index)
    top_similar_indices_m2 = np.delete(top_similar_indices_m2, max_m2_index)

    top_similar_scores_m1[i][0:topn] = similar_scores_m1[i][top_similar_indices_m1]
    top_similar_scores_m2[i][topn:] = similar_scores_m2[i][top_similar_indices_m2]
    word_indices_m1 = {}

    current_word = chosen_words[i]
    if current_word not in m1.vocab or current_word not in m2.vocab:
      continue
    for j in range(top_similar_indices_m1.shape[0]):
      similar_m1_word = m1.index2word[top_similar_indices_m1[j]]
      word_indices_m1[similar_m1_word] = j
      if similar_m1_word in m2.vocab:
        top_similar_scores_m2[i][j] = similar_scores_m2[i][m2.vocab[similar_m1_word].index]
    for j in range(top_similar_indices_m2.shape[0]):
      similar_m2_word = m2.index2word[top_similar_indices_m2[j]]
      if similar_m2_word in word_indices_m1:
        top_similar_scores_m2[i][j+topn] = 0.0
      elif similar_m2_word in m1.vocab:
        top_similar_scores_m1[i][j+topn] = similar_scores_m1[i][m1.vocab[similar_m2_word].index]
  top_similar_scores_m1 = np.array(top_similar_scores_m1)
  top_similar_scores_m2 = np.array(top_similar_scores_m2)
  similarity_sum = abs(top_similar_scores_m1 - top_similar_scores_m2).sum()
  return similarity_sum/(2*topn*num_chosen_words)

if len(sys.argv) > 1:
  m1 = Word2Vec.load(sys.argv[1])
  m2 = Word2Vec.load(sys.argv[2])
  cProfile.run('print(model_similarity(m1, m2))')
else:
  m1 = Word2Vec.load('models/text8-model-epochs-1-size-100')
  for epochs in range(1, 10):
    for size in range(100,500,100):
      m2 = Word2Vec.load('models/text8-model-epochs-%d-size-%d' % (epochs,size))
      print "Model similarity for model of size - %d and epochs - %d" % (size, epochs)
      print model_similarity(m1, m2)