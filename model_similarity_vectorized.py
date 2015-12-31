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
  all_words = m1.vocab.keys() + m2.vocab.keys()
  chosen_words = []
  for i in range(num_chosen_words):
    chosen_words.append(all_words[random.randint(0, len(all_words)-1)])
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

  similar_scores_m1 = np.matmul(chosen_words_embeddings_m1, m1.syn0norm.T)
  similar_scores_m2 = np.matmul(chosen_words_embeddings_m2, m2.syn0norm.T)

  top_similar_indices_m1 = np.argpartition(similar_scores_m1, topn)[:, -topn-1:]
  top_similar_indices_m2 = np.argpartition(similar_scores_m2, topn)[:, -topn-1:]

  temp_similar_indices_m1 = []
  temp_similar_indices_m2 = []
  for i in range(top_similar_indices_m1.shape[0]):
    max_m1_index = similar_scores_m1[i][top_similar_indices_m1[i]].argmax()
    max_m2_index = similar_scores_m2[i][top_similar_indices_m2[i]].argmax()
    temp_similar_indices_m1.append(np.delete(top_similar_indices_m1[i], max_m1_index).tolist())
    temp_similar_indices_m2.append(np.delete(top_similar_indices_m2[i], max_m2_index).tolist())

  top_similar_indices_m1 = np.array(temp_similar_indices_m1)
  top_similar_indices_m2 = np.array(temp_similar_indices_m2)
  del temp_similar_indices_m1
  del temp_similar_indices_m2

  top_similar_scores_m1 = [[0 for i in range(2*topn)] for i in range(num_chosen_words)]
  top_similar_scores_m2 = [[0 for i in range(2*topn)] for i in range(num_chosen_words)]

  for i in range(top_similar_indices_m1.shape[0]):
    top_similar_scores_m1[i][0:topn] = similar_scores_m1[i][top_similar_indices_m1[i]]
    top_similar_scores_m2[i][topn:] = similar_scores_m2[i][top_similar_indices_m2[i]]
    word_indices_m1 = {}
    current_word = chosen_words[i]
    if current_word not in m1.vocab or current_word not in m2.vocab:
      continue
    for j in range(top_similar_indices_m1.shape[1]):
      similar_m1_word = m1.index2word[top_similar_indices_m1[i][j]]
      word_indices_m1[similar_m1_word] = j
      if similar_m1_word in m2.vocab:
        top_similar_scores_m2[i][j] = similar_scores_m2[i][m2.vocab[similar_m1_word].index]
    for j in range(top_similar_indices_m2.shape[1]):
      similar_m2_word = m2.index2word[top_similar_indices_m2[i][j]]
      if similar_m2_word in word_indices_m1:
        top_similar_scores_m2[i][j+topn] = 0.0
      elif similar_m2_word in m1.vocab:
        top_similar_scores_m1[i][j+topn] = similar_scores_m1[i][m1.vocab[similar_m2_word].index]
        
  top_similar_scores_m1 = np.array(top_similar_scores_m1)
  top_similar_scores_m2 = np.array(top_similar_scores_m2)
  similarity_sum = abs(top_similar_scores_m1 - top_similar_scores_m2).sum()
  print similarity_sum/(2*topn*num_chosen_words)
  pdb.set_trace()

if len(sys.argv) > 1:
  m1 = Word2Vec.load(sys.argv[1])
  m2 = Word2Vec.load(sys.argv[2])
  print cProfile.run('model_similarity(m1, m2)')
else:
  m1 = Word2Vec.load('models/text8-model-epochs-1-size-100')
  for epochs in range(1, 10):
    for size in range(100,500,100):
      m2 = Word2Vec.load('models/text8-model-epochs-%d-size-%d' % (epochs,size))
      print "Model similarity for model of size - %d and epochs - %d" % (size, epochs)
      print model_similarity(m1, m2)