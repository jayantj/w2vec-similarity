from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from sklearn.cluster import AffinityPropagation, SpectralClustering
from sklearn.manifold import TSNE, SpectralEmbedding
from codecs import open
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import random, sys, pdb, cProfile, time, json, os
import mpld3
import math

SIMILARITY_DIR = 'models/similarity_matrices/new/'
def model_similarity(m1, m2, num_chosen_words=100, topn=10):
  m1.init_sims()
  m2.init_sims()
  num_chosen_words = 2000
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
  diff_sum = 0.0
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
    common_words = 0
    if current_word not in m1.vocab or current_word not in m2.vocab:
      diff_sum += (abs(np.array(top_similar_scores_m1[i]) - np.array(top_similar_scores_m2[i])).sum())/topn
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
        common_words += 1
      elif similar_m2_word in m1.vocab:
        top_similar_scores_m1[i][j+topn] = similar_scores_m1[i][m1.vocab[similar_m2_word].index]
    diff_sum += (abs(np.array(top_similar_scores_m1[i]) - np.array(top_similar_scores_m2[i])).sum())/(2*topn-common_words)
  top_similar_scores_m1 = np.array(top_similar_scores_m1)
  top_similar_scores_m2 = np.array(top_similar_scores_m2)
  similarity_sum = abs(top_similar_scores_m1 - top_similar_scores_m2).sum()
  # return (similarity_sum/(2*topn*num_chosen_words), diff_sum/num_chosen_words)
  return diff_sum/num_chosen_words

def model_similarity_word(given_word, given_model, target_model, topn = 10):
  divergence = 0.0
  given_most_similar_dict = {obj[0]:obj[1] for obj in given_model.most_similar(given_word, topn=topn)}
  if not given_word in target_model:
    # target_most_similar_dict = {obj[0]:obj[1] for obj in target_model.most_similar(target_model.index2word[0], topn=topn)}
    divergence += sum(given_most_similar_dict.values())/topn
  else:
    target_most_similar_dict = {obj[0]:obj[1] for obj in target_model.most_similar(given_word, topn=topn)}
    common_words = 0
    for word, similarity in given_most_similar_dict.iteritems():
      if word in target_most_similar_dict:
        divergence += abs(similarity - target_most_similar_dict[word])
        common_words += 1
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
    divergence /= 2 * topn - common_words
  return divergence

def model_similarity_basic(m1, m2, num_chosen_words=100, topn=10):
  divergence = 0.0
  num_chosen_words = 2000
  topn = 10
  considered_words = {}
  combined_vocab = {}
  for w in m1.vocab:
    combined_vocab[w] = 1
  for w in m2.vocab:
    combined_vocab[w] = 1
  for i in range(num_chosen_words/2):
    m1_word = m1.index2word[random.randint(0,len(m1.vocab)-1)]
    word_divergence = model_similarity_word(m1_word, m1, m2, topn)
    divergence += word_divergence

    m2_word = m2.index2word[random.randint(0,len(m2.vocab)-1)]
    word_divergence = model_similarity_word(m2_word, m2, m1, topn)
    divergence += word_divergence
  divergence /= num_chosen_words
  return divergence

def save_similarity_matrix(model_files, output_file=''):
  models = []
  for model_file in model_files:
    models.append(Word2Vec.load(model_file))
  similarity_matrix = [[None for i in range(len(models))] for i in range(len(models))]
  for i in range(len(models)):
    for j in range(len(models)):
      print "evaluating similarity_matrix indices %d,%d" % (i, j)
      if i is j:
        similarity_matrix[i][j] = 1.0
      elif similarity_matrix[j][i]:
        similarity_matrix[i][j] = similarity_matrix[j][i]
      else:
        print "evaluating similarities for models %s, %s" % (model_files[i], model_files[j])
        similarity_matrix[i][j] = 1 - model_similarity(models[i], models[j])
  output_file = output_file or (SIMILARITY_DIR + '/similarities-%d.json' % time.time())
  print output_file
  with open(output_file, 'w', 'utf-8') as f:
    json.dump({"model_names":model_files, "similarity_matrix":similarity_matrix}, f)

def affinity_propagation_clusters(similarity_matrix):
  return AffinityPropagation(affinity='precomputed').fit(similarity_matrix)

def spectral_clustering_clusters(similarity_matrix):
  return SpectralClustering(n_clusters=10, affinity='precomputed').fit(similarity_matrix)


def scale_sim_scores(similarity_matrix):
  max_sim_score = 0.0
  min_sim_score = 1.0
  for row in similarity_matrix:
    for score in row:
      max_sim_score = score if (score > max_sim_score and score != 1.0) else max_sim_score
      min_sim_score = score if (score < min_sim_score) else min_sim_score

  for i, row in enumerate(similarity_matrix):
    for j, score in enumerate(row):
      similarity_matrix[i][j] = 0.1 + 0.8 * (score-min_sim_score)/(max_sim_score-min_sim_score) if score != 1.0 else score
  return similarity_matrix

def evaluate_clusters(model_files = [], similarity_file = ''):
  if len(model_files):
    similarity_file = (SIMILARITY_DIR + '/similarities-%d.json' % time.time())
    print similarity_file
    save_similarity_matrix(model_files, similarity_file)
  with open(similarity_file, 'r', 'utf-8') as f:
    similarity_data = json.load(f)
    cluster_data = spectral_clustering_clusters(scale_sim_scores(similarity_data['similarity_matrix']))
  model_names = similarity_data['model_names']
  doc2cluster = {}
  cluster2doc = {}
  for i in range(len(model_names)):
    model_name = os.path.splitext(os.path.basename(model_names[i]))[0]
    label = str(cluster_data.labels_[i])
    doc2cluster[model_name] = label
    if label in cluster2doc:
      cluster2doc[label].append(model_name)
    else:
      cluster2doc[label] = [model_name]
  similarity_data['cluster2doc'] = cluster2doc
  similarity_data['doc2cluster'] = doc2cluster

  current_time = time.time()
  similarity_file_name = os.path.splitext(similarity_file)[0]
  with open(similarity_file_name + '-%d.json' % current_time, 'w', 'utf-8') as f:
    json.dump(similarity_data, f)

  with open(similarity_file_name + '-readable-%d' % current_time, 'w', 'utf-8') as f:
    f.write("cluster document mapping\n")
    for key, value in cluster2doc.iteritems():
      f.write(str(key) + "\n\t" + "\n\t".join(sorted(value))+"\n\n")
    f.write("\ndocument cluster mapping\n")
    for key in sorted(doc2cluster.keys()):
      f.write(str(doc2cluster[key]).ljust(3) + "\t" + key + "\n")
    f.write("\nsimilarity scores\n")
    for i, row in enumerate(similarity_data['similarity_matrix']):
      f.write("\n\n"+os.path.basename(model_names[i]))
      score_strings = {}
      sorted_indices = [k[0] for k in sorted(enumerate(row), key=lambda x:-x[1])]
      for index in sorted_indices:
        f.write("\n\t" + str(round(row[index], 3)).ljust(5) + "\t" + os.path.basename(model_names[index]))
  return similarity_file_name + '-%d.json' % current_time

def evaluate_doc2vec_similarities(doc2vec_file):
  model = Doc2Vec.load(doc2vec_file)
  model.docvecs.init_sims(True)
  similarity_matrix = [[None for i in range(len(model.docvecs))] for i in range(len(model.docvecs))]
  # doc_names = [unicode(tag, 'utf-8') for tag in model.docvecs.doctags.keys()]
  doc_names = model.docvecs.doctags.keys()
  for i, doc1 in enumerate(doc_names):
    for j, doc2 in enumerate(doc_names):
      if i is j:
        similarity_matrix[i][j] = 1.0
      elif similarity_matrix[j][i]:
        similarity_matrix[i][j] = similarity_matrix[j][i]
      else:
        doc1_vec = model.docvecs[doc1]
        doc2_vec = model.docvecs[doc2]
        similarity_matrix[i][j] = round((doc1_vec * doc2_vec).sum(),3)
  with open(SIMILARITY_DIR + '/similarity-' + os.path.basename(doc2vec_file), 'w', 'utf-8') as f:
    json.dump({"model_names":doc_names, "similarity_matrix":similarity_matrix}, f)

def evaluate_doc2vec_clusters(doc2vec_file):
  evaluate_doc2vec_similarities(doc2vec_file)
  evaluate_clusters(similarity_file=SIMILARITY_DIR + '/similarity-' + os.path.basename(doc2vec_file))

def cluster_scatter_plot(similarity_file):
  def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

  with open(similarity_file, 'r', 'utf-8') as f:
    similarity_data = json.load(f)
  labels = []
  point_colors = []
  num_clusters = len(similarity_data['cluster2doc'].keys())
  cmap = get_cmap(num_clusters)
  for model_name in similarity_data['model_names']:
    model_name = os.path.splitext(os.path.basename(model_name))[0]
    cluster_label = similarity_data['doc2cluster'][model_name]
    point_colors.append(cmap(cluster_label))
    labels.append(model_name)#"%s-%s"%(cluster_label, ''))#model_name.replace('gutenberg-', '')))
  embeddings = SpectralEmbedding(affinity='precomputed').fit_transform(np.array(similarity_data['similarity_matrix']))
  fig, ax = plt.subplots()
  x = embeddings[:, 0]
  y = embeddings[:, 1]
  annotes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] * 10
  N = 100
  scatter = ax.scatter(x, y, c=point_colors[:],s=100*np.ones(shape=N))
  tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
  mpld3.plugins.connect(fig, tooltip)
  mpld3.show()
  # plt.scatter(tsne_embeddings[20:40, 0], tsne_embeddings[20:40, 1], c='b')
  # for label, x, y in zip(labels, tsne_embeddings[:, 0], tsne_embeddings[:, 1]):
  #   plt.annotate(
  #       label, 
  #       xy = (x, y),
  #       # textcoords = 'offset points',
  #       bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
  # plt.show()