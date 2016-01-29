from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random, sys, pdb, cProfile, time, json, os

def model_similarity(m1, m2, num_chosen_words=100  , topn=10):
  m1.init_sims()
  m2.init_sims()
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
  output_file = output_file or 'models/similarity_matrices/similarities-%d.json' % time.time()
  print output_file
  with open(output_file, 'w') as f:
    json.dump({"model_names":model_files, "similarity_matrix":similarity_matrix}, f)
  # return output_file

def affinity_propagation_clusters(similarity_matrix, options = {}):
  return AffinityPropagation(affinity='precomputed').fit(similarity_matrix)

def evaluate_clusters(model_files = [], similarity_file = ''):
  if len(model_files):
    similarity_file = 'models/similarity_matrices/similarities-%d.json' % time.time()
    print similarity_file
    pdb.set_trace()
    save_similarity_matrix(model_files, similarity_file)
  with open(similarity_file) as f:
    similarity_data = json.load(f)
    afp = affinity_propagation_clusters(similarity_data['similarity_matrix'])
  model_names = similarity_data['model_names']
  doc2cluster = {}
  cluster2doc = {}
  for i in range(len(model_names)):
    model_name = os.path.splitext(os.path.basename(model_names[i]))[0]
    label = afp.labels_[i]
    doc2cluster[model_name] = label
    if label in cluster2doc:
      cluster2doc[label].append(model_name)
    else:
      cluster2doc[label] = [model_name]
  similarity_data['cluster2doc'] = cluster2doc
  similarity_data['doc2cluster'] = doc2cluster
  with open(similarity_file, 'w') as f:
    json.dump(similarity_data, f)
  with open(similarity_file[0:-5]+'-readable-%d' % time.time(), 'w') as f:
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

  pdb.set_trace()

def tsne_similarity_plot(similarity_file):
  with open(similarity_file) as f:
    similarity_data = json.load(f)
  labels = []
  for model_name in similarity_data['model_names']:
    model_name = os.path.splitext(os.path.basename(model_name))[0]
    cluster_label = similarity_data['doc2cluster'][model_name]
    labels.append("%s-%s"%(cluster_label, ''))#model_name.replace('gutenberg-', '')))
  tsne_embeddings = TSNE(metric='precomputed').fit_transform(np.array(similarity_data['similarity_matrix']))
  fig = plt.figure(figsize=(15,8))
  fig.add_subplot(111)
  plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
  for label, x, y in zip(labels, tsne_embeddings[:, 0], tsne_embeddings[:, 1]):
    plt.annotate(
        label, 
        xy = (x, y),
        # textcoords = 'offset points',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
  plt.show()
  # pdb.set_trace()

def plot_embedding(X, title=None):
  x_min, x_max = np.min(X, 0), np.max(X, 0)
  X = (X - x_min) / (x_max - x_min)

  plt.figure()
  ax = plt.subplot(111)
  for i in range(X.shape[0]):
    plt.text(X[i, 0], X[i, 1], str(digits.target[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

  if hasattr(offsetbox, 'AnnotationBbox'):
      # only print thumbnails with matplotlib > 1.0
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(digits.data.shape[0]):
      dist = np.sum((X[i] - shown_images) ** 2, 1)
      if np.min(dist) < 4e-3:
        # don't show points that are too close
        continue
      shown_images = np.r_[shown_images, [X[i]]]
      imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i])
      ax.add_artist(imagebox)
  plt.xticks([]), plt.yticks([])
  if title is not None:
    plt.title(title)