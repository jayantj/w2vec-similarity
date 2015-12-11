import sys, os
from nltk.corpus import gutenberg
from gensim.models import Word2Vec

for corpus_name in sys.argv[1:]:
  corpus_sents = gutenberg.sents(corpus_name)
  print "Loaded sentences for corpus " + corpus_name
  model = Word2Vec(corpus_sents)
  print "Trained model for corpus " + corpus_name
  model.save('models/' + os.path.splitext(corpus_name)[0] + '-model') 

