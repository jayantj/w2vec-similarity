import sys, os, time
from nltk.corpus import brown
from gensim.models import Word2Vec

# if sys.argv[-1] == "batch_train":
#   inputs = sys.argv[1:-1]
#   batch_train = True
# else:
#   inputs = sys.argv[1:]

# sents = []
# for corpus_name in inputs:
#   corpus_sents = gutenberg.sents(corpus_name)
#   print "Loaded sentences for corpus " + corpus_name
#   if batch_train:
#     print corpus_name, len(corpus_sents)
#     sents += list(corpus_sents)
#   else:
#     model = Word2Vec(corpus_sents)
#     print "Trained model for corpus " + corpus_name
#     model.save('models/' + os.path.splitext(corpus_name)[0] + '-model') 

# print len(sents)
# if batch_train: 
#   model = Word2Vec(sents)
#   print "Trained model for corpus in batch"
#   model.save('models/batch-model-' + str(time.time())) 

sents = brown.sents(categories = brown.categories())
print "Training on brown corpus, number of sentences: ", len(sents)

model = Word2Vec(sents)
model.save("models/brown-complete-model")