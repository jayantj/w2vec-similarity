from gensim.models import Word2Vec
import importlib, pdb

MODELS_DIR = 'models/'

def train_and_save(sents, output_file, options = {}):
  print "Training model..."
  model = Word2Vec(sents, **options)
  model.save(output_file)

def load_corpus(corpus_name, package_name = 'nltk.corpus'):
  package = importlib.import_module(package_name)
  corpus = getattr(package, corpus_name)
  return corpus

def train_from_corpuses(corpus_names=[], output_file='', options={}):
  all_sents = []
  for corpus_name in corpus_names:
    corpus = load_corpus(corpus_name)
    all_sents += corpus.sents()
  output_file = "{0}{1}-{2}-model".format(MODELS_DIR, '-'.join(corpus_names), options_to_string(options)) or output_file
  train_and_save(all_sents, output_file, options)

def train_from_corpus_components(corpus_name, components_method_name='', component_names=[], output_file = '', separate = False, options={}):
  corpus = load_corpus(corpus_name)
  components_method = getattr(corpus, components_method_name)
  if component_names:
    component_names = set(component_names).intersection(components_method())
  else:
    component_names = set(components_method())

  if separate:
    for component_name in component_names:
      sents = corpus.sents(**{components_method_name:component_name})
      output_file = "{0}{1}-{2}-{3}-model".format(MODELS_DIR, corpus_name, component_name, options_to_string(options)) or output_file
      train_and_save(sents, output_file, options)
  else:
    all_sents = corpus.sents(**{components_method_name:component_names})
    output_file = "{0}{1}-{2}-{3}-model".format(MODELS_DIR, corpus_name, '-'.join(component_names), options_to_string(options)) or output_file
    train_and_save(all_sents, output_file, options)

def options_to_string(options):
  return '-'.join("{0}-{1}".format(k, v) for k,v in options.iteritems())





