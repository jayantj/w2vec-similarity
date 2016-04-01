from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from glob import glob
from nltk.corpus import PlaintextCorpusReader
import importlib, pdb, os, chardet

MODELS_DIR = 'models/'

def train_and_save(sents, output_file, options = {}):
  print "Training model..."
  model = Word2Vec(sents, **options)
  model.save(output_file)

def train_and_save_doc2vec(docs, output_file, options = {}):
  print "Training model..."
  model = Doc2Vec(docs, **options)
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

def train_from_files(files=[], root_dir='', pattern='', separate=False, output_file = '', options={}):
  if pattern:
    root_dir = os.path.dirname(pattern) + '/'
    files = glob(pattern)
  file_names = [os.path.basename(file) for file in files]
  if separate:
    for fname in file_names:
      read_and_train(root_dir, fname, output_file, options)
  else:
    read_and_train(root_dir, file_names, output_file, options)

def options_to_string(options):
  return '-'.join("{0}-{1}".format(k, v) for k,v in options.iteritems())

def read_and_train(root_dir, fileids, output_file='', options={}):
  fileids =  fileids if isinstance(fileids, list) else [fileids]
  fileids = [unicode(f, 'utf8') for f in fileids]
  output_file = output_file or '-'.join(fileids)
  output_file = u"{0}{1}-{2}".format(MODELS_DIR, output_file, options_to_string(options))
  reader = PlaintextCorpusReader(root=root_dir, fileids=fileids)
  try:
    sents = reader.sents()
    print fileids
    train_and_save(sents, output_file, options)
  except UnicodeDecodeError:
    print "here"
    file_encodings = {}
    for fileid in fileids:
      file_content = open(root_dir + fileid).read()
      file_encoding = chardet.detect(file_content)
      file_encodings[fileid] = file_encoding['encoding']
    reader._encoding = file_encodings
    sents = reader.sents()
    train_and_save(sents, output_file, options)

def train_doc2vec_from_files(files=[], root_dir='', pattern='', output_file = '', options={}):
  if pattern:
    root_dir = os.path.dirname(pattern) + '/'
    files = glob(pattern)
  file_names = [os.path.basename(file) for file in files]
  read_and_train_doc2vec(root_dir, file_names, output_file, options)

def read_and_train_doc2vec(root_dir, fileids, output_file='', options={}):
  fileids =  fileids if isinstance(fileids, list) else [fileids]
  fileids = [unicode(f, 'utf8') for f in fileids]
  output_file = output_file or '-'.join(fileids)
  output_file = u"{0}{1}-{2}".format(MODELS_DIR, output_file, options_to_string(options))
  reader = PlaintextCorpusReader(root=root_dir, fileids=fileids)
  try:
    docs = [ TaggedDocument(reader.words(fileid), [fileid]) for fileid in fileids ]
    train_and_save_doc2vec(docs, output_file, options)
  except UnicodeDecodeError:
    file_encodings = {}
    for fileid in fileids:
      file_content = open(root_dir + fileid).read()
      file_encoding = chardet.detect(file_content)
      file_encodings[fileid] = file_encoding['encoding']
    reader._encoding = file_encodings
    pdb.set_trace()
    docs = [ TaggedDocument(reader.words(fileid), [fileid]) for fileid in fileids ]
    train_and_save_doc2vec(docs, output_file, options)


