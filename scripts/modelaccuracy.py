from gensim.models import Word2Vec
import sys, json, os
from tqdm import tqdm
inputs = sys.argv[1:]
def print_accuracy(model_name):
  print "Loading model: " + model_name
  model = Word2Vec.load("models/" + model_name)
  print "Loading model " + model_name +" : Done"
  accuracy_sections = model.accuracy('benchmarks/questions.txt')
  for section in tqdm(accuracy_sections): 
    print str(len(section['correct'])) + "/" + str(len(section['correct']) + len(section['incorrect'])), section['section']

for inp in inputs:
  if(os.path.isdir(inp)):
    model_names = os.listdir(inp)
    for model_name in model_names:
      print_accuracy(model_name)
  else:
    print_accuracy(inp)