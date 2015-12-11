from gensim.models import Word2Vec
import sys, json

def save_model_vocab(model_name):
  print "Loading model: " + model_name
  model = Word2Vec.load("models/" + model_name)
  print "Loading model " + model_name +" : Done"

  vocab_dict = {}
  for word in model.vocab:
    most_similar = model.most_similar(word)
    most_similar_rounded = [[word_score[0], round(word_score[1], 3)] for word_score in most_similar]
    vocab_dict[word] = {res[0]:res[1] for res in most_similar_rounded}


  f = open("vocab/" + model_name + '-vocab.json','w')
  json_str = "{\n"
  for word in sorted(vocab_dict):
    json_str += "    "
    json_str += "\"" + word + "\": " + json.dumps(vocab_dict[word]) + "\n"



  # json.dump(vocab_dict, f, indent=4, sort_keys=True)
  f.write(json_str)
  f.close()

inputs = sys.argv[1:]
for inp in inputs:
  if(os.path.isdir(inp)):
    model_names = os.listdir(inp)
    for model_name in model_names:
      save_model_vocab(model_name)
  else:
    save_model_vocab(model_name)
  