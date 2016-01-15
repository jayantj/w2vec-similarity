from train import train
import argparse, sys, pdb
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Script to train word2vec models from corpuses or raw files\nRun python run.py --help for help')
  parser.add_argument('-f', '--files', nargs='+', help='Input raw text files to be used for training')
  parser.add_argument('-c', '--corpuses', nargs='+', help='Input corpuses to be used for training')
  parser.add_argument('-cc', '--components', nargs='+', help='Corpus components to be used for training from given corpus')
  parser.add_argument('-cm', '--components_method', help='Corpus method to be called for components be used for training from given corpus')

  parser.add_argument('-s', '--separate', default=False, action='store_true', help='Create separate models for passed corpuses/files')
  args = parser.parse_args()
  if isinstance(args.files, list) and len(args.files):
    pass
  elif isinstance(args.corpuses, list) and len(args.corpuses):
    if args.components_method:
      train.train_from_corpus_components(args.corpuses[0], args.components_method, args.components)
    else:
      train.train_from_corpuses(args.corpuses)
  else:
    parser.print_help()

  