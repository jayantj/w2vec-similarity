from train import train
from glob import glob
import argparse, sys, pdb
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Script to train word2vec models from corpuses or raw files\nRun python run.py --help for help')

  parser.add_argument('-f', '--files', nargs='+', help='Input raw text files to be used for training')
  parser.add_argument('-d', '--directory', type=str, help='Directory containing input raw text files')
  parser.add_argument('-p', '--pattern', type=str, help='Input pattern for raw text files to be used for training')

  parser.add_argument('-c', '--corpuses', nargs='+', help='Input corpuses to be used for training')
  parser.add_argument('-cc', '--components', nargs='+', help='Corpus components to be used for training from given corpus')
  parser.add_argument('-cm', '--components_method', help='Corpus method to be called for components be used for training from given corpus')
  parser.add_argument('-s', '--separate', default=False, action='store_true', help='Create separate models for passed corpuses/files')

  parser.add_argument('-o', '--output_file', default='', help='Output file name in which to save the trained model')

  parser.add_argument('-i', '--iter', default=1, type=int, help='Numer of iterations/epochs for the trained model(s)')
  parser.add_argument('-n', '--size', default=100, type=int, help='Dimensionality of the feature vectors to be trained')
  
  args = parser.parse_args()
  options = {}
  for option in ('iter', 'size'):
    options[option] = getattr(args, option)
  if (args.files and args.directory) or args.pattern:
    train.train_from_files(args.files, args.directory, args.pattern, args.separate, options)
  elif isinstance(args.corpuses, list) and len(args.corpuses):
    if args.components_method:
      train.train_from_corpus_components(args.corpuses[0], args.components_method, args.components, output_file=args.output_file, separate=args.separate, options=options)
    else:
      train.train_from_corpuses(args.corpuses, output_file=args.output_file, options=options)
  else:
    parser.print_help()

  