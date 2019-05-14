from keras.applications.inception_v3 import InceptionV3, preprocess_input
import argparse
from utils import *
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from evaluate_classifiers_multiclass import dimension_reduce, single_eval_multi

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--classifier', type=str, help='What classifiers you would like to evaluate.', default='all')
  parser.add_argument('--reduced_dims', type=int, help='Number of dimensions to reduce down to each run.', default=128)
  parser.add_argument('--layer', type=str, help='Names of layers you would like evaluate the features from', default='all' )
  parser.add_argument('--dataset_name', type=str, help='Name of the dataset to be evaluated, used to find the pickle files.', required=True)
  parser.add_argument('--feature_dir', type=str, help='Directory that the feature files are stored at.', required=True)
  parser.add_argument('--output_dir', type=str, help='Where to save the output files',required=True) 
  parser.add_argument('--rand_seed', type=int, help='Seed for random parts of algorithm', default=1)
  parser.add_argument('--test_dir',type=str, help='where the test images are stored', required=True)
  args = parser.parse_args()
  return args

def extract_test_features(data_dir, model=InceptionV3(include_top=False, weights='imagenet', pooling='max')):
  dataframe = pd.DataFrame([[x,x] for x in os.listdir(os.path.join(data_dir, '0'))], columns=['filename', 'class'])
  datagen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(dataframe, directory=os.path.join(data_dir,'0'), shuffle=False, target_size=(299,299))
  test_features = model.predict_generator(datagen)
  return test_features, dataframe['filename']
  
def make_predictions(features, labels, label_2_index, test_features, args):
  index_2_label = {i:l for l,i in label_2_index.items()} 
  word_labels = [index_2_label[list(x).index(1)] for x in list(labels)]
  
  X_train, X_test = dimension_reduce(features, test_features, args.reduced_dims, args.rand_seed) 
  preds = single_eval_multi(X_train, X_test, word_labels, None, args.classifier, args.rand_seed)
  return preds
   


def main():
  args = parse_args()
  assert args.classifier in  ['LinearSVC','SVCRBF', 'ET', 'RF','KNN', 'MLP','GNB',  'LDA']
    
  

  
  if not os.path.isdir(args.output_dir):
    print('Output directory {} does not exist. Aborting.'.format(args.output_dir))
    raise ValueError
  
  if not os.path.isdir(args.feature_dir):
    print('Feature directory {} does not exist, Aborting.'.format(args.feature_dir))
    raise ValueError
  
  layers_acc_dict = {}
  
  if args.layer == 'None':
    features, labels, label_2_index = load_from_disk(args.dataset_name, args.feature_dir)
    
  else:
    features, labels, label_2_index = load_from_disk(args.dataset_name+'_'+ args.layer, args.feature_dir)
  test_features, filenames = extract_test_features(args.test_dir)
  
  preds = make_predictions(features,labels, label_2_index, test_features, args)  
  imnumber = [int(x.split('-')[-1][:-4]) for x in filenames]
  out_dataframe = pd.DataFrame(zip(preds, filenames, imnumber), columns=['Category','Id', 'imnumber']).sort_values('imnumber')
  
  out_dataframe.to_csv(os.path.join(args.output_dir, 'predictions.csv'),columns=['Category','Id'], index=False)
if __name__ == '__main__':
  main()
  
