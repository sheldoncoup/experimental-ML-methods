
import os
import csv
import pickle
import argparse
import numpy as np
from utils import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neural_network  import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def single_eval_multi(X_train, X_test,y_train, y_test, classifier, rand_seed):
  # select the classifier to use
  if classifier == 'LinearSVC':
    clf = LinearSVC(random_state = rand_seed, max_iter=-1)
  elif classifier == "LDA":
    clf = LinearDiscriminantAnalysis()
  elif classifier == 'SVCRBF':
    clf = SVC(kernel='rbf', random_state = rand_seed, gamma='auto', max_iter=-1)
  elif classifier == 'ET':
    clf = ExtraTreesClassifier(random_state=rand_seed)
  elif classifier == 'RF':
    clf = RandomForestClassifier(random_state=rand_seed)
  elif classifier == 'KNN':
    clf = KNeighborsClassifier()
  elif classifier == 'GNB':
    clf = GaussianNB()
  elif classifier == 'MLP':
    clf = MLPClassifier(max_iter=500,random_state=rand_seed)
  else:
    print('{} is not a recognised classifier'.format(classifier))
  # train the given classifier and make predictions
  clf.fit(X_train, y_train)
  y_preds = clf.predict(X_test)
  return y_preds


def dimension_reduce(X_train, X_test, nb_reduced_dims, rand_seed):
  #use pca to perform dimension reduction on the current data            
  pca = PCA(n_components=nb_reduced_dims,whiten=True, random_state=rand_seed)
  pca.fit(X_train)
  X_train = pca.transform(X_train)
  X_test = pca.transform(X_test)
  return X_train, X_test


def monte_carlo_eval_multi(features, labels, label_2_index,intermediate_layer_name, args):
  classifier_acc_dict = {}
  
  index_2_label = {i:l for l,i in label_2_index.items()} 
  word_labels = [index_2_label[list(x).index(1)] for x in list(labels)]
  
  for clf in args.classifiers:
    classifier_acc_dict[clf] = {'accuracys':[],
                                'ground-truth':[],
                                'predictions':[]}
  
  for i in range(args.num_evals):
    # split dataset into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(list(features), list(word_labels), test_size=args.percent_test,stratify=labels, random_state=args.rand_seed+i)
    X_train, X_test = dimension_reduce(X_train, X_test, args.reduced_dims, args.rand_seed+i)
    # make predictions 
    for clf in args.classifiers:  
      y_preds = single_eval_multi(X_train, X_test,y_train,y_test,clf, args.rand_seed+i)
  
      # record precictions and calculate metrics
      classifier_acc_dict[clf]['accuracys'].append(accuracy_score(y_test,y_preds))
      classifier_acc_dict[clf]['predictions'].extend(y_preds)
      classifier_acc_dict[clf]['ground-truth'].extend(y_test)
    
    # print some progress report
    if (i+1)%10 == 0:
      print('Running Accuracy for run {} out of {} using intermediate layer {}'.format( str(i+1),str(args.num_evals), intermediate_layer_name))
      for clf in args.classifiers:
        print('{} : {}'.format( clf, str(np.mean(classifier_acc_dict[clf]['accuracys']))))
     
 
  print('Finished accuracy evaluation run')
  return classifier_acc_dict

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--classifiers', type=str, help='What classifiers you would like to evaluate.', default='all')
  parser.add_argument('--reduced_dims', type=int, help='Number of dimensions to reduce down to each run.', default=128)
  parser.add_argument('--layers', type=str, help='Names of layers you would like evaluate the features from', default='all' )
  parser.add_argument('--dataset_name', type=str, help='Name of the dataset to be evaluated, used to find the pickle files.', required=True)
  parser.add_argument('--feature_dir', type=str, help='Directory that the feature files are stored at.', required=True)
  parser.add_argument('--output_dir', type=str, help='Where to save the output files',required=True) 
  parser.add_argument('--num_evals', type=int, help='Number of monte carlo runs to perform before terminating.', default=100)
  parser.add_argument('--percent_test', type=float, help='Amount of the dataset to use for testing, during each run.', default=0.1)
  parser.add_argument('--rand_seed', type=int, help='Seed for random parts of algorithm', default=1)
  args = parser.parse_args()
  return args


def main():
  args = parse_args()

  if args.classifiers == 'all':
    #args.classifiers = ['LinearSVC','SVCRBF', 'ET', 'RF','KNN', 'MLP','GNB',  'LDA']
    args.classifiers = ['LinearSVC','SVCRBF','KNN', 'MLP','GNB',  'LDA']
  else:
    assert args.classifiers in  ['LinearSVC','SVCRBF', 'ET', 'RF','KNN', 'MLP','GNB',  'LDA']
    args.classifiers = [args.classifiers]
  
  if args.layers == 'all':
    args.layers = ['mixed0', 'mixed1', 'mixed2','mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9']
  else:
    args.layers = [args.layers]
  
  if not os.path.isdir(args.output_dir):
    print('Output directory {} does not exist. Aborting.'.format(args.output_dir))
    raise ValueError
  
  if not os.path.isdir(args.feature_dir):
    print('Feature directory {} does not exist, Aborting.'.format(args.feature_dir))
    raise ValueError
  
  layers_acc_dict = {}
  for intermediate_layer_name in args.layers:
    if intermediate_layer_name == 'None':
      features, labels, label_2_index = load_from_disk(args.dataset_name, args.feature_dir)
    else:
      features, labels, label_2_index = load_from_disk(args.dataset_name+'_'+ intermediate_layer_name, args.feature_dir)
    layers_acc_dict[intermediate_layer_name] = monte_carlo_eval_multi(features, labels, label_2_index,intermediate_layer_name, args)
  
  # Save run details to disk
  txt_name=os.path.join(args.output_dir, 'eval-output-multiclass{}.csv'.format(args.dataset_name))
  results_dict_name=os.path.join(args.output_dir, 'eval-output-dict-multiclass-{}.p'.format(args.dataset_name))
  save_run_as_txt(layers_acc_dict, txt_name)
  pickle.dump(layers_acc_dict, open(os.path.join(args.output_dir, results_dict_name + '.p' ), 'wb'))
  return layers_acc_dict

if __name__ =='__main__':
  main()

