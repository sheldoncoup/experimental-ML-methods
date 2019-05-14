
import os
import pickle
import numpy as np
import argparse

from evaluate_classifiers_multiclass import dimension_reduce
from gaussianmixedmodel import GaussianMixed
from utils import *

from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score,roc_curve, auc
from sklearn.decomposition import PCA





def single_eval_one_class(X_train, X_test, y_test,species_train, args, class_list):
 
  if args.classifier == 'GaussianMixed':
    # define the gaussian mixed model, fit to training data and make predictions
    clf = GaussianMixed(class_list, threshold=args.threshold, mixmodel=True, epsilon=1e-6)
    clf.fit(X_train, species_train)
    y_preds, y_score = clf.predict(X_test)
    
  else:
    # define the one class svm, and fit it to the training data
    clf = OneClassSVM(kernel=args.classifier, gamma='auto' )
    clf.fit(X_train)
    #make predictions and calculate the roc curve
    y_preds = clf.predict(X_test)
    y_score = clf.decision_function(X_test)
  
  # calculate the false and true positive rate, followed by the AUROC
  fpr, tpr, _ = roc_curve(y_test, y_score)
  roc_auc = auc(fpr, tpr)
  return y_preds, roc_auc, fpr,tpr

  
def get_oneclass_split(features, labels, invasive_species, label_2_index,percent_test, rand_seed):
  # return a train/test split with all of the given invasive species in the test set
  one_class_labels = [-1 if x[label_2_index[invasive_species]] == 1 else 1 for x in labels]
  X_train, X_test, y_train, y_test, species_train, species_test = train_test_split(list(features), list(one_class_labels),list(labels), test_size=percent_test,stratify=labels, random_state=rand_seed)
  
  k = 0
  while k < len(X_train):
    if y_train[k] == -1:
      X_test.append(X_train.pop(k))
      y_test.append(y_train.pop(k))
      species_test.append(species_train.pop(k))
    else:
      k+=1
    
  return X_train, X_test, y_test, species_train

def encoding_to_species(labels, label_2_index):
  # takes a list of one hot encoded labels and returns the true names of the classes
  index_2_label = {index:label for (label,index) in label_2_index.items()}
  species_labels = []
  for x in labels:
    species_labels.append(index_2_label[list(x).index(1)])
  return species_labels


def monte_carlo_eval_one_class(features, labels, label_2_index,args):
  possible_species = label_2_index.keys()
  acc_dict = {}
  for invasive_species in possible_species:
    species_acc = []
    species_preds_archive = []
    species_test_data = []
    species_auroc = []
    species_fpr = []
    species_tpr = []
    class_list = [x for x in possible_species if x != invasive_species]
    
    for i in range(args.num_evals):
      # split dataset into train/test/invasive species
      X_train, X_test, y_test, species_train = get_oneclass_split(features, labels, invasive_species, label_2_index,args.percent_test, args.rand_seed+i)
      X_train, X_test = dimension_reduce(X_train, X_test, args.reduced_dims, args.rand_seed+i)
      
      # make predictions 
      y_preds, roc_auc, fpr,tpr = single_eval_one_class(X_train, X_test,y_test, encoding_to_species(species_train,label_2_index), args, class_list)
      
      # record precictions and calculate metrics
      species_acc.append(accuracy_score(y_test, y_preds) * 100)
      species_preds_archive.extend(y_preds)
      species_test_data.extend(y_test)
      species_auroc.append(roc_auc)
      species_fpr.append(fpr)
      species_tpr.append(tpr)
      if i%10 == 0:
        print( '{}{}'.format(invasive_species,args.dataset_name))
        print('Running Accuracy for run {} out of {}'.format(i+1, args.num_evals))
        print( '{}: {}'.format(args.classifier, str(np.mean(species_acc))))
        print( 'Running AUROC : {}'.format(str(np.mean(species_auroc))))
    # final recording of predictions and metrics
    acc_dict[str(invasive_species)] = {'accuracys':species_acc,
                                     'kappa-score':cohen_kappa_score(species_test_data, species_preds_archive),
                                     'ground-truth':species_test_data,
                                     'predictions':species_preds_archive,
                                     'aurocs':species_auroc,
                                     'fpr':species_fpr,
                                     'tpr':species_tpr}
  return acc_dict


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--classifier', type=str, help='What classifier you would like to evaluate, either linear, rbf (both SVM) or GaussianMixed.', required=True)
  parser.add_argument('--reduced_dims', type=int, help='Number of dimensions to reduce down to each run.', default=128)
  parser.add_argument('--layers', type=str, help='Names of layers you would like evaluate the features from', default='all' )
  parser.add_argument('--dataset_name', type=str, help='Name of the dataset to be evaluated, used to find the pickle files.', required=True)
  parser.add_argument('--feature_dir', type=str, help='Directory that the feature files are stored at.', required=True)
  parser.add_argument('--output_dir', type=str, help='Where to save the output files',required=True)
  parser.add_argument('--threshold', type=float, help='Threshold level to classification', default=-20) 
  parser.add_argument('--num_evals', type=int, help='Number of monte carlo runs to perform before terminating.', default=100)
  parser.add_argument('--percent_test', type=float, help='Amount of the dataset to use for testing, during each run.', default=0.1)
  parser.add_argument('--rand_seed', type=int, help='Seed for random parts of algorithm', default=1)
  args = parser.parse_args()
  return args

def main():
  # Parse command line arguments
  args = parse_args()
  print(args)
  if args.layers == 'all':
    args.layers = ['mixed0', 'mixed1', 'mixed2','mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9']
  else:
    assert args.layers in ['mixed0', 'mixed1', 'mixed2','mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9', 'None']
    args.layers = [args.layers]

  print(type(args.layers))
  print(args)
 
  # Check that the input arguments are valid to complete the run
  if not args.classifier in ['GaussianMixed', 'linear', 'rbf']:
    print('The given classifier {} is not a supported one class classifier'.format(args.classifier))
    raise ValueError
  
  if not os.path.isdir(args.output_dir):
    print('Output directory {} does not exist. Aborting.'.format(args.output_dir))
    raise ValueError
  
  if not os.path.isdir(args.feature_dir):
    print('Feature directory {} does not exist, Aborting.'.format(args.feature_dir))
    raise ValueError
  
  acc_dict_layers = {}
  
  for intermediate_layer_name in args.layers:
    if intermediate_layer_name == 'None':
      features, labels, label_2_index = load_from_disk(args.dataset_name, args.feature_dir)
    else:
      features, labels, label_2_index = load_from_disk(args.dataset_name+'_'+ intermediate_layer_name, args.feature_dir)
    # Evaluate the accuracy of the classifier for this layer
    acc_dict_layers[intermediate_layer_name] = monte_carlo_eval_one_class(features, labels, label_2_index,args) 
  
  # Save information to disk
  txt_name=os.path.join(args.output_dir, 'eval-output-oneclass{}.csv'.format(args.dataset_name))
  results_dict_name=os.path.join(args.output_dir, 'eval-output-dict-oneclass-{}.p'.format(args.dataset_name))
  save_run_as_txt_with_auroc(acc_dict_layers, txt_name)
  pickle.dump(acc_dict_layers, open(results_dict_name + '.p', 'wb'))
  return acc_dict_layers




if __name__ == '__main__':
  main()
