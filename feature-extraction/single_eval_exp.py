from evaluate_classifiers_multiclass import single_eval_multi, dimension_reduce
from evaluate_classifiers_oneclass import single_eval_one_class, encoding_to_species
from utils import *
import argparse
from sklearn.metrics import accuracy_score

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--classifier', type=str, help='What classifiers you would like to evaluate.', default='all')
  parser.add_argument('--reduced_dims', type=int, help='Number of dimensions to reduce down to each run.', default=128)
  parser.add_argument('--layers', type=str, help='Names of layers you would like evaluate the features from', default='all' )
  parser.add_argument('--dataset_name', type=str, help='Name of the dataset to be evaluated, used to find the pickle files.', required=True)
  parser.add_argument('--feature_dir', type=str, help='Directory that the feature files are stored at.', required=True)
  parser.add_argument('--output_dir', type=str, help='Where to save the output files',required=True) 
  parser.add_argument('--num_evals', type=int, help='Number of monte carlo runs to perform before terminating.', default=100)
  parser.add_argument('--percent_test', type=float, help='Amount of the dataset to use for testing, during each run.', default=0.1)
  parser.add_argument('--rand_seed', type=int, help='Seed for random parts of algorithm', default=1)
  parser.add_argument('--threshold', type=float, help='Threshold level to classification', default=-20) 
  args = parser.parse_args()
  return args




def main_multi_class():
  args = parse_args()

  if args.classifier == 'all':
    args.classifier = ['LinearSVC','SVCRBF','KNN', 'MLP','GNB',  'LDA']
  else:
    assert args.classifier in  ['LinearSVC','SVCRBF', 'ET', 'RF','KNN', 'MLP','GNB',  'LDA']
    args.classifiers = [args.classifier]
  
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
  '''
  for intermediate_layer_name in args.layers:
    X_train, y_train, train_label_2_index = load_from_disk('{}_train_{}'.format(args.dataset_name, intermediate_layer_name), args.feature_dir)
    X_valid, y_valid, valid_label_2_index = load_from_disk('{}_valid_{}'.format(args.dataset_name, intermediate_layer_name), args.feature_dir)
    X_test, y_test, test_label_2_index = load_from_disk('{}_test_{}'.format(args.dataset_name, intermediate_layer_name), args.feature_dir)
    assert test_label_2_index == train_label_2_index
    #X_test  = X_valid
    #y_test = y_valid
    index_2_label = {i:l for l,i in train_label_2_index.items()} 
    y_train = [index_2_label[list(x).index(1)] for x in list(y_train)]
    y_test = [index_2_label[list(x).index(1)] for x in list(y_test)]
    X_train, X_test = dimension_reduce(X_train, X_test, args.reduced_dims, args.rand_seed)
    
    
    
    classifier_acc_dict = {}
    for clf in args.classifiers:  
      y_preds =  single_eval_multi(X_train, X_test,y_train,y_test,clf, args.rand_seed)
  
      # record precictions and calculate metrics
      classifier_acc_dict[clf] = {}
      classifier_acc_dict[clf]['accuracys'] = [accuracy_score(y_test,y_preds)]
      classifier_acc_dict[clf]['predictions'] = y_preds
      classifier_acc_dict[clf]['ground-truth'] = y_test
    layers_acc_dict[intermediate_layer_name] = classifier_acc_dict
  '''
  
  X_train, y_train, train_label_2_index = load_from_disk('{}_valid'.format(args.dataset_name), args.feature_dir)
  #X_valid, y_valid, valid_label_2_index = load_from_disk('{}_valid'.format(args.dataset_name), args.feature_dir)
  X_test, y_test, test_label_2_index = load_from_disk('{}_test'.format(args.dataset_name), args.feature_dir)
  assert test_label_2_index == train_label_2_index
  #X_test  = X_valid
  #y_test = y_valid
  #data_splits  = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.899]
  data_splits  = [0.0]
  index_2_label = {i:l for l,i in train_label_2_index.items()} 
  y_train = [index_2_label[list(x).index(1)] for x in list(y_train)]
  y_test = [index_2_label[list(x).index(1)] for x in list(y_test)]
  for percent_test in data_splits:
    print(percent_test)
    if percent_test == 0.0:
      X_train_reduced = X_train
      y_train_reduced = y_train
    else:
      X_train_reduced, X_test_discard, y_train_reduced, y_test_discard = train_test_split(list(X_train), list(y_train), test_size=percent_test,stratify=y_train)
      
    print(len(X_train_reduced))
    if len(X_train_reduced) < args.reduced_dims:
      args.reduced_dims = len(X_train_reduced)
      
    X_train_reduced, X_test_reduced = dimension_reduce(X_train_reduced, X_test, args.reduced_dims, args.rand_seed)
    
    
    
    classifier_acc_dict = {}
    for clf in args.classifier:  
      y_preds =  single_eval_multi(X_train_reduced, X_test_reduced,y_train_reduced,y_test,clf, args.rand_seed)
      # record precictions and calculate metrics
      classifier_acc_dict[clf] = {}
      classifier_acc_dict[clf]['accuracys'] = [accuracy_score(y_test,y_preds)]
      classifier_acc_dict[clf]['predictions'] = y_preds
      classifier_acc_dict[clf]['ground-truth'] = y_test
    layers_acc_dict[str(percent_test)] = classifier_acc_dict
  
  
  
  
  
  # Save run details to disk
  txt_name=os.path.join(args.output_dir, 'eval-output-multiclass{}.csv'.format(args.dataset_name))
  results_dict_name=os.path.join(args.output_dir, 'eval-output-dict-multiclass-{}.p'.format(args.dataset_name))
  save_run_as_txt(layers_acc_dict, txt_name)
  pickle.dump(layers_acc_dict, open(os.path.join(args.output_dir, results_dict_name + '.p' ), 'wb'))
  return layers_acc_dict

def main_one_class():
  # Parse command line arguments
  args = parse_args()
  print(args)
  if args.layers == 'all':
    args.layers = ['mixed0', 'mixed1', 'mixed2','mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9']
  else:
    assert args.layers in ['mixed0', 'mixed1', 'mixed2','mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9']
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
    X_train, y_train, train_label_2_index = load_from_disk('{}_train_{}'.format(args.dataset_name, intermediate_layer_name), args.feature_dir)
    #X_valid, y_valid, valid_label_2_index = load_from_disk('{}_valid_{}'.format(args.dataset_name, intermediate_layer_name), args.feature_dir)
    X_test, y_test, test_label_2_index = load_from_disk('{}_test_{}'.format(args.dataset_name, intermediate_layer_name), args.feature_dir)
    # Evaluate the accuracy of the classifier for this layer
    assert test_label_2_index == train_label_2_index
    print(intermediate_layer_name)
    #X_test  = X_valid
    #y_test = y_valid
    label_2_index = train_label_2_index
    
    index_2_label = {i:l for l,i in train_label_2_index.items()} 
    #y_train = [index_2_label[list(x).index(1)] for x in list(y_train)]
    #y_test = [index_2_label[list(x).index(1)] for x in list(y_test)]
    X_train, X_test = dimension_reduce(X_train, X_test, args.reduced_dims, args.rand_seed)
    
    
    possible_species = label_2_index.keys()
    acc_dict_species = {}
    for invasive_species in possible_species:
      #print(invasive_species)
      class_list = [x for x in possible_species if x != invasive_species]
      y_train_one_class = [-1 if x[label_2_index[invasive_species]] == 1 else 1 for x in y_train]
      y_test_one_class =  [-1 if x[label_2_index[invasive_species]] == 1 else 1 for x in y_test]
      zipped_train_data = [ x for x in zip(X_train,y_train, y_train_one_class) if x[2] != -1 ]
      X_train_species = [x[0] for x in zipped_train_data]
      species_train = [x[1] for x in zipped_train_data]
      
      y_preds, roc_auc, fpr,tpr = single_eval_one_class(X_train_species, X_test,y_test_one_class, encoding_to_species(species_train,label_2_index), args, class_list)
      #print(y_preds)
      # record precictions and calculate metrics
    
      acc_dict_species[str(invasive_species)] = {'accuracys':accuracy_score(y_test_one_class, y_preds) * 100,
                                     'kappa-score':cohen_kappa_score(y_test_one_class, y_preds),
                                     'ground-truth':y_test,
                                     'predictions':y_preds,
                                     'aurocs':roc_auc,
                                     'fpr':fpr,
                                     'tpr':tpr}
      
    acc_dict_layers[intermediate_layer_name] = acc_dict_species
  '''
  X_train, y_train, train_label_2_index = load_from_disk('{}_train'.format(args.dataset_name), args.feature_dir)
  X_valid, y_valid, valid_label_2_index = load_from_disk('{}_valid'.format(args.dataset_name), args.feature_dir)
  X_test, y_test, test_label_2_index = load_from_disk('{}_test'.format(args.dataset_name), args.feature_dir)
  # Evaluate the accuracy of the classifier for this layer
  assert test_label_2_index == train_label_2_index
  #print(intermediate_layer_name)
  #X_test  = X_valid
  #y_test = y_valid
  label_2_index = train_label_2_index
  
  index_2_label = {i:l for l,i in train_label_2_index.items()} 
  #y_train = [index_2_label[list(x).index(1)] for x in list(y_train)]
  #y_test = [index_2_label[list(x).index(1)] for x in list(y_test)]
  X_train, X_test = dimension_reduce(X_train, X_test, args.reduced_dims, args.rand_seed)
    
  n = 0
  possible_species = label_2_index.keys()
  acc_dict_species = {}
  for invasive_species in possible_species:
    #print(invasive_species)
    class_list = [x for x in possible_species if x != invasive_species]
    y_train_one_class = [-1 if x[label_2_index[invasive_species]] == 1 else 1 for x in y_train]
    y_test_one_class =  [-1 if x[label_2_index[invasive_species]] == 1 else 1 for x in y_test]
    zipped_train_data = [ x for x in zip(X_train,y_train, y_train_one_class) if x[2] != -1 ]
    X_train_species = [x[0] for x in zipped_train_data]
    species_train = [x[1] for x in zipped_train_data]
    
    y_preds, roc_auc, fpr,tpr = single_eval_one_class(X_train_species, X_test,y_test_one_class, encoding_to_species(species_train,label_2_index), args, class_list)
    print(n/len(possible_species))
    n+=1
    # record precictions and calculate metrics
    
    acc_dict_species[str(invasive_species)] = {'accuracys':accuracy_score(y_test_one_class, y_preds) * 100,
                                     'kappa-score':cohen_kappa_score(y_test_one_class, y_preds),
                                     'ground-truth':y_test,
                                     'predictions':y_preds,
                                     'aurocs':roc_auc,
                                     'fpr':fpr,
                                     'tpr':tpr}
      
  acc_dict_layers['last_layer'] = acc_dict_species
  '''
  # Save information to disk
  txt_name=os.path.join(args.output_dir, 'eval-output-oneclass{}.csv'.format(args.dataset_name))
  results_dict_name=os.path.join(args.output_dir, 'eval-output-dict-oneclass-{}.p'.format(args.dataset_name))
  save_run_as_txt_with_auroc(acc_dict_layers, txt_name)
  pickle.dump(acc_dict_layers, open(results_dict_name + '.p', 'wb'))
  return acc_dict_layers
if __name__ =='__main__':
  main_multi_class()
  #main_one_class()
