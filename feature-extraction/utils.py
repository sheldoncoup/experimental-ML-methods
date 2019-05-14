
import shutil
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from termcolor import colored

def recursive_save(save_list, directory):
  # walks the given directory 
  for dirpath, dirnames, filenames in  os.walk(directory):
    # appends any files that are present to the file list
    for filename in filenames:
      save_list.append(os.path.join(dirpath, filename))
    #check if there are any subdirectories to be checked
    if dirnames:
      for subdir in dirnames:
        save_list = recursive_save(save_list,os.path.join(dirpath, subdir))
  return save_list 

def move_data_family(full_data_dir, train_data_dir,valid_data_dir, percent_valid = 0.1 ,rand_seed=1):
  # find the list of subdirectories to be checked
  classes = os.listdir(full_data_dir)
  for species in classes:
    # creates folder to be moved to
    species_train_dir = os.path.join(train_data_dir, species)
    species_valid_dir = os.path.join(valid_data_dir, species)
    if not os.path.exists(species_train_dir):
      os.mkdir(species_train_dir)
    if not os.path.exists(species_valid_dir):
      os.mkdir(species_valid_dir)
    # subdirectory to be searched
    species_dir = os.path.join(full_data_dir, species)
    # recursively finds all files
    species_files = recursive_save([], species_dir)
    # split into train/validation split
    train_images, valid_images = train_test_split(species_files, test_size=percent_valid, random_state=rand_seed)
    # moves images into the appropriate folders
    for image in train_images:
      shutil.copy(image, species_train_dir)
    for image in valid_images:
      shutil.copy(image, species_valid_dir)  
    print('Copied ' + str(len( species_files)) + ' files of the family ' + species)
  print('Done moving files around!')

def move_data_species(full_data_dir, train_data_dir, valid_data_dir, percent_valid = 0.1, rand_seed=1):
  image_list = recursive_save([], full_data_dir)
  class_list = [str(os.path.basename(x)).split('_')[0] for x in image_list ]
  for species in class_list:
    species_train_dir = os.path.join(train_data_dir, species)
    species_valid_dir = os.path.join(valid_data_dir, species)
    if not os.path.exists(species_train_dir):
      os.mkdir(species_train_dir)
    if not os.path.exists(species_valid_dir):
      os.mkdir(species_valid_dir)
  images_train, images_valid, class_train, class_valid = train_test_split(image_list, class_list, test_size=percent_valid, random_state=rand_seed)
  for image, im_class in zip(images_train, class_train):
    print(im_class)
    print(image)
    shutil.copy(image, os.path.join(train_data_dir, im_class))
  for image, im_class in zip(images_valid, class_valid):
    shutil.copy(image, os.path.join(valid_data_dir, im_class))
  print('Done moving files around')    



def print_avg_auroc(acc_dict):
  #prints accurcy auroc and cohen kappa scores for a given run, also returns the aver values for these metrics 
  auroc_archive = []
  acc_archive = []
  pred_archive = []
  gt_archive = []
  for layer in acc_dict.keys():
    for species in acc_dict[layer].keys():
      auroc_archive.append(np.mean(acc_dict[layer][species]['aurocs']))
      acc_archive.append(np.mean(acc_dict[layer][species]['accuracys']))
      pred_archive.extend(acc_dict[layer][species]['predictions'])
      gt_archive.extend(acc_dict[layer][species]['ground-truth'])
  print('The overall average auroc is {}.'.format(np.mean(auroc_archive)))
  print('The overall average accuracy is {}'.format(np.mean(acc_archive)))
  print('The overall Cohen Kappa score is {}'.format(cohen_kappa_score(pred_archive, gt_archive)))
  return cohen_kappa_score(pred_archive, gt_archive), np.mean(auroc_archive), np.mean(acc_archive)

def save_to_disk(features, labels,label_2_index, dataset_name, save_dir):
  # save feature and label arrays to disk
  with open(os.path.join(save_dir, dataset_name + '_features.p'), 'wb') as f:
    pickle.dump(features,f)
  with open(os.path.join(save_dir, dataset_name + '_labels.p'), 'wb') as f:
    pickle.dump(labels,f)
  with open(os.path.join(save_dir, dataset_name + '_class_indices.p'), 'wb') as f:
    pickle.dump(label_2_index,f)

def load_from_disk(dataset_name, save_dir):
  # save feature and label arrays to disk
  with open(os.path.join(save_dir, dataset_name + '_features.p'), 'rb') as f:
    features = pickle.load(f)
  
  with open(os.path.join(save_dir, dataset_name + '_labels.p'), 'rb') as f:
    labels = pickle.load(f)
  with open(os.path.join(save_dir, dataset_name + '_class_indices.p'), 'rb') as f:
    label_2_index = pickle.load(f)
  return features, labels, label_2_index

def save_run_as_txt(acc_dict, txt_name):
  # save accuracy results as a csv for the given run
  layers = sorted(list(acc_dict.keys()))
  classifiers = list(acc_dict[layers[0]].keys())
  results = []
  top_row = ['Classifier']
  top_row.extend(layers)
  results.append(top_row)
  for clf in classifiers:
    row = [clf]
    for lyr in layers:
      row.append(np.mean(acc_dict[lyr][clf]['accuracys']))
    results.append(row)
  np.savetxt(txt_name,np.array(results),fmt='%s', delimiter=',')

def save_run_as_txt_with_auroc(acc_dict, txt_name):
  # save accuracy and auroc results as a csv for the given run
  layers = sorted(list(acc_dict.keys()))
  species = list(acc_dict[layers[0]].keys())
  results = []
  top_row = ['Species']
  top_row.extend(layers)
  results.append(top_row)
  for sp in species:
    row = [sp]
    for lyr in layers:
      row.append(np.mean(acc_dict[lyr][sp]['accuracys']))
    results.append(row)
  results.append([0]* len(results[0]))
  for sp in species:
    row = [sp]
    for lyr in layers:
      row.append(np.mean(acc_dict[lyr][sp]['aurocs']))
    results.append(row) 
  np.savetxt(txt_name,np.array(results),fmt='%s', delimiter=',')
  

