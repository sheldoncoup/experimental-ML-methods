import os
import pickle
import keras
import shutil
import numpy as np
import keras.backend as K
import argparse
import gc
from termcolor import colored
from utils import *

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D


import csv

# which GPUs to use for training
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=str, help='Names of layers you would like evaluate the features from.', default='all' )
parser.add_argument('--dataset_name', type=str, help='Name of the dataset to be evaluated, used to find the pickle files.')
parser.add_argument('--feature_dir', type=str, help='Directory that the feature files are to be saved at.')
parser.add_argument('--data_dir', type=str, help='Directory where the data is stored.')
parser.add_argument('--ckpt_dir', type=str, help='Directory of the fine tuned checkpoint.', default=None)
args = parser.parse_args()
print(args)

if args.layers == 'all':
  intermediate_layers = ['mixed0', 'mixed1', 'mixed2','mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9']
else:
  intermediate_layers = [args.layers]

if args.ckpt_dir == None:
  import_ckpt = False
  ckpt = 'imagenet'
else:
  import_ckpt=True
  ckpt = args.ckpt_dir


dataset_name = args.dataset_name
save_dir = args.feature_dir
data_dir = args.data_dir




def extract_features(model,data_dir, intermediate_layer_name):
  # defines the model for multi depth feature extraction 
  #intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(intermediate_layer_name).output)
  #intermediate_layer_model.summary()
  
  # defines the generator to get the data 
  datagen = ImageDataGenerator(rescale=1./255)
  extraction_generator = datagen.flow_from_directory(data_dir, target_size = (299,299), batch_size=1)
  
  # save the data into array form the data 
  data = []
  data_length = len(recursive_save([], data_dir))//2
  print(data_length)
  while len(data)< data_length:
    data.append(extraction_generator.next())
  
  features = np.squeeze([x[0] for x in data])
  labels = np.squeeze([x[1] for x in data])
  
  # gets the output vectors
  #intermediate_output = intermediate_layer_model.predict(features)
  main_output = model.predict(features)
  
  # process the outputs into single dimensional vector forms
  #intermediate_output = np.amax(intermediate_output,axis=(1,2))
  #main_output = np.mean(main_output, axis=(1,2))
  main_output = np.max(main_output, axis=(1,2))
  # check the shapes
  #assert main_output.shape[0] == intermediate_output.shape[0]
  print(main_output.shape)
  #print(intermediate_output.shape)
  
  #concatenate outputs together and return
  #total_output = np.concatenate((main_output, intermediate_output), axis=1)
  #print('Finished feature extraction with layer ' + intermediate_layer_name)
  #del intermediate_layer_model
  return main_output, labels, extraction_generator.class_indices

def save_to_disk(features, labels,label_2_index, dataset_name, save_dir):
  # save feature and label arrays to disk
  with open(os.path.join(save_dir, dataset_name + '_features.p'), 'wb') as f:
    pickle.dump(features,f)
  with open(os.path.join(save_dir, dataset_name + '_labels.p'), 'wb') as f:
    pickle.dump(labels,f)
  with open(os.path.join(save_dir, dataset_name + '_class_indices.p'), 'wb') as f:
    pickle.dump(label_2_index,f)


  

def main():
    
  layers_acc_dict = {}
  
  if import_ckpt:
    # create graph, import checkpoint for fine tuned  model
    model = InceptionV3(include_top=False, weights=None, input_shape=(299,299,3))
    print('Loading checkpoint {}'.format(ckpt))
    #model.load_weights(ckpt,by_name = True, skip_mismatch = False)
    model.load_weights(ckpt, by_name=True)
  else:
    # import model with standard imagenet weights and work with that
    model = InceptionV3(include_top=False, weights='imagenet')
  
  # sumamry of model
  model.summary()
  features,labels,label_2_index = extract_features(model, data_dir, None)
  save_to_disk(features, labels,label_2_index, dataset_name, save_dir)
  print(features.any())
  print(features[0])
  '''
  for intermediate_layer_name in intermediate_layers:  
    # extracting features
    features, labels, label_2_index = extract_features(model, data_dir, intermediate_layer_name)
     
    # save features and labels
    save_to_disk(features, labels,label_2_index, dataset_name+'_'+intermediate_layer_name, save_dir)
    del features
    gc.collect()
  '''
 

if __name__ == '__main__':
  main()
