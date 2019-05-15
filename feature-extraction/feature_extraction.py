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

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image 

from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.utils import to_categorical

import csv

# which GPUs to use for training
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--extra_layer', type=str, help='Names of layers you would like evaluate the features from.', default='None' )
  parser.add_argument('--dataset_name', type=str, help='Name of the dataset to be evaluated, used to find the pickle files.')
  parser.add_argument('--feature_dir', type=str, help='Directory that the feature files are to be saved at.')
  parser.add_argument('--data_dir', type=str, help='Directory where the data is stored.')
  parser.add_argument('--ckpt_dir', type=str, help='Directory of the fine tuned checkpoint.', default='None')
  args = parser.parse_args()
  return args




def extract_features(model,data_dir, intermediate_layer_name, target_size=(299,299), pooling_type='mean'):
  # defines the model for multi depth feature extraction 
  if not intermediate_layer_name == 'None':
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(intermediate_layer_name).output)
    intermediate_layer_model.summary()
  
  #defines the generator to get the data 
  target_data = []
  labels = []
  name_to_index = {}
  i = 0
  for classname in os.listdir(data_dir):
    target_data.extend([image.load_img(os.path.join(data_dir, classname, f), target_size=target_size) for f in os.listdir(os.path.join(data_dir, classname))])
    labels.extend([i for f in os.listdir(os.path.join(data_dir, classname)) ])
    name_to_index[classname] = i
    i += 1
  print(len(labels))
  labels = to_categorical(labels, len(name_to_index))
  target_data = np.stack(target_data, axis=0)
  #target_data = [ (np.asarray(image.load_img(os.path.join(target_data_dir, f) ,target_size=target_size)), os.path.dirname(f)) for f in target_data_ids]
  #datagen = image.ImageDataGenerator(rescale=1./255)
  #extraction_generator = datagen.flow_from_directory(data_dir, target_size = (299,299), batch_size=1)
  
 
  # gets the output vectors
  if not intermediate_layer_name == 'None':
    intermediate_output = np.amax(intermediate_layer_model.predict(target_data), axis=(1,2))
  main_output = model.predict(target_data)
  
  # process the outputs into single dimensional vector forms
  if pooling_type == 'max':
    
    main_output = np.max(main_output, axis=(1,2))
  else:
    
    main_output = np.mean(main_output, axis=(1,2))
  # check the shapes
  #assert main_output.shape[0] == intermediate_output.shape[0]
  
  #print(intermediate_output.shape)
  
  #concatenate outputs together and return
  if not intermediate_layer_name == 'None':
    main_output = np.concatenate((main_output, intermediate_output), axis=1)
  print(main_output.shape)
  print('Finished feature extraction with layer ' + intermediate_layer_name)
  #del intermediate_layer_model
  return main_output, labels, name_to_index

def save_to_disk(features, labels,label_2_index, dataset_name, save_dir):
  # save feature and label arrays to disk
  with open(os.path.join(save_dir, dataset_name + '_features.p'), 'wb') as f:
    pickle.dump(features,f)
  with open(os.path.join(save_dir, dataset_name + '_labels.p'), 'wb') as f:
    pickle.dump(labels,f)
  with open(os.path.join(save_dir, dataset_name + '_class_indices.p'), 'wb') as f:
    pickle.dump(label_2_index,f)


  

def main():
  args = get_args()

  
  if not args.ckpt_dir == 'None':
    # create graph, import checkpoint for fine tuned  model
    model = InceptionV3(include_top=False, weights=None, input_shape=(299,299,3))
    print('Loading checkpoint {}'.format(args.ckpt_dir))
    #model.load_weights(ckpt,by_name = True, skip_mismatch = False)
    model.load_weights(ckpt, by_name=True)
  else:
    # import model with standard imagenet weights and work with that
    model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299,299,3))
  
  # sumamry of model
  model.summary()
  features,labels,label_2_index = extract_features(model, args.data_dir, args.extra_layer)
  save_to_disk(features, labels,label_2_index, '{}_{}'.format(args.dataset_name, args.extra_layer), args.feature_dir)
  
  print(features[0])
  

if __name__ == '__main__':
  main()
