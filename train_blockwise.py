
'''
Copyright 2017 TensorFlow Authors and Sheldon Coup

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adadelta
from keras.callbacks import EarlyStopping
import argparse
import inception_v4
import alexnet
import numpy as np
import cv2
import os
import pickle
import shutil
import gc
from keras.utils.training_utils import multi_gpu_model
import time
from plotting_blockwise import plot_ete, plot_phase
from utils import *

# If you want to use a GPU set its index here
<<<<<<< HEAD
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', type=str, help='Name of the dataset to be evaluated.', required=True)
  parser.add_argument('--num_classes', type=int, help='Number of classes in the dataset.', required=True)
  parser.add_argument('--image_size', type=int, help='Desired side length of the input images.', default=227)
  parser.add_argument('--model_name', type=str, help='Name of the model to train on, only "alexnet" and "inception_v4" currently allowed', default='alexnet')
  parser.add_argument('--batch_size', type=int, help='Number of images per batch', default=32)
  parser.add_argument('--data_dir', type=str, help='Directory where the data is stored.', required=True)
  parser.add_argument('--ckpt_dir', type=str, help='Directory of the fine tuned checkpoint.', default=None)
  parser.add_argument('--num_gpu', type=int, help='Number of gpus to use for training', default = 1)
  parser.add_argument('--start_phase', type=int, help='Phase to start training at, only non zero if resuming from a previous run', default = 0)
  parser.add_argument('--freeze_layers', type=bool, help='Wherether of not phase training should be run, set false for standard training', default = True)
  parser.add_argument('--optimizer', type=str, help='Optimizer choice: Adadelta or Adam', default = 'Adadelta')
  parser.add_argument('--lr', type=float, help='Learning rate to use', default = 0)
  parser.add_argument('--dr', type=float, help='Decay rate to use.', default = 0)
  parser.add_argument('--max_epoch', type=int, help='Maximum number of epochs to train for on each run', default = 100)
  parser.add_argument('--use_test', type=bool, help='Whether or not to evaluate the test set', default=False)
  parser.add_argument('--rand_seed', type=int, help='What seed to use for the random initializations', default=1)
  args = parser.parse_args()
  return args


=======
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#### DATASET PARAMETERS ####
# Name of the dataset to train on
dataset_to_use = 'caltech-birds'
# Number of classes in the dataset
num_classes = 200
# Where the images can be found
data_dir = '/Scratch/stc53/images/'
#If the images need to be sorted into train/valid/test directories, what file is to be used for that
split_file_dir = '/Scratch/stc53/caltech_birds_trainsize_20_validation.p'
#If the images need to be sorted into train/valid/test directories, where should they be moved to 
target_data_dir = '/Scratch/stc53/images-split/'

#### MODEL PARAMETERS ####
#Which model to use
model_name = 'inceptionV4'
#How many training epochs to do between validation set checking
epoch_per_run = 1
# if the validation accuracy decreases by less than this amount in a run, 
#the script considers the program to have stalled for that run
accuracy_metric = 0.0025
# batch size
batch_size = 32
# how many sections the given model has been split into
# Do not change unless the model model structure has been specifically tweaked
total_phases = 5
#where the checkpoints and info pickle files should be saved to
checkpoint_dir = '/localhome/stc53/phasenets-saves/checkpoints/'
# how many times the training has to have been considered to be stalled before the training is cancelled
stall_cutoff = 25
# Boolean to check if the script should load the imagenet weights checkpoint
load_imagenet = False
# minimum number of epochs to train for and how often to save in this training stage
min_epoch = 200



#### OPTIMIZER PARAMETERS ####
#What optimizer to use
opt_name = 'Adadelta'
#Learning rate
lr = None


#### MISC OPTIONS ####
#Whether or not the data needs to be moved into train/test/validation directories
data_to_be_moved = False
# Whether or not the moved data should be deleted at the end of training
# Non functional currenty, can be implemented to enable automatic training on a number of of different training set sizes
clear_moved_data = False
#whether or not the n-1 phase layers should be frozen when training a model of phase n
# Set True for phase training, False for Complete (end to end) training 
freeze_layers=True
# boolean to decide if exit conditions should be used
exit_if_stalled = True
# seed to use for random initializations
rand_seed = 1
#whether or not to use multi-gpu training
multigpu = False






def copy_file_list(file_list, target_data_dir):
  # Given a list of file directories, moves them into the given target directory 
  if not os.path.exists(target_data_dir):
    os.mkdir(target_data_dir)
>>>>>>> 624c808217c04481a50f7c8a7a0778b0ef112293
  
    
def model_loader(model_name, num_classes = 200, weights=None, include_top=True, phase=4, freeze_layers = False, img_size=299):
  # wrapper for loading models, currently only for the inceptionV4 model but more to be created in the future.
  if model_name == 'inception_v4':
    return inception_v4.create_model(num_classes = num_classes, weights=weights, include_top=True, phase=phase, freeze_layers=freeze_layers)
  elif model_name == 'alexnet':
    return alexnet.create_model(num_classes = num_classes,img_size=img_size, weights=weights, phase=phase, freeze_layers=freeze_layers)

def optimizer_loader(optimizer_name, learning_rate = None, decay_rate = 0.0):
<<<<<<< HEAD
  if learning_rate == 0:
=======
  if learning_rate == None:
>>>>>>> 624c808217c04481a50f7c8a7a0778b0ef112293
    if optimizer_name == 'Adadelta':
      return Adadelta()
    elif optimizer_name == 'Adam':
      return Adam()
  else:
    if optimizer_name == 'Adadelta':
      return Adadelta(lr = learning_rate, decay_rate = decay_rate)
    elif optimizer_name == 'Adam':
      return Adam(lr = learning_rate, decay_rate = decay_rate)

def build_generators(target_data_dir, img_size, batch_size,usetest=False, rand_seed=1):
  # initialize the data pipelines
  train_datagen = ImageDataGenerator(
                                    rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    rotation_range = 180)
  
  test_datagen = ImageDataGenerator(rescale=1./255)
  
  train_generator = train_datagen.flow_from_directory(
                                    os.path.join(target_data_dir, 'train'),
                                    target_size = (img_size,img_size),
                                    batch_size = batch_size,
                                    seed = rand_seed)
  if usetest:
    test_generator = test_datagen.flow_from_directory(
                                      os.path.join(target_data_dir, 'test'),
                                      target_size = (img_size,img_size),
                                      batch_size = batch_size,
                                      seed = rand_seed)
  else:
    test_generator = None
  
  valid_generator = test_datagen.flow_from_directory(
                                    os.path.join(target_data_dir, 'validation'),
                                    target_size = (img_size,img_size),
                                    batch_size = batch_size,
                                    seed = rand_seed) 
  return train_generator, valid_generator, test_generator

def main():
  args = get_args()
  

  train_generator, valid_generator, test_generator = build_generators(args.data_dir, args.image_size, args.batch_size,usetest=args.use_test, rand_seed=1)
  
  # inital setup
  if not args.start_phase == 0:
    # check if weights file exists, crash if not
    weights_to_load = os.path.join( args.ckpt_dir, args.model_name + '_phase_' + str(args.start_phase-1) + '.h5')
    if not os.path.isfile(weights_to_load):
      weights_to_load = 'random'
      raise ValueError('Required weights file does not exist! Aborting...')
  else:
    weights_to_load = 'random'
  model = None                                 
  if not args.freeze_layers:
    args.start_phase = 4
  
  for phase_num in range(args.start_phase,5):
    
    # clear previous model from memory to prevent memory leaking
    if model:
        print('Clearing model from memory')
        del model
        for b in range(10):
            gc.collect()
        K.clear_session()

    
    # Create model and load pre-trained weights
    model = model_loader(args.model_name, num_classes = args.num_classes, weights=weights_to_load, include_top=True, phase=phase_num, freeze_layers = args.freeze_layers,img_size = args.image_size)
    if  not args.num_gpu == 1:
      model = multi_gpu_model(model, gpus=args.num_gpu)
    opt = optimizer_loader(args.optimizer, learning_rate = args.lr, decay_rate = args.dr) 
    model.compile(loss= 'categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    model.summary()
    if args.freeze_layers:
      run_name =  args.model_name + '_phase_' + str(phase_num)
    else:
      run_name = model_name + '_end_to_end' 
    checkpoint_name = os.path.join(args.ckpt_dir, run_name + '.h5')
    log_dir = os.path.join(args.ckpt_dir, run_name +'.p') 
    

<<<<<<< HEAD
    #Train the model on the training data for a set number of epochs
    run_hist = model.fit_generator(train_generator, validation_data = valid_generator, epochs = args.max_epoch)
    train_record = {'run_name':run_name, 
                    'train_acc': run_hist.history['acc'],
                    'train_loss':run_hist.history['loss'], 
                    'valid_acc':run_hist.history['val_acc'], 
                    'valid_loss':run_hist.history['val_loss']}  
=======
      # print information so training can be monitored
      print('Current Validation Accuracy : ' + str(current_valid_acc))
      print('Current Validation Loss : ' + str(current_valid_loss))
      print('Top Validation Accuracy : ' + str(top_valid_acc))
      if exit_if_stalled:
        print('Stall count : ' + str(training_stalled))
      
      # record information about run to be saved later
      train_record['train_acc'].extend(list(run_hist.history['acc']))
      train_record['train_loss'].extend(list(run_hist.history['loss']))
      train_record['valid_acc'].extend(list(run_hist.history['val_acc']))
      train_record['valid_loss'].extend(list(run_hist.history['val_loss']))
      
      # dump the current train record in a pickle file and plot the results
      with open(log_dir, 'wb') as f:
        pickle.dump(train_record,f)
      if freeze_layers:
        plot_phase(phase_num, model_name, log_dir, total_phases)
      else:
        plot_ete(model_name, log_dir)
      
      #check if its time to stop training    
      if training_stalled > stall_cutoff and exit_if_stalled :
        still_learning = False
      
      #update tracking variables
      prev_valid_acc = current_valid_acc
      cycle_count += 1
      
>>>>>>> 624c808217c04481a50f7c8a7a0778b0ef112293
    
    
    if usetest:
      test_loss, test_acc = model.evaluate_generator(test_generator)
      train_record['test_acc'] = test_acc
      train_record['test_loss'] = test_loss
    
    with open(log_dir, 'wb') as f:
      pickle.dump(train_record,f)
    if freeze_layers:
      plot_phase(phase_num, args.model_name, os.path.dirname(log_dir), 5)
    else:
      plot_ete(args.model_name, os.path.dirname(log_dir))
    weights_to_load = checkpoint_name
    model.save_weights(weights_to_load)
  if usetest:
    test_loss, test_acc = model.evaluate_generator(test_generator)

    print()
    print('Test loss : ' + str(test_loss))
    print('Test acc : ' + str(test_acc))
      
  

if __name__ == "__main__":
  main()
