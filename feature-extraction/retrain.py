

import os
import pickle
import keras
import shutil
import numpy as np
import keras.backend as K


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, help='Directory where the training data is stored')
parser.add_argument('--validation_data_dir', type=str, help='Directory where the validation data is stored.' )
parser.add_argument('--freeze_between', type=list, help='list of length two which specifies which section of the model to freeze for training', default = [])
parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
parser.add_argument('--ckpt_dir', type=str, help='Directory for the fine tuned checkpoint.', default=None)
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=64)
args = parser.parse_args()

train_data_dir = args['train_data_dir']
valid_data_dir = args['validation_data_dir']
freeze_batween = args['freeze_between']
dataset_name = args['dataset_name']
batch_size = args['batch_size']


checkpoint_name = os.path.join(args['ckpt_dir'], '{}.h5'.format(dataset_name))

    
def main():
  num_classes = len(os.listdir(train_data_dir))
  # create graph and initialize with imagenet weights, 
  base_model = InceptionV3(include_top=False, weights='imagenet')
  #randomly initialize the dense layers
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(num_classes, activation='softmax')(x)
  model = Model(inputs=base_model.inputs, outputs=predictions)
  
  # do we want to insert a quick pretraining run here were we just train the dense layers?
  if freeze_between:
    for i in range(freeze_between[0], freeze_between[1]):
      model.layers[i].trainable = False 
  # compile model for training 
  model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  
  # define data generators
  train_datagen = ImageDataGenerator(rescale = 1./255,
                                     rotation_range = 40,
                                     horizontal_flip = True)
  valid_datagen = ImageDataGenerator(rescale = 1./255)
  
  train_generator = train_datagen.flow_from_directory(train_data_dir, target_size = (299,299), batch_size =batch_size)
  valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size=(299,299), batch_size=batch_size)
  
  # define early stopping procedure
  early_stopping_monitor = EarlyStopping(monitor='val_acc', verbose=1,patience = 30)
  
  # refit model and return
  print('Refitting model.')
  model.fit_generator(train_generator,epochs=250, validation_data=valid_generator, callbacks = [early_stopping_monitor])
  # save model weights to disk and clear from memory
  model.save_weights(checkpoint_dir)
  K.clear_session()

if __name__ == '__main__':
  main()
