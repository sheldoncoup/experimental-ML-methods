import keras
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import plot_history
from argparse import ArgumentParser
from keras.layers import Input, Concatenate, Conv2D, Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.utils.training_utils import multi_gpu_model
#from keras.utils import plot_model
from keras import optimizers
from keras.preprocessing import image



def construct_rider_model(trainable_model, frozen_model, layer_names_to_connect, num_classes,dense_layer_size = 1024):
  # check the models are the same
  assert len(trainable_model.layers) == len(frozen_model.layers)
  identical_layer_names = [x.name for x in frozen_model.layers if x.name in [y.name for y in trainable_model.layers]]
  # ensure the frozen model is actually frozen
  for layer in frozen_model.layers:
    layer.trainable = False
    
  # Extract list of the layers from the given network
  frozen_layers = [fl for fl in frozen_model.layers] 
  dynamic_layers = [dl for dl in trainable_model.layers]
  #layer_inputs = [ [tensor_input.name for tensor_input in tensor.input] for tensor in dynamic_layers]
  #print('Networks generated.')
  
  # Initialize storace dictionaries to work with functional api
  tensor_dict_frozen = {}
  tensor_dict_dynamic = {}
  
  
  # Initialize the new input layer
  rider_input = Input(shape = frozen_model.input_shape[1:])
  tensor_dict_frozen[frozen_layers[1].name] = frozen_layers[1](rider_input)
  tensor_dict_dynamic[dynamic_layers[1].name] =  dynamic_layers[1](rider_input)
  
  
  #debugging
  print(tensor_dict_frozen)
  n = 0
  
  for i in range(2, len(frozen_layers)):
    # select current layer and retreive which layers need to be input into it
    current_tensor_frozen = frozen_layers[i]
    current_tensor_dynamic = dynamic_layers[i]
    frozen_tensor_input  = current_tensor_frozen.input
    dynamic_tensor_input = current_tensor_dynamic.input
    
    # Figure out what other layers are referring to this current layer as
    frozen_tensor_outname = current_tensor_frozen.get_output_at(0).name.split('/')[0]
    dynamic_tensor_outname = current_tensor_dynamic.get_output_at(0).name.split('/')[0]
    
    #C heck if the tensor names will be doubled up and change the frozen layer name if need be
    if frozen_tensor_outname in identical_layer_names:
      current_tensor_frozen.name = '{}_fr'.format(frozen_tensor_outname)
 

    
    if not type(frozen_tensor_input) == list:
      #single input, non-concatenation layer
      
      # retreive the input layer from the dictionary and feed it into the current layer
      # store the layer output in the dictionary archive
      tensor_dict_frozen[frozen_tensor_outname] = current_tensor_frozen(tensor_dict_frozen[frozen_tensor_input.name.split('/')[0]])
      tensor_dict_dynamic[dynamic_tensor_outname] = current_tensor_dynamic(tensor_dict_dynamic[dynamic_tensor_input.name.split('/')[0]])
    else:
      # Multi-input layer, must be a concatenation layer, gather the inputs
      frozen_input_list = [tensor_dict_frozen[input_tensor.name.split('/')[0]] for input_tensor in frozen_tensor_input]
        
      dynamic_input_list = [tensor_dict_dynamic[input_tensor.name.split('/')[0]] for input_tensor in dynamic_tensor_input]
      
      # pass input list to frozen layer and store in archive
      tensor_dict_frozen[frozen_tensor_outname] = current_tensor_frozen(frozen_input_list)
      
      if current_tensor_frozen.name in layer_names_to_connect or current_tensor_dynamic.name in layer_names_to_connect:
        # The current layer is a being used to knit the models together, the archive name of the dynamic layer needs to be changed
        tensor_dict_dynamic['{}_dynamic_old'.format(dynamic_tensor_outname)] = current_tensor_dynamic(dynamic_input_list)
      else:
        # otherwise continue as normal
        tensor_dict_dynamic[dynamic_tensor_outname] = current_tensor_dynamic(dynamic_input_list)
    
    if current_tensor_frozen.name in layer_names_to_connect or current_tensor_dynamic.name in layer_names_to_connect:
      # The current tensor is one that is being used to knit the models together
      print('Stitching together layer {}.'.format(dynamic_tensor_outname))
      
      # concatenate the layers from both models together
      tensor_dict_dynamic['link_concat_{}'.format(n)] = Concatenate(name = 'link_concat_{}'.format(n))([tensor_dict_dynamic['{}_dynamic_old'.format(dynamic_tensor_outname)], 
                                                                                                                            tensor_dict_frozen[frozen_tensor_outname]])
      # feed concatenation layer to convolutional layer reducing the number of channels so its fits back into the model correctly
      tensor_dict_dynamic[dynamic_tensor_outname] = Conv2D( current_tensor_frozen.get_output_shape_at(0)[-1], (1,1))(tensor_dict_dynamic['link_concat_{}'.format(n)])
      n+=1
 
  
  # Build dense layers and predictions    
  x = GlobalAveragePooling2D()(tensor_dict_dynamic[dynamic_layers[-1].get_output_at(0).name.split('/')[0]])
  x = Dense(dense_layer_size, activation='relu')(x)
  predictions = Dense(num_classes, activation='softmax')(x)   
  
  # finaloze model
  rider_model = Model(inputs=rider_input, outputs=predictions)
  rider_model.summary()
  return rider_model

def get_args():
  # Manages arguments required for training
  parser = ArgumentParser()
  parser.add_argument('--dataset_name', type = str, help='Name of the dataset.', required=True)
  parser.add_argument('--target_data_train_dir', type = str, help='Directory where the target training data is saved.', required=True)
  parser.add_argument('--target_data_valid_dir', type = str, help='Directory where the target validation data is saved.')
  parser.add_argument('--output_dir', type=str, help='Directory where to save out model checkpoints and training history.', required=True)
  parser.add_argument('--model_to_train', type=str, help='Which model architecture to train, default is inceptionv3.', default='inceptionv3')
  parser.add_argument('--batch_size', type=int, help='Batch size to use for training.', default=64)
  parser.add_argument('--max_epochs', type = int, help='Maximum number of epochs to train on.', default=100)
  #parser.add_argument('--pretrain_epochs', type=int, help='How many epochs to pretrain for, default is 0, skips pretraining by default.', default = 0)
  parser.add_argument('--gpu_util', type=int, help='The number of gpus to use for training.', default = 1)
  parser.add_argument('--augment_images', type=bool, help='Whether or not to perform image augmentation when generating data. Turned off by default', default=False)
  parser.add_argument('--optimizer', type=str, help='Which optimizer to use, default is Adadelta.', default='adadelta')
  parser.add_argument('--lr', type=float, help='What learning rate to use.')
  
  args = parser.parse_args()
  return args

def get_base_models(args):
  if args.model_to_train == 'inceptionv3':
    # intialize separate models
    frozen_model = InceptionV3(include_top = False, weights='imagenet', input_shape=(299,299,3))
    trainable_model = InceptionV3(include_top = False, weights='imagenet', input_shape=(299,299,3))
    # set which layers to use as links between models
    reminder_layers = ['mixed0', 'mixed1', 'mixed2','mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9', 'mixed10'] 
  else:
    raise ValueError( 'Model {} not currently supported.'.format(args.model_to_train))
  return frozen_model, trainable_model, reminder_layers

def get_optimizer(args):
  # returns an optimiser with the given learning rate
  if args.optimizer == 'adadelta':
    if args.lr == None:
      opt  = optimizers.Adadelta()
    else:
      opt  = optimizers.Adadelta(lr=args.lr)
  elif args.optimizer == 'adam':
    if args.lr == None:
      opt  = optimizers.Adam()
    else:
      opt  = optimizers.Adam(lr=args.lr)
  else:
    raise ValueError('{} is not a valid optimizer.'.format(args.optimizer))
  return opt  

def main():
  args = get_args()
  
  frozen_model, trainable_model, reminder_layers = get_base_models(args)
  num_classes = len(os.listdir(args.target_data_train_dir))
  
  if args.augment_images:
    train_datagen = image.ImageDataGenerator(rescale = 1./255,
                                             rotation_range=40,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True,
                                             fill_mode='nearest',
                                             validation_split = 0.1 if args.target_data_valid_dir == None else 0.0 )
  else:
    train_datagen = image.ImageDataGenerator(rescale = 1./255, validation_split = 0.1 if args.target_data_valid_dir == None else 0.0)
  
  train_data = train_datagen.flow_from_directory(args.target_data_train_dir,target_size=(299,299), shuffle=True, batch_size = args.batch_size, subset='training')
  
  # if validation directory not set use some training image for validation
  if args.target_data_valid_dir == None:
    valid_data = train_datagen.flow_from_directory(args.target_data_train_dir, target_size = (299,299),shuffle=True, batch_size = args.batch_size, subset='validation')
  
  # construct rider model
  rider_model = construct_rider_model(trainable_model,frozen_model, reminder_layers, num_classes )
  
  if args.gpu_util > 1:
    # Employs multiple gpus for training if specified 
    model = multi_gpu_model(model, gpus=args.gpu_util)
  
  
  rider_model.compile('Adadelta',loss='categorical_crossentropy', metrics=['accuracy'])
  
  if not args.target_data_valid_dir == None:
    valid_datagen = image.ImageDataGenerator(rescale=1./255)
    valid_data = valid_datagen.flow_from_directory(args.target_data_valid_dir,target_size=(299,299), batch_size=args.batch_size)
    history = rider_model.fit_generator(train_data, epochs=args.max_epochs,validation_data=valid_data)
  else:
    history = rider_model.fit_generator(train_data, epochs=args.max_epochs,validation_data=valid_data )
  
  # Save all of the data for the run
  run_name = '{}_{}_rider_network_{}_total_{}_batch'.format(args.dataset_name, args.model_to_train, args.max_epochs, args.batch_size)
  
  if args.gpu_util > 1:
    # if using a multi gpu model, extract the appropriate layers
    rider_model = model.get_layer('model_1') 
    
  rider_model.save(os.path.join(args.output_dir, '{}_ckpt.h5'.format(run_name)))
  try:
    pickle.dump(dict(history.history), open(os.path.join(args.output_dir,'{}_history.p'.format(run_name)), 'wb'))
    print('Successfully dumped history file')
  except:
    print('History dump failed')
  plot_history(history, os.path.join(args.output_dir,run_name), True)


if __name__ == '__main__':
  main()
  
      
      
    
