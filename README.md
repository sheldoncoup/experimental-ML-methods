# phasenets
  Phasenets inspired by "Effective training of convolutional neural ntworks with small, specialized datasets" -  Plata, Diego Ruedaa, Ramos-Pollán, Raúla, González, Fabio ( https://content.iospress.com/articles/journal-of-intelligent-and-fuzzy-systems/ifs169131 )

Phasenets are identical to standard Convolutional neural networks in terms of how they function and how they are designed.
The key difference here is the method my which the networks are trained. In this code, each network (only Alexnet and InceptionV4 currently finished but more in the future hopefully!) is split into 5 blocks such that in the overall architecture the output of black 1 is the input of block 2 and so on.
When the network is trained on a new dataset, initally only the first block is trained. Once the learning has finished for this training the weights are frozen for this block and the second block is added. Training then continues with only the weights of the block 2 being changed.
This pattern continues until the entire originial network is built back up.

This code utilizes the Keras Deep Learning Library run on top of TensorFlow.

Training is done with the train_blockwise.py script with the following arguments:
 
 Required arguments:
 --dataset_name Name of the dataset to be evaluated. //
 --num_classes Number of classes in the dataset. //
 --data_dir Directory where the data is stored. 
 
 Optional arguments:
 --image_size Desired side length of the input images.  Default=227 //
 --model_name Name of the model to train on, only "alexnet" and "inception_v4" currently allowed. Default=alexnet //
 --batch_size Number of images per batch for training. Default=32 //
 --ckpt_dir  Directory of the fine tuned checkpoint. Default=None //
 --num_gpu Number of gpus to use for training. Default = 1 //
 --start_phase Phase to start training at, only non zero if resuming from a previous run. Default = 0 //
 --freeze_layers Wherether of not phase training should be run, set false for standard training. Default = True //
 --optimizer Optimizer choice: Adadelta or Adam. Default = 'Adadelta' //
 --lr Learning rate to use,  Default = Whatever the optimizer default value is //
 --dr Decay rate to use. Default = 0 //
 --max_epoch Maximum number of epochs to train for on each run Default = 100 //
 --use_test Whether or not to evaluate the test set. Default=False //
 --rand_seed What seed to use for the random initializations. Default=1 
 
 The script expects the data to all be contained in --data_dir, where data_dir contains two (or three) subdirectories names, train, validation (and test). Within each of these directories each class should have a its own subdirectory containing the images from that class.

Inception V4 architecture coded by Kent Sommer (https://github.com/kentsommer/keras-inceptionV4)
