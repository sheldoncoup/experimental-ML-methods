# Feature extraction scripts

Theses scripts perform feature extraction from images using pre-trained convolutional neural networks. Theses extracted features can be used in conjunction with classical classification techniques to provide a lightweight classification framework that is often surprisingly effective. All scripts can be run diretly from the terminal by passing arguments to them.

## feature_exraction.py

Used to extract features from a dataset of images.
### Args
--dataset_name    What the dataset is called, used for naming the output files. \
--data_dir        Where the data is saved. \
--feature_dir     Where to output the saved features. \
--ckpt_dir        What ckpt to use for feature extraction, if left blank the script will default to using imagenet weights. \
--extra_layer     Which extra layer to append to the feature output, if left blank only the pooled output of the final convolutional layer will be used as the extracted features \

* Additional model support coming soon *

## evaluate_classifiers_multiclass.py
For use when there is no test set defined. Imports the previously extracted features, creates a straified train/test split and evaluates the performance of a range of classifiers using the split. This is repeated a number of times (monte-carlo cross-validation) and the average accuracy of the classifiers is given.

--dataset_name    What the dataset is called, used for finding the feature files \\
--classifiers     Which classifiers to evaluate, leave blank to evaluate all of them. \\
--reduced_dims    What dimension to reduce the data to each turn, 128 dimensional by default (uses PCA) \\
--extra_layer     Name of the extra layer used for feature extraction. Default is None \\
--feature_dir     Where the feature files are saved. \\
--output_dir      Where to output the results. \\
--num_evals       How many monte-carlo evaluations to perform. Default 100 \\
--percent_test    What percent of the dataset to evaluate on. Default 0.1 \\
--rand_seed       What number to use for the random seed. \\



## single_eval_exp.py
Same as evaluate_classifiers_multiclass.py but used when there is a pre-defined test set to evaluate.

Needs fixing.
