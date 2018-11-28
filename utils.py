
import os
import shutil

def copy_file_list(file_list, target_data_dir):
  # Given a list of file directories, moves them into the given target directory 
  if not os.path.exists(target_data_dir):
    os.mkdir(target_data_dir)
  
  for img_dir in file_list:
    classname = os.path.basename(os.path.dirname(img_dir))
    img_name = os.path.basename(img_dir)
    new_img_dir = os.path.join(target_data_dir, classname)
    if not os.path.exists(new_img_dir):
      os.mkdir(new_img_dir)
    shutil.copy(img_dir, new_img_dir)
    
def move_data(split_file_dir, target_data_dir ):
  #moves data into directories according to the given split file
  print('Moving images around to simplify data pipeline')
  with open(split_file_dir, 'rb') as f:
    split_dict = pickle.load(f)
  
  print('Shuffling training images')
  X_train = split_dict['X_train']
  y_train = split_dict['y_train']
  
  train_dir = os.path.join(target_data_dir, 'train')
  copy_file_list(X_train, train_dir)
  
  
  print('Shuffling test images')
  X_test = split_dict['X_test']
  y_test = split_dict['y_test']
  
  test_dir = os.path.join(target_data_dir, 'test')
  copy_file_list(X_test, test_dir)
  
  if 'X_valid' in split_dict.keys():
    print('Shuffling validation images')
    X_valid = split_dict['X_valid']
    y_valid = split_dict['y_valid']
  
    valid_dir = os.path.join(target_data_dir, 'valid')
    copy_file_list(X_valid, valid_dir)
    
  print('Image moving complete')
  
