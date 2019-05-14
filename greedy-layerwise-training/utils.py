
import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

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

def copy_file_list(file_list, target_data_dir):
  # Given a list of file directories, moves them into the given target directory 
  #Ensure the target exists
  if not os.path.exists(target_data_dir):
    os.mkdir(target_data_dir)
  
  for img_dir in file_list:
    classname = os.path.basename(os.path.dirname(img_dir))
    #img_name = os.path.basename(img_dir)
    new_img_dir = os.path.join(target_data_dir, classname)
    if not os.path.exists(new_img_dir):
      os.mkdir(new_img_dir)
    shutil.copy(img_dir, new_img_dir)
    
def move_data(source_data_dir, target_data_dir,train_size = 0.8, valid_size=0.1,  test_size = 0.1 ):
  #moves data into train/test/validation folders
  print('Moving images around to simplify data pipeline')
  total_size = test_size + valid_size + train_size
  # Get the list of classes and the list of image directories
  image_dir_list = recursive_save(source_data_dir)
  im_classes = os.listdir(source_data_dir)
  
  # Split the directory list into test, train, and validation sets
  train_dirs, test_valid_dirs = train_test_split(image_dir_list, test_size=(test_size+valid_size)/total_size, stratify=[os.path.basename(os.path.dirname(img_dir)) for img_dir in image_dir_list])
  valid_dirs, test_dirs = train_test_split(test_valid_dirs, test_size=test_size/(test_size+valid_size), stratify=[os.path.basename(os.path.dirname(img_dir)) for img_dir in test_valid_dirs])
  
  # Copy files
  copy_file_list(train_dirs, os.path.join(target_data_dir, 'train'))
  copy_file_list(valid_dirs, os.path.join(target_data_dir, 'test'))
  copy_file_list(test_dirs,  os.path.join(target_data_dir, 'validation'))
  print('Finished moving data.')
  


def main():
  # Get needed arguments to move data and execute
  parser = argparse.parser
  parser.add_argument('--source_data_dir', type=str, help='Where the dataset is saved.', required=True)
  parser.add_argument('--target_data_dir', type=str, help='Where the dataset is to be moved into.', required=True)
  parser.add_argument('--train_size', type=float, help='Percent of the dataset to use for training.' default = 0.8)
  parser.add_argument('--valid_size', type=float, help='Percent of the dataset to use for validation.' default = 0.1)
  parser.add_argument('--test_size', type=float, help='Percent of the dataset to use for testing.' default = 0.1)
  args = parser.parse_args()
  move_data(args.source_data_dir, args.target_data_dir,train_size = args.train_size, valid_size=args.valid_size,  test_size = args.test_size)

if __name__ == '__main__':
  main()
