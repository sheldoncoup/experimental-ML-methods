

#Required imports
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pickle
import os

def get_args():
  #Function for getting use input argument for when just the plotting is taking place
  parser.add_argument('--log_dir', type=str, help='Where the output logs for the model trainings are saved', required=True)
  parser.add_argument('--model_name', type=str, help='Name of the model to plot, only "alexnet" and "inception_v4" currently allowed', default='alexnet')
  parser.add_argument('--plot_phase', type=bool, help='Whether or not to plot the phase training information', required=True)
  parser.add_argument('--plot_ete', type=bool, help='Whether or not to plot the phase training information', default=False)
  args = parser.parse_args()
  return args


def plot_phase(phase_num, model_name, log_dir, num_phases):
  color = iter(cm.rainbow(np.linspace(0,1,num_phases)))
  fig = plt.figure(figsize=(20,20))
  ax = fig.add_subplot(111)
  for i in range(phase_num + 1):
    file_dir = os.path.join(log_dir, model_name + '_phase_' + str(i)+ '.p')
    with open(file_dir, 'rb') as f:
      run_dict = pickle.load(f)
    num_epochs = len(run_dict['train_acc'])
    c = next(color)
    ax.plot(range(num_epochs), run_dict['train_acc'], '-',c=c, label='phase ' + str(i) + ' train')
    ax.plot( range(len(run_dict['valid_acc'])),run_dict['valid_acc'], '--',c=c ,label='phase ' + str(i) + ' valid')
    plt.autoscale(tight=True)
    plt.legend()
    
  fig.savefig(model_name + ' acc.png')
  color=iter(cm.rainbow(np.linspace(0,1,num_phases)))
  
  fig = plt.figure(figsize=(20,20))
  ax = fig.add_subplot(111)
  
  for i in range(phase_num +1):
    file_dir = os.path.join(log_dir, model_name + '_phase_' + str(i)+ '.p')
    with open(file_dir, 'rb') as f:
      run_dict = pickle.load(f)
    num_epochs = len(run_dict['train_loss'])
    c = next(color)
    ax.plot(range(num_epochs), run_dict['train_loss'], '-',c=c, label='phase ' + str(i) + ' train')
    ax.plot(range(len(run_dict['valid_loss'])),run_dict['valid_loss'], '--', c=c,label='phase ' + str(i) + ' valid')
    plt.autoscale(tight=True)
    plt.legend()
  
  
  fig.savefig(model_name + 'loss.png')
  plt.close('all')

def plot_ete(model_name, log_dir):
  color = iter(cm.rainbow(np.linspace(0,1, 1)))
  fig = plt.figure(figsize=(20,20))
  ax = fig.add_subplot(111)
  file_dir = os.path.join(log_dir, model_name + '_end_to_end.p')
  with open(file_dir, 'rb') as f:
    run_dict = pickle.load(f)
  num_epochs = len(run_dict['train_acc'])
  c = next(color)
  ax.plot(range(num_epochs), run_dict['train_acc'], '-',c=c, label='end to end train')
  ax.plot( range(len(run_dict['valid_acc'])),run_dict['valid_acc'], '-',c=c ,label='end to end valid')
  plt.autoscale(tight=True)
  plt.legend()
    
  fig.savefig(model_name + ' acc_ete.png')
  color=iter(cm.rainbow(np.linspace(0,1,num_phases)))
  
  fig = plt.figure(figsize=(20,20))
  ax = fig.add_subplot(111)
  

  file_dir = os.path.join(log_dir, model_name + '_end_to_end.p')
  with open(file_dir, 'rb') as f:
    run_dict = pickle.load(f)
  num_epochs = len(run_dict['train_loss'])
  c = next(color)
  ax.plot(range(num_epochs), run_dict['train_loss'], '-',c=c, label='end to end train')
  ax.plot( range(len(run_dict['valid_loss'])),run_dict['valid_loss'], '-', c=next(color),label='end to end valid')
  plt.autoscale(tight=True)
  plt.legend()
  
  
  fig.savefig(model_name + 'loss_ete.png')
  plt.close('all')

def plot_both(phase_num, model_name, log_dir, num_phases):
  color = iter(cm.rainbow(np.linspace(0,1,num_phases+1)))
  fig = plt.figure(figsize=(20,20))
  ax = fig.add_subplot(111)
  for i in range(phase_num + 1):
    file_dir = os.path.join(log_dir, model_name + '_phase_' + str(i)+ '.p')
    with open(file_dir, 'rb') as f:
      run_dict = pickle.load(f)
    num_epochs = len(run_dict['train_acc'])
    c = next(color)
    ax.plot(range(num_epochs), run_dict['train_acc'], '-',c=c, label='phase ' + str(i) + ' train')
    ax.plot( range(len(run_dict['valid_acc'])),run_dict['valid_acc'], '--',c=c ,label='phase ' + str(i) + ' valid')
  
  file_dir = os.path.join(log_dir, model_name + '_end_to_end.p')
  with open(file_dir, 'rb') as f:
    run_dict = pickle.load(f)
  num_epochs = len(run_dict['train_acc'])
  c = next(color)
  ax.plot(range(num_epochs), run_dict['train_acc'], '-',c=c, label='end to end train')
  ax.plot( range(len(run_dict['valid_acc'])),run_dict['valid_acc'], '--',c=c ,label='end to end valid')
  
  plt.autoscale(tight=True)
  plt.legend()
    
  fig.savefig(model_name + ' acc.png')
  
  color=iter(cm.rainbow(np.linspace(0,1,num_phases+1)))
  
  fig = plt.figure(figsize=(20,20))
  ax = fig.add_subplot(111)
  
  for i in range(num_phases):
    file_dir = os.path.join(log_dir, model_name + '_phase_' + str(i)+ '.p')
    with open(file_dir, 'rb') as f:
      run_dict = pickle.load(f)
    num_epochs = len(run_dict['train_loss'])
    c = next(color)
    ax.plot(range(num_epochs), run_dict['train_loss'], '-',c=c, label='phase ' + str(i) + ' train')
    ax.plot(range(len(run_dict['valid_loss'])),run_dict['valid_loss'], '--', c=c,label='phase ' + str(i) + ' valid')
  
  file_dir = os.path.join(log_dir, model_name + '_end_to_end.p')
  with open(file_dir, 'rb') as f:
    run_dict = pickle.load(f)
  num_epochs = len(run_dict['train_loss'])
  c = next(color)
  ax.plot(range(num_epochs), run_dict['train_loss'], '-',c=c, label='end to end train')
  ax.plot( range(len(run_dict['valid_loss'])),run_dict['valid_loss'], '--', c=c,label='end to end valid')
  
  plt.autoscale(tight=True)
  plt.legend()
  
  
  fig.savefig(model_name + 'loss.png')
  plt.close('all')



def main():
  args = get_args()
  if args.plot_ete and args.plot_:
    plot_both(4, args.model_name, args.log_dir, 5):
  elif args.plot_ete :
    plot_ete(args.model_name, args.log_dir)
  elif args.plot_phase:
    plot_phase(4, args.model_name, args.log_dir, 5))
  else:
    raise ValueError( 'You did not select something to plot!')

if __name__ == '__main__':
  main()
  
  
