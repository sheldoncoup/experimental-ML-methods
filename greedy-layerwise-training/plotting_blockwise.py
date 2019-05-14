

#Required imports
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pickle
import os

def plot_phase(phase_num, model_name, log_dir, num_phases):
  figures = ['acc', 'loss']
  for figname in figures:
    # Setup of  figure
    color = iter(cm.rainbow(np.linspace(0,1,num_phases)))
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    
    # Iterate through phases adding lines for train and validation sets
    for i in range(phase_num + 1):
      file_dir = os.path.join(log_dir, model_name + '_phase_' + str(i)+ '.p')
      with open(file_dir, 'rb') as f:
        run_dict = pickle.load(f)
      num_epochs = len(run_dict['train_{}'.format(figname)])
      c = next(color)
      ax.plot(range(num_epochs), run_dict['train_{}'.format(figname)], '-',c=c, label='phase ' + str(i) + ' train')
      ax.plot( range(num_epochs),run_dict['valid_{}'.format(figname)], '--',c=c ,label='phase ' + str(i) + ' valid')
      plt.autoscale(tight=True)
      plt.legend()
    # Save figure  
    fig.savefig(os.path.join(log_dir,'{}_{}_.png'.format(model_name, figname)))
  
  plt.close('all')

def plot_ete(model_name, log_dir):
  figures = ['acc', 'loss']
  # Load in run data 
  file_dir = os.path.join(log_dir, '{}_end_to_end.p'.format(model_name))
  with open(file_dir, 'rb') as f:
    run_dict = pickle.load(f)
  num_epochs = len(run_dict['train_acc'])
  for figname in figures:
    #Setup for figure
    color = iter(cm.rainbow(np.linspace(0,1, 1)))
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
  
    # Plot data
    c = next(color)
    ax.plot(range(num_epochs), run_dict['train_{}'.format(figname)], '-',c=c, label='end to end train')
    ax.plot( range(num_epochs),run_dict['valid_{}'.format(figname)], '-',c=c ,label='end to end valid')
    plt.autoscale(tight=True)
    plt.legend()
  
    # Save figure  
    fig.savefig(os.path.join(log_dir,'{}_{}_ete.png'.format(model_name, figname)))
  
  plt.close('all')

def plot_both(phase_num, model_name, log_dir, num_phases):
  figures = ['acc', 'loss']
  for figname in figures:
    # Figure setup
    color = iter(cm.rainbow(np.linspace(0,1,num_phases+1)))
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    
    # Iterate through phases adding line for each phase
    for i in range(phase_num + 1):
      file_dir = os.path.join(log_dir, '{}_phase_{}.p'.format(model_name, i))
      with open(file_dir, 'rb') as f:
        run_dict = pickle.load(f)
      num_epochs = len(run_dict['train_{}'.format(figname)])
      c = next(color)
      ax.plot(range(num_epochs), run_dict['train_{}'.format(figname)], '-',c=c, label='phase {} train'.format(i))
      ax.plot(range(num_epochs), run_dict['valid_{}'.format(figname)], '--',c=c ,label='phase {} valid'.format(i))
    
    # Add line from end to end model
    file_dir = os.path.join(log_dir, model_name + '_end_to_end.p')
    with open(file_dir, 'rb') as f:
      run_dict = pickle.load(f)
    num_epochs = len(run_dict['train_{}'.format(figname)])
    c = next(color)
    ax.plot(range(num_epochs), run_dict['train_{}'.format(figname)], '-',c=c, label='end to end train')
    ax.plot( range(num_epochs),run_dict['valid_{}'.format(figname)], '--',c=c ,label='end to end valid')
  
    plt.autoscale(tight=True)
    plt.legend()
    
    # Save figure  
    fig.savefig(os.path.join(log_dir,'{}_{}_both.png'.format(model_name, figname)))
  
  plt.close('all')



def main():
  parser.add_argument('--log_dir', type=str, help='Where the output logs for the model trainings are saved', required=True)
  parser.add_argument('--model_name', type=str, help='Name of the model to plot, only "alexnet" and "inception_v4" currently allowed', default='alexnet')
  parser.add_argument('--plot_phase', type=bool, help='Whether or not to plot the phase training information', required=True)
  parser.add_argument('--plot_ete', type=bool, help='Whether or not to plot the phase training information', default=False)
  args = parser.parse_args()
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
  
  
