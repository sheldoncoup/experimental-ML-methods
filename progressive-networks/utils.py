
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


 
def plot_history(history, savename, use_validation):
  fig, ax1 = plt.subplots()
  
  acc  = history.history['acc']
  loss = history.history['loss']
  assert  len(acc) == len(loss)
  x = list(range(len(acc)))
  
  ax1.plot(x, acc, 'b-', label='Train Acc')
  if use_validation:
    ax1.plot(x, history.history['val_acc'], 'c-', label='Val Acc')
    
  ax1.set_xlabel('Epochs')
  # Make the y-axis label, ticks and tick labels match the line color.
  ax1.set_ylabel('Accuracy', color='b')
  ax1.tick_params('y', colors='b')
  
  ax2 = ax1.twinx()

  ax2.plot(x, loss, 'r-', label='Train_loss')
  if use_validation:
    ax2.plot(x, history.history['val_loss'], 'm-', label='Val Loss')
  ax2.set_ylabel('Loss', color='r')
  ax2.tick_params('y', colors='r')
  ax1.legend(loc='upper center', bbox_to_anchor=(0.25, -0.1),
          fancybox=True, shadow=True, ncol=2)
  ax2.legend(loc='upper center', bbox_to_anchor=(0.75, -0.1),
          fancybox=True, shadow=True, ncol=2)
  fig.tight_layout()
  plt.savefig('plot_output/{}.png'.format(savename))


