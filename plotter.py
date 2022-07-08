import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

class Plotter(object):
  # Calculate the mean of several files
  def determine_mean_plotting_data(self, data, data_dict, variable, **kwargs):
      # Empty variables
      # Empty variables
      samples = []
      sum_samples = 0
      data_mean = None
      confidence_intervall=0
      
      # Iterate over all paths of one algorithm and add up the different runs
      for sample in data:
          if data_mean is None:
              data_mean= np.zeros(len(sample[variable]))
          data_mean+=(sample[variable])
          samples.append(sample[variable])

      # Determine the mean
      data_mean/=len(data)

      # Calculate the 95% confidence intervall
      # Formula taken from https://web.stat.tamu.edu/~suhasini/teaching301/stat301CI_t-dist.pdf
      for sample in samples:
          sum_samples+=(sample-data_mean)**2
      deviation=np.sqrt(sum_samples/(len(data)-1))
      # 1.812 for 10 used samples from t table https://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf
      # 1.68 for 50 samples from https://faculty.washington.edu/heagerty/Books/Biostatistics/TABLES/t-Tables/
      ## ATTENTION change value for other sample numbers to get 95 % confidence intervall!!
      confidence_intervall =  1.68 * deviation / (len(data))**(1/2)  

      # Create Dictionary for further process
      data_dict[variable] = data_mean

      data_dict[variable + '_confidence_intervall'] = confidence_intervall

      return data_dict

  # Plot different metrics over the epochs
  def plot_loss_and_accuracy(self, data, data_description, **kwargs):
    epochs = [i for i in range(len((data['loss'])))]
    train_acc = data['accuracy']
    train_loss = data['loss']
    test_acc = data['val_accuracy']
    test_loss = data['val_loss']
    conf_train_loss= data['loss_confidence_intervall']
    conf_test_loss = data['val_loss_confidence_intervall']
    conf_train_acc = data['accuracy_confidence_intervall']
    conf_test_acc = data['val_accuracy_confidence_intervall']
    x_axis = np.arange(0, len(train_acc), 1)
    
    plt.plot(epochs , train_loss , label = 'Training Loss', color='blue')
    plt.fill_between(x_axis, train_loss+conf_train_loss, train_loss-conf_train_loss, color='blue', alpha=.1)

    plt.plot(epochs , test_loss , label = 'Validation Loss',color='orange')
    plt.fill_between(x_axis, test_loss+conf_test_loss, test_loss-conf_test_loss,color='orange', alpha=.1)
    plt.title(data_description + ' Training and Validation Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(data_description+'_Training_and_Validation_Loss.png')
    plt.show()

    plt.plot(epochs , train_acc , label = 'Training Accuracy', color='blue')
    plt.fill_between(x_axis, train_acc+conf_train_acc, train_acc-conf_train_acc,color='blue', alpha=.1)
    plt.plot(epochs , test_acc , label = 'Validation Accuracy',color='orange')
    plt.fill_between(x_axis, test_acc+conf_test_acc, test_acc-conf_test_acc,color='orange', alpha=.1)
    plt.title(data_description + ' Training and Validation Accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig(data_description+'_Training_and_Validation_Accuracy.png')
    plt.show()

  def __init__(self, file_path, **kwargs):
    self.file_path = file_path

  def __call__(self, **kwargs):
    with open(self.file_path, "rb") as f:
        data = pickle.load(f)
    plotting_data = dict()
    plotting_data = self.determine_mean_plotting_data(data, plotting_data, 'accuracy', **kwargs)
    plotting_data = self.determine_mean_plotting_data(data, plotting_data, 'val_accuracy', **kwargs)
    plotting_data = self.determine_mean_plotting_data(data, plotting_data, 'loss', **kwargs)
    plotting_data = self.determine_mean_plotting_data(data, plotting_data, 'val_loss', **kwargs)
    self.plot_loss_and_accuracy(plotting_data, **kwargs)

def parse_args():

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--file_path', type=str, default='audio.dat')
  parser.add_argument('--data_description', type=str, default='')
  args = parser.parse_args()
  dict_args = vars(args)

  return dict_args


if __name__ == '__main__':
    args=parse_args()
    plotting = Plotter(**args)
    plotting(**args)
