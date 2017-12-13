import numpy as np

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 32, dim_y = 32, batch_size = 32, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.batch_size = batch_size
      self.shuffle = shuffle

  def generate(self, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(labels, list_IDs_temp)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))
      y = np.empty((self.batch_size))

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store volume
          X[i, :, :, 0] = np.load(ID)

          # Store class
          y[i] = label_from_id(ID)

      return X, y

all_labels = ['ECONOMICS', 'ENTERTAINMENT', 'POLITICS', 'SPORTS', 'TECHNOLOGY']

def label_from_id(filename):
    label_text = filename.split('-')[1]
    labels = label_text.replace('.npy', '').split('_')
    y = np.zeros(len(all_labels))
    for label in labels:
        y[all_labels.index(label)] = 1
    return y
