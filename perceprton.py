import numpy as np

class Perceptron:

  def __init__(self, n):
    self.weights = np.random.rand(n)
    self.lr = 0.001

  """
  Trains a single example and adjusts the weights accordingly.
  """
  def train(self, inputs, label):
    guess = self.activate(np.dot(self.weights, inputs.T))
    error = label - guess

    self.weights = self.weights + (self.lr * error * inputs)

  """
  Test function to test how well the perceptron has learned.
  Outputs the percentage of correct answers.
  """
  def test(self, inputs, labels):
    result = np.dot(inputs[:, :2], self.weights)

    result[result >= 0] = 1;
    result[result < 0] = -1;

    percent = np.count_nonzero(result == labels) / result.size

    print("Test Accuracy ", percent * 100)

  """
  Activation function representing the sign function.
  """    
  def activate(self, sum):
    if sum >=0:
      return 1
    else:
      return -1 


"""
Creates a test/train data set when given a specific data size.
Returns the labels based on whether the input was above the margin x = y.
"""
def create_data(size):
  data = np.random.randint(-400, 600, size)
  labels = np.zeros(size[0])
  idxs = data[:,0] >= data[:,1]
  labels[idxs] =  1
  idxs = data[:,0] < data[:,1]
  labels[idxs] = -1

  return data,labels

"""
Main script code lies here for testing
"""
ptron = Perceptron(2)
tr_data, tr_labels = create_data((128, 2))

for x in range(1000):
  for inputs,label in zip(tr_data,tr_labels):
    ptron.train(inputs, label)

ts_data, ts_labels = create_data((2048, 2))
ptron.test(ts_data, ts_labels)
