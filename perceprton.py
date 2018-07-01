import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

  def __init__(self, n, epochs):
    self.weights = np.random.rand(n + 1)
    self.epochs = epochs
    self.lr = 0.001
  """
  Batch training of examples, on each iteration it adjust the weight vector.
  """
  def train(self, inputs, labels):
    # Add an extra column for the bias
    inputs = np.insert(inputs, 2 , 1, axis=1)
    for _ in range(self.epochs):
      for row,label in zip(inputs, labels):
        guess = self.activate(np.dot(self.weights, row.T))
        error = label - guess
        self.weights = self.weights + (self.lr * error * row) 

  """
  Test function to test how well the perceptron has learned.
  Outputs the percentage of correct answers.
  """
  def test(self, inputs, labels):
    inputs = np.insert(inputs, 2 , 1, axis=1)
    result = np.dot(inputs, self.weights)

    result[result >= 0] = 1
    result[result <  0] = -1

    correct = (result == labels)
    not_correct = (result != labels)

    plt.figure()
    plt.plot(inputs[correct, 0], inputs[correct,1], 'go')
    plt.plot(inputs[not_correct, 0], inputs[not_correct,1], 'ro')
    
    # Draw Margin
    x = np.linspace(-400, 600, 100)
    plt.plot(x, x, label='linear')

    percent = np.count_nonzero(correct) / result.size

    print("Test Accuracy ", percent * 100)

    plt.show(block=True)

  """
  Activation function representing the sign function.
  """    
  def activate(self, x):
    return 1 if x >=0 else -1 


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
Main Script
"""
ptron = Perceptron(2, 1000)
tr_data, tr_labels = create_data((128, 2))

ptron.train(tr_data, tr_labels)

ts_data, ts_labels = create_data((1024, 2))

ptron.test(ts_data, ts_labels)

