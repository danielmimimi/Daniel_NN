from numpy import ndarray
import numpy as np
from layer.conv2D import Conv2D

from layer.dense import Dense
from loss.mse import MeanSquaredError
from loss.softmaxCrossentropy import SoftmaxCrossEntropy
from network.neuronalNetwork import NeuralNetwork
from operations.linear import Linear
from operations.sigmoid import Sigmoid
from operations.tanh import Tanh
from optimizer.sgd import SGD
from optimizer.sgdMomentum import SGDMomentum
from trainer.trainer import Trainer

from utils import mnist

X_train, y_train, X_test, y_test = mnist.load()

num_labels = len(y_train)
print(num_labels)


# one-hot encode
num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1



X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
print(np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test))

X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)
print(np.min(X_train), np.max(X_train), np.min(X_test), np.max(X_test))

X_train_conv, X_test_conv = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)

def calc_accuracy_model(model, test_set):
    return print(f'''The model validation accuracy is: 
    {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')

model = NeuralNetwork(
    layers=[Conv2D(out_channels=16,
                   param_size=5,
                   dropout=0.8,
                   weight_init="glorot",
                   flatten=True,
                  activation=Tanh()),
            Dense(neurons=10, 
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(), 
seed=20190402)
trainer = Trainer(model, SGDMomentum(lr = 0.1, momentum=0.9))
trainer.fit(X_train_conv, train_labels, X_test_conv, test_labels,
            epochs = 1,
            eval_every = 1,
            seed=20190402,
            batch_size=60,
            conv_testing=True)
print()
calc_accuracy_model(model, X_test)