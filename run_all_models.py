# Model 1 - KNN

# imports 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
from numpy import asarray
import pandas as pd
from pathlib import Path
import numpy as np

loaded = np.load('x_data.npz')
x_data = loaded['data']

# loaded = np.load('y_target.npz')
loaded = np.load('y_target.npz')
y_target = loaded['data']

# normalize pixels
x_data = x_data/255

# shuffle the dataframe then split 70% training 30% test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_target, test_size=0.3,shuffle=True)

from collections import Counter
def get_distance(a, b):
    # return np.sqrt(np.sum((b-a) ** 2))
    return np.sum((b - a) ** 2)

class KNN_Classifier():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    def predict(self, new_points):
        predictions = [self.predict_class(new_point) for new_point in new_points]
        return np.array(predictions)
    def predict_class(self, new_point):
        distances = [get_distance(point, new_point) for point in self.x_train]
        k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common


# KNN - 1
model = KNN_Classifier(1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
accuracy = np.mean(predictions == y_test) * 100
print(f'accuracy: {accuracy}%')

print(classification_report(y_test, predictions))

# KNN - 3
model = KNN_Classifier(3 )
model.fit(x_train, y_train)
predictions = model.predict(x_test)
accuracy = np.mean(predictions == y_test) * 100
print(f'accuracy: {accuracy}%')

print(classification_report(y_test, predictions))

# KNN - 5
model = KNN_Classifier(5)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
accuracy = np.mean(predictions == y_test) * 100
print(f'accuracy: {accuracy}%')

print(classification_report(y_test, predictions))



# LINEAR CLASSIFIER PERCEPTRON
import numpy as np
from sklearn.model_selection import train_test_split

# loaded = np.load("x_short_data.npz")
loaded = np.load("x_data.npz")
x_data = loaded["data"]

# loaded = np.load("y_short_target.npz")
loaded = np.load("y_target.npz")

y_target = loaded["data"]

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)  # [0..9]
        n_classes = len(self.classes_)
        self.w_ = rgen.normal(
            loc=0.0, scale=0.01, size=(X.shape[1], n_classes)
        )  # (784, 10)
        self.b_ = np.zeros(n_classes)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for i in rgen.permutation(len(X)):  # shuffle each epoch
                xi, target = X[i], y[i]
                pred = self.predict(xi)
                if pred != target:
                    # reward the correct class, penalize the wrong guess
                    self.w_[:, target] += self.eta * xi
                    self.b_[target] += self.eta
                    self.w_[:, pred] -= self.eta * xi
                    self.b_[pred] -= self.eta
                    errors += 1
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input (one score per class)"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return the class with the highest score"""
        return np.argmax(self.net_input(X), axis=-1)

# normalize pixels
x_data = x_data / 255
# shuffle the dataframe then split 70% training 30% test
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_target, test_size=0.3, shuffle=True
)

ppn = Perceptron(eta=0.1, n_iter=10).fit(x_train, y_train)
print("accuracy:", np.mean(ppn.predict(x_test) == y_test))

y_pred = ppn.predict(x_test)   # shape (n_samples,), since argmax over axis=-1
print(classification_report(y_test, y_pred))


# NAIVE BAYES
loaded = np.load("x_data.npz")
x_data = loaded["data"]

loaded = np.load("y_target.npz")
y_target = loaded["data"]

# normalize
x_data = x_data / 255
# y_target = y_target.ravel()

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_target, test_size=0.3, shuffle=True
)

class NaiveBayes:
    def fit(self, x_train, y_train):
        x_train = (np.asarray(x_train) > 0.5).astype(np.float64)
        y_train = np.asarray(y_train).ravel()

        self.classes = np.unique(y_train)
        class_counts = np.array([np.sum(y_train == c) for c in self.classes])

        # Laplace smoothing for the probability that each pixel is on.
        self.feature_probs = np.array(
            [
                (x_train[y_train == c].sum(axis=0) + 1) / (class_counts[i] + 2)
                for i, c in enumerate(self.classes)
            ]
        )
        self.log_priors = np.log(class_counts / len(y_train))

    def predict(self, x):
        x = (np.asarray(x) > 0.5).astype(np.float64)

        log_probs = (
            x @ np.log(self.feature_probs).T
            + (1 - x) @ np.log(1 - self.feature_probs).T
            + self.log_priors
        )
        return self.classes[np.argmax(log_probs, axis=1)]

model = NaiveBayes()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(classification_report(y_test, predictions))


# MULTILAYER PERCEPTRON
loaded = np.load("x_short_data.npz")
x_data = loaded["data"]

loaded = np.load("y_short_target.npz")
y_target = loaded["data"]

# normalize
x_data = x_data / 255

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_target, test_size=0.3, shuffle=True
)

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

class NeuralNetMLP:
    def __init__(self, num_features, num_hidden,
                 num_classes, random_seed=123):
        self.num_classes = num_classes

        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):

        y_onehot = int_to_onehot(y, self.num_classes)

        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]

        d_a_out__d_z_out = a_out * (1. - a_out) # sigmoid derivative

        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_z_out__dw_out = a_h

        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        d_z_out__a_h = self.weight_out

        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        d_a_h__d_z_h = a_h * (1. - a_h) # sigmoid derivative

        d_z_h__d_w_h = x

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T,
                                d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)
        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h, d_loss__d_b_h)


model = NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)

num_epochs = 50
minibatch_size = 100

def minibatch_generator(x, y, minibatch_size):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield x[batch_idx], y[batch_idx]


def mse_loss(targets, probas,num_labels=10):
    onehot_targets = int_to_onehot(
        targets, num_labels=num_labels
    )
    return np.mean((onehot_targets - probas)**2)

def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

_, probas = model.forward(x_test)
mse = mse_loss(y_test, probas)
print(f'Initial validation MSE:  {mse:.1f}')

predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_test, predicted_labels)
print(f'Initial validation accuracy: {acc*100:.1f}%')

def compute_mse_and_acc(nnet, x, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(x, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    mse = mse/i
    acc = correct_pred/num_examples
    return mse, acc

mse, acc = compute_mse_and_acc(model, x_test, y_test)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc*100:.1f}%')

def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    for e in range(num_epochs):
        # iterate over minibatches
        minibatch_gen = minibatch_generator(
            X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            #### Compute outputs ####
            a_h, a_out = model.forward(X_train_mini)
            #### Compute gradients ####
            d_loss__d_w_out, d_loss__d_b_out, \
            d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out,
                               y_train_mini)
            #### Update weights ####
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        
        #### Epoch Logging ####
        train_mse, train_acc = compute_mse_and_acc(
            model, X_train, y_train
        )
        valid_mse, valid_acc = compute_mse_and_acc(
            model, X_valid, y_valid
        )
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')
    return epoch_loss, epoch_train_acc, epoch_valid_acc

np.random.seed(123) # for the training set shuffling
epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, x_train, y_train, x_test, y_test,num_epochs=50, learning_rate=0.1)

_, probas = model.forward(x_test)
predictions = np.argmax(probas, axis=1)

print(classification_report(y_test, predictions))

# CONVOLUTIONAL NEURAL NETWORK
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import TensorDataset, DataLoader

# loaded = np.load('x_short_data.npz')
loaded = np.load("x_data.npz")
x_data = loaded['data']

# loaded = np.load('y_short_target.npz')
loaded = np.load("y_target.npz")
y_target = loaded["data"]

x_data = x_data / 255

x_train, x_test, y_train, y_test = train_test_split(x_data, y_target, test_size=0.3,shuffle=True)

x_train_t = torch.as_tensor(x_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
x_test_t = torch.as_tensor(x_test, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train_t = torch.as_tensor(y_train, dtype=torch.long)
y_test_t = torch.as_tensor(y_test, dtype=torch.long)

train_dl = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=64, shuffle=True)
valid_dl = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=64)

model = nn.Sequential()
model.add_module(
    'conv1',
    nn.Conv2d(
        in_channels=1,
        padding=2,
        kernel_size=5,
        out_channels=32
    )
)

model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module(
    'conv2',
    nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=5,
        padding=2
    )
)
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))

model.add_module('flatten', nn.Flatten())
x = torch.ones((4,1,28,28))

model.add_module('fc1', nn.Linear(3136,1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.5))
model.add_module('fc2', nn.Linear(1024, 10))

loss_fn = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (
                torch.argmax(pred, dim=1) == y_batch
            ).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        model.eval()
        
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (
                    torch.argmax(pred,dim=1) == y_batch
                ).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
                
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        print(f'Epoch {epoch+1} accuracy: ')
        print(f'{accuracy_hist_train[epoch]:.4f} val_accuracy: ')
        print(f'{accuracy_hist_valid[epoch]:.4f}')
        
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

torch.manual_seed(1)
num_epochs = 20
hist = train(model, num_epochs, train_dl, valid_dl)