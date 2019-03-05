"https://github.com/kwananth/VMWorkloadPredictor/blob/master/main_cpu.py"

import glob
import csv
import os
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

num_classes = 20
sample_history = 10


def load_data():
    path = "./ml_data/*.csv"
    data = []
    labels = []

    print("loading data...")

    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')

            for row in reader:
                r = np.array(row, dtype=float)
                rr = []
                for i in range(sample_history): ## taking only cpu readings to train model in predicting future reading.
                    rr.append(r[i * 7 + 1])
                # print(rr)
                data.append(rr)
                labels.append(r[-1])

    data = np.array(data)
    labels = np.array(labels)
    n = int(float(data.shape[0]) * 0.8)
    train_data = data[:n]
    train_labels = labels[:n]
    test_data = data[n:]
    test_labels = labels[n:]
    print("finished loading data")
    return train_data, train_labels, test_data, test_labels


# def grid_search(param_grid, clf, train_data, train_labels):
#     print("grid search...")
#     grid = GridSearchCV(clf, param_grid=param_grid)
#     grid.fit(train_data, train_labels)
#     print("Best: {0}".format(grid.best_estimator_))
#     print("end grid search...")
#     return grid.best_estimator_
#
#
# def plot_confusion_matrix(con_mat, nor=False):
#     print("plotting confusion matrix...")
#     if nor:
#         con_mat = con_mat.astype(float) / con_mat.sum(axis=0)
#     plt.imshow(con_mat, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion Matrix")
#     plt.colorbar()
#     plt.xticks(np.arange(num_classes), np.arange(num_classes))
#     plt.yticks(np.arange(num_classes), np.arange(num_classes))
#     fmt = '.2f' if nor else '.0f'
#     threshhold = con_mat.max() / 2.
#     for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
#         plt.text(j, i, format(con_mat[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if con_mat[i, j] > threshhold else "black")
#     plt.tight_layout()
#     plt.xlabel('True label')
#     plt.ylabel('Predicted label')
#     print("end of plotting.")
#
#
# def train(train_data, train_labels, test_data, test_labels):
#     print("starting training data... ")
#     param_grid = {
#         "alpha": [0.0001, 0.01, 0.1, 1]
#     }
#     clf = MLPClassifier()
#     clf = grid_search(param_grid, clf, train_data, train_labels)
#     print("Train accuracy = {}".format(clf.score(train_data, train_labels)))
#     print("Test accuracy = {}".format(clf.score(test_data, test_labels)))
#
#     # confusion matrix
#     test_pred = clf.predict(test_data)
#     con_mat = np.zeros((num_classes, num_classes))
#     for t, l in zip(test_pred, test_labels):
#         con_mat[int(t)][int(l)] += 1
#
#     print("Confusion Matrix = {}".format(con_mat))
#     plt.figure()
#     plot_confusion_matrix(con_mat, nor=True)
#     plt.show()

train_set, train_target, test_set, test_target = load_data()

from keras.models import Sequential
from keras.layers import Dense
## Building the model
NN_model = Sequential()

## Add layers to the network
# "https://keras.io/layers/core/" description of Dense function to add layers
NN_model.add(Dense(10, kernel_initializer='normal',input_dim = train_set.shape[1], activation='relu'))
NN_model.add(Dense(10, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

############
#NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train_set.shape[1], activation='relu'))

# # The Hidden Layers :
#NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#
# # The Output Layer :
#NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
#############

print(NN_model.summary())
#plot_model(NN_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

## Compile the network : To configure network for training
# "https://keras.io/models/model/#compile" description of 'compile' function.
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


## Train the model
# "https://keras.io/models/sequential/" description of 'fit' function which is used for model training.
model_training = NN_model.fit(train_set, train_target, epochs=50, batch_size=32, validation_split = 0.2)
print("Model Training History: \n" , model_training.history , "\n")

## Evaluate the
model_evaluation = NN_model.evaluate(test_set, test_target, batch_size=32 , verbose=1)
print("Model Evaluation: \n",NN_model.metrics_names,  model_evaluation,"\n")


## prediction
model_prediction_1 = NN_model.predict(np.array([[0.572777778 ,0.608333333, 0.571666667 , 0.592777778 , 0.59, 0.569444444, 0.572777778 , 4.341111111 , 70.36666667 , 53.13888889]])) # actual value = 92.60757937
print("model prediction : ", model_prediction_1)

print("one data example : ", train_set[10] , "  ," , train_target[10])
model_prediction_1 = NN_model.predict(np.array([train_set[10]])) # actual value = 92.60757937
print("model prediction : ", model_prediction_1)

# if __name__ == '__main__':
#     train_data, train_labels, test_data, test_labels = load_data()
#     #train(train_data, train_labels, test_data, test_labels)