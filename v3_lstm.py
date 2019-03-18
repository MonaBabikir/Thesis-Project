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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


num_classes = 20
sample_history = 10

def plotting_cpu_timeseries():
    path = "./output/*.csv"

    cpu_avg = []
    timestep = []
    i = 0
    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')
            for row in reader:
                r = np.array(row, dtype=float)
                cpu_avg.append(r[1])
                timestep.append(r[0])

        plt.plot(timestep, cpu_avg)
        plt.xlabel("timestamp")
        plt.ylabel("cpu_avg")
        plt.title(" cpu avg vs timestamp")
        #plt.savefig("cpu_timing.png")
        plt.show()
        i+=1
        if i > 2:
            break



def load_data(): #"https://github.com/kwananth/VMWorkloadPredictor/blob/master/main_cpu.py"
    path = "./ml_data_classes/*.csv"
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

####################33
def load_data_random(): #"https://github.com/kwananth/VMWorkloadPredictor/blob/master/main_cpu.py"
    path = "./ml_data_classes/*.csv"
    data = []
    #labels = []
    print("loading data...")

    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')

            for row in reader:
                r = np.array(row, dtype=float)
                rr = []
                for i in range(sample_history): ## taking only cpu readings to train model in predicting future reading.
                    rr.append(r[i * 7 + 1])
                    # for j in range(6):
                    #     rr.append(r[i * 7 + 1 + j])
                #print(rr)
                rr.append(r[-1])
                data.append(rr)
                #labels.append(r[-1])

    data = np.array(data)
    data = pd.DataFrame(data=data)
    print(data.shape, "  ", list(data))
    data.columns = ['cpu_1', 'cpu_2', 'cpu_3', 'cpu_4', 'cpu_5', 'cpu_6', 'cpu_7', 'cpu_8', 'cpu_9', 'cpu_10', 'class']
    # data.columns = ['in_1', 'in_2', 'in_3','in_4', 'in_5', 'in_6', 'in_7', 'in_8','in_9',
    #                 'in_10','in_11', 'in_12', 'in_13','in_14', 'in_15', 'in_16', 'in_17', 'in_18','in_19', 'in_20',
    #                 'in_21', 'in_22', 'in_23','in_24', 'in_25', 'in_26', 'in_27', 'in_28','in_29',
    #                 'in_30','in_31', 'in_32', 'in_33','in_34', 'in_35', 'in_36', 'in_37', 'in_38','in_39', 'in_40',
    #                 'in_41', 'in_42', 'in_43','in_44', 'in_45', 'in_46', 'in_47', 'in_48','in_49',
    #                 'in_50','in_51', 'in_52', 'in_53','in_54', 'in_55', 'in_56', 'in_57', 'in_58','in_59', 'in_60','class']

    train_data, test_data = train_test_split(data, test_size=0.2)

    print("sample : ", train_data.iloc[50])
    train_labels = train_data["class"]
    train_data.drop(["class"], axis=1, inplace=True)

    test_labels = test_data["class"]
    test_data.drop(["class"], axis=1, inplace=True)

    print("input features : ", list(train_data))
    print("finished loading data")
    return train_data, train_labels, test_data, test_labels

####################
#plotting_cpu_timeseries()

train_set, train_target, test_set, test_target = load_data()
#train_set, train_target, test_set, test_target =load_data_random()

### scaling data
scaler1 = StandardScaler()
scaler2 = StandardScaler()
train_set = scaler1.fit_transform(train_set)
test_set = scaler2.fit_transform(test_set)

### end of scaling


##### reshaping the data for LSTM network [samples , timestep, features]
train_set = np.reshape(train_set, (train_set.shape[0] , 1, train_set.shape[1] ))
test_set = np.reshape(test_set, (test_set.shape[0] , 1, test_set.shape[1] ))
#### end of reshaping

from keras.models import Sequential
from keras.layers import Dense , LSTM
## Building the model
NN_model = Sequential()

## Add layers to the network
# "https://keras.io/layers/core/" description of Dense function to add layers
#NN_model.add(Dense(85, kernel_initializer='normal',input_dim = train_set.shape[1], activation='relu'))
#NN_model.add(Dense(1, kernel_initializer='normal',activation='relu'))
#lstm layer
look_back = 10
NN_model.add(LSTM(90, input_shape=(1, look_back)))
NN_model.add(Dense(1))

print(NN_model.summary())
#plot_model(NN_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

## Compile the network : To configure network for training
# "https://keras.io/models/model/#compile" description of 'compile' function.
NN_model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #'mean_absolute_error', 'mean_squared_error'


## Train the model
# "https://keras.io/models/sequential/" description of 'fit' function which is used for model training.
model_training = NN_model.fit(train_set, train_target, epochs=50, batch_size=1, validation_split = 0.1)
print("Data saved in history: \n", print(model_training.history.keys()))
print("Model Training History: \n" , model_training.history , "\n")

## Evaluate the
model_evaluation_test = NN_model.evaluate(test_set, test_target, batch_size=1 , verbose=1)
model_evaluation_train = NN_model.evaluate(train_set, train_target, batch_size=1 , verbose=1)
print("Model Evaluation (testset evaluation): \n",NN_model.metrics_names,  model_evaluation_test,"\n")
print("trainset evaluation): \n",NN_model.metrics_names,  model_evaluation_train,"\n")

## Save the model # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
NN_model.save('my_model.h5')
# NN_model = load_model('my_model.h5')

## prediction
print("\nfirst data example : ", train_set[1000] , "  ," , train_target[1000])
model_prediction_1 = NN_model.predict(np.array([train_set[1000]])) # actual value = 92.60757937
print("model prediction : ", model_prediction_1)

print("\n\n second data example : ", train_set[100] , "  ," , train_target[100])
model_prediction_1 = NN_model.predict(np.array([train_set[100]])) # actual value = 92.60757937
print("model prediction : ", model_prediction_1)

print("\n\n third data example : ", train_set[400] , "  ," , train_target[400])
model_prediction_1 = NN_model.predict(np.array([train_set[400]])) # actual value = 92.60757937
print("model prediction : ", model_prediction_1)


########## modeling using scikit linear regression model.
# reg_model = LinearRegression()
# reg_model.fit(train_set,train_target)
# test_prediction = reg_model.predict(test_set)
# print("Linear Regression mean square error : ", mean_squared_error(test_target, test_prediction))

# reg_model = LogisticRegression()
# reg_model.fit(train_set,train_target)
# test_prediction = reg_model.predict(test_set)
# print("\n\n Logistic Regression mean square error : ", mean_squared_error(test_target, test_prediction))


### neural network loss(MSE) plotting (https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
plt.plot(model_training.history['loss'])
plt.plot(model_training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(model_training.history['acc'])
plt.plot(model_training.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
