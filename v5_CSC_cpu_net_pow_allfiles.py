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
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense , LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint


def build_model(train_data):
    NN_model = Sequential()

    NN_model.add(Dense(20, kernel_initializer='normal', input_dim=train_data.shape[1],activation='relu'))  # input_dim = train_data.shape[1]
    NN_model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    # NN_model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(1, kernel_initializer='normal'))

    # ## model compilation
    NN_model.compile(loss='mean_squared_error',optimizer='adam')  # 'mean_absolute_error', 'mean_squared_error' , , metrics=['accuracy']

    return NN_model

def generate_model(cpu_pow_data, net_data,machine_name):
    #cpu_pow_data = pd.read_csv("./real_data_prepared/epouta/e23_epouta_csc_fi.csv", header=None)
    cpu_pow_data = pd.read_csv(cpu_pow_data, header=None)
    cpu_pow_data.columns = ["Time", "CPU", "Power"]
    cpu_data = cpu_pow_data["CPU"]
    pow_data = cpu_pow_data["Power"]
    #net_data = pd.read_csv("./real_data_prepared/epouta_net/e23_epouta_csc_fi.csv", header=None)
    net_data = pd.read_csv(net_data, header=None)

    data = pd.concat([net_data, cpu_data, pow_data], axis=1)
    data.columns = ["Time", "Net", "CPU", "Power"]
    data.drop(["Time"], axis=1, inplace=True)

    print("\n min value in label data", np.amin(data['Power']))
    print(data.shape)
    ## data scaling
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    data.columns = ["Net" , "CPU" , "Power"]
    #data.columns = ["Net", "Power"]
    ##end of data scaling

    label_data = data["Power"]
    in_data = data.drop(["Power"], axis=1)

    label_data = label_data.to_numpy()
    in_data = in_data.to_numpy()

    ## train data split in randomly
    print(type(label_data))
    train_data, test_data, train_labels, test_labels = train_test_split(in_data, label_data, test_size=0.2,random_state=42)  # , random_state=42

    # ## min/max values for the target
    print("\n min value in the train target and test target :", np.amin(train_labels), " , ", np.amin(test_labels))
    print("\n max value in the train target and test target :", np.amax(train_labels), " , ", np.amax(test_labels))

    # ## model building
    NN_model = build_model(train_data)

    #Model_Checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # ## model fitting
    ## normal:
    model_training = NN_model.fit(train_data, train_labels, epochs=1, batch_size=32,validation_data=(test_data, test_labels))  # validation_split = 0.2 , ,callbacks=[Model_Checkpoint]

    NN_model.save("./models/final_model_" + machine_name + ".h5")

    ## inverse of data scaling
    train_set_df = pd.DataFrame(train_data)
    train_label_df = pd.DataFrame(train_labels)
    train_set = pd.concat((train_set_df, train_label_df), axis=1)

    test_set_df = pd.DataFrame(test_data)
    test_label_df = pd.DataFrame(test_labels)
    test_set = pd.concat((test_set_df, test_label_df), axis=1)

    train_set = scaler.inverse_transform(train_set)
    test_set = scaler.inverse_transform(test_set)

    inversed_train_label = train_set[:, -1]
    inversed_test_label = test_set[:, -1]

    predicted_train_label = NN_model.predict(train_data)
    predicted_train_label = pd.DataFrame(predicted_train_label)
    predicted_test_label = NN_model.predict(test_data)
    predicted_test_label = pd.DataFrame(predicted_test_label)

    inversed_predicted_train_label = pd.concat((train_set_df, predicted_train_label), axis=1)
    inversed_predicted_train_label = scaler.inverse_transform(inversed_predicted_train_label)
    inversed_predicted_train_label = inversed_predicted_train_label[:, -1]

    inversed_predicted_test_label = pd.concat((test_set_df, predicted_test_label), axis=1)
    inversed_predicted_test_label = scaler.inverse_transform(inversed_predicted_test_label)
    inversed_predicted_test_label = inversed_predicted_test_label[:, -1]

    print("\n min value in the train target and test target after rescaling:", np.amin(inversed_train_label), " , ",np.amin(inversed_test_label))
    print("\n max value in the train target and test target after rescaling:", np.amax(inversed_train_label), " , ",np.amax(inversed_test_label))

    train_score = mean_squared_error(inversed_train_label, inversed_predicted_train_label)
    test_score = mean_squared_error(inversed_test_label, inversed_predicted_test_label)

    print("After inverse scaling Train MSE: ", train_score, ", Test MSE: ", test_score)

    train_score_mae = mean_absolute_error(inversed_train_label, inversed_predicted_train_label)
    test_score_mae = mean_absolute_error(inversed_test_label, inversed_predicted_test_label)

    print("After inverse scaling Train MAE: ", train_score_mae, ", Test MAE: ", test_score_mae)

    ## end of inverse of data scaling

    #
    # ### neural network loss(MSE) plotting (https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
    plt.plot(model_training.history['loss'], label='train')
    plt.plot(model_training.history['val_loss'], label='test')
    # plt.plot(loss_his , label='train')
    # plt.plot(val_loss_his , label='test')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig('./models_plots/loss_' + machine_name + '.png')

    ### plotting data as sequence,
    data_predict = NN_model.predict(in_data)
    data_predict = pd.DataFrame(data_predict)
    in_data = pd.DataFrame(in_data)
    inversed_data_predict = pd.concat((in_data, data_predict), axis=1)
    inversed_data_predict = scaler.inverse_transform(inversed_data_predict)
    inversed_data_predict = inversed_data_predict[:, -1]

    inversed_data_label = scaler.inverse_transform(data)
    inversed_data_label = inversed_data_label[:, -1]

    plt.plot(inversed_data_label, label='actual power data')
    plt.plot(inversed_data_predict, label='predicted power data')
    plt.title(' Actual and Predicted ')
    plt.ylabel('Power')
    plt.xlabel('time')
    plt.legend(loc='lower left')
    #plt.show()

    plt.savefig('./models_plots/series_' + machine_name + '.png')

    plt.close('all')



def model_all_machines():
    cpu_pow_data_path = "./real_data_prepared/epouta/"
    net_data_path = "./real_data_prepared/epouta_net/"
    dir_names = os.listdir(net_data_path)
    for i in range(len(dir_names)):
        print("In directory : ", dir_names[i])
        cpu_pow_data = cpu_pow_data_path + dir_names[i]
        net_data = net_data_path + dir_names[i]
        generate_model(cpu_pow_data, net_data,dir_names[i])