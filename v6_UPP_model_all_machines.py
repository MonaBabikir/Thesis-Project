import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense , LSTM , Dropout

from keras import optimizers

def build_model(train_data):
    NN_model = Sequential()

    #NN_model.add(Dropout(0.1 ,input_shape = (train_data.shape[1],)))
    NN_model.add(Dense(150, kernel_initializer='normal' , input_dim = train_data.shape[1] , activation='relu')) #input_dim = train_data.shape[1]
    #NN_model.add(Dropout(0.2))
    NN_model.add(Dense(150, kernel_initializer='normal', activation='relu'))
    #NN_model.add(Dropout(0.2))

    #NN_model.add(Dense(150, kernel_initializer='normal', activation='relu'))
    #NN_model.add(Dense(150, kernel_initializer='normal', activation='relu'))
    #NN_model.add(Dense(150, kernel_initializer='normal', activation='relu'))
    #NN_model.add(Dense(150, kernel_initializer='normal', activation='relu'))

    NN_model.add(Dense(1, kernel_initializer='normal'))

    # ## model compilation
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8 ) # , decay=1.0
    NN_model.compile(loss='mean_squared_error', optimizer=adam) #'mean_absolute_error', 'mean_squared_error' , , metrics=['accuracy']

    return NN_model


def generate_model(temp_data , cpu_data , mem_data , diskio_data , diskspace_data , machine_name):
    cpu_data = pd.read_csv(cpu_data)
    mem_data = pd.read_csv(mem_data)
    disk_space_data = pd.read_csv(diskspace_data)
    disk_io_data = pd.read_csv(diskio_data)
    temp_data = pd.read_csv(temp_data)

    data = pd.concat([cpu_data, mem_data, disk_space_data, disk_io_data, temp_data], axis=1)
    data.columns = ["Time", "CPU", "Time2", "Mem", "Time3", "Disk_Space", "Time4", "Disk_IO", "Time5", "Temp"]
    data.drop(["Time", "Time2", "Time3", "Time4", "Time5"], axis=1, inplace=True)

    # print(data)

    # print("\n min value in label data", np.amin(data['Temp']))
    # print(data.shape)

    ## data scaling
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    data.columns = ["CPU", "Mem", "Disk_Space", "Disk_IO", "Temp"]
    # data.columns = ["CPU" , "Mem" , "Disk_IO" ,"Temp"]
    ##end of data scaling

    label_data = data["Temp"]
    in_data = data.drop(["Temp"], axis=1)

    label_data = label_data.to_numpy()
    in_data = in_data.to_numpy()

    ## train data split in randomly
    # print(type(label_data))
    train_data, test_data, train_labels, test_labels = train_test_split(in_data, label_data, test_size=0.2,random_state=42)

    NN_model = build_model(train_data)

    # ## model fitting
    ## normal:
    model_training = NN_model.fit(train_data, train_labels, epochs=5000, batch_size=32, validation_data=(test_data, test_labels), verbose=0)

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

    print("\n min value in the train target and test target after rescaling:", np.amin(inversed_train_label), " , ",
          np.amin(inversed_test_label))
    print("\n max value in the train target and test target after rescaling:", np.amax(inversed_train_label), " , ",
          np.amax(inversed_test_label))

    train_score = mean_squared_error(inversed_train_label, inversed_predicted_train_label)
    test_score = mean_squared_error(inversed_test_label, inversed_predicted_test_label)

    print("After inverse scaling Train MSE: ", train_score, ", Test MSE: ", test_score)

    train_score_mae = mean_absolute_error(inversed_train_label, inversed_predicted_train_label)
    test_score_mae = mean_absolute_error(inversed_test_label, inversed_predicted_test_label)

    print("After inverse scaling Train MAE: ", train_score_mae, ", Test MAE: ", test_score_mae)

    ## end of inverse of data scaling

    print("********************************************************************")

    ### plotting
    plt.plot(model_training.history['loss'], label='train')
    plt.plot(model_training.history['val_loss'], label='test')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig('./models_plots/loss_' + machine_name + '.png')
    plt.clf()

    ### plotting data as sequence,
    data_predict = NN_model.predict(in_data)
    data_predict = pd.DataFrame(data_predict)
    in_data = pd.DataFrame(in_data)
    inversed_data_predict = pd.concat((in_data, data_predict), axis=1)
    inversed_data_predict = scaler.inverse_transform(inversed_data_predict)
    inversed_data_predict = inversed_data_predict[:, -1]

    inversed_data_label = scaler.inverse_transform(data)
    inversed_data_label = inversed_data_label[:, -1]

    plt.plot(inversed_data_label, label='actual Temp data')
    plt.plot(inversed_data_predict, label='predicted Temp data')
    plt.title(' Actual and Predicted ')
    plt.ylabel('Temperature')
    plt.xlabel('time')
    plt.legend(loc='lower right')

    plt.savefig('./models_plots/series_' + machine_name + '.png')

    plt.close('all')

def model_all_machines():
    temp_data_path= "./real_data_prepared_Upp/Temp_mean/"
    cpu_data_path= "./real_data_prepared_Upp/CPU/"
    mem_data_path= "./real_data_prepared_Upp/Mem/"
    diskspace_data_path= "./real_data_prepared_Upp/Disk_space/"
    diskio_data_path= "./real_data_prepared_Upp/Disk_io/"

    dir_names = os.listdir(cpu_data_path)
    for i in range(len(dir_names)):
        print("\nIn directory : ", dir_names[i])
        temp_data = temp_data_path + dir_names[i]
        cpu_data= cpu_data_path + dir_names[i]
        mem_data = mem_data_path + dir_names[i]
        diskio_data = diskio_data_path + dir_names[i]
        diskspace_data = diskspace_data_path + dir_names[i]

        generate_model(temp_data , cpu_data , mem_data , diskio_data , diskspace_data , dir_names[i] )


#model_all_machines()