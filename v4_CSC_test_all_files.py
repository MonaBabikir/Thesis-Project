import os
from keras.models import Sequential, load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler , MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from v4_CSC_data_preprocess import window_data , window_data_general

f = open("all_files_error.txt" , "a+")

def test_all_files():
    data_path = "./real_data_prepared/epouta/"
    dir_names = os.listdir(data_path)
    model = load_model("best_model_lstm_ss.h5")
    for i in range(len(dir_names)):
        print("In directory : ", dir_names[i])
        file_path = data_path +  dir_names[i]
        #data = pd.read_csv(file_path, header=None)

        model_batch_size = 32
        ### normal network (Dense layer):
        # lock_back = 1
        # data = window_data_general("./real_data_prepared/epouta/e101_epouta_csc_fi.csv", lock_back)
        # label_data = data['Cpu_t+' + str(lock_back)]
        # in_data = data.drop(['Cpu_t+' + str(lock_back)], axis=1)

        ### lstm network:
        lock_back = 3
        no_features = 1
        data = window_data_general(file_path , lock_back)
        ## data scaling
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data)
        data.columns = ['Cpu_t+0', 'Cpu_t+1', 'Cpu_t+2', 'Cpu_t+3']
        ## end of scaling

        label_data = data['Cpu_t+'+ str(lock_back)]
        in_data = data.drop(['Cpu_t+' + str(lock_back)], axis=1)
        in_data = in_data.to_numpy()
        in_data = np.reshape(in_data, (in_data.shape[0], lock_back, no_features))

        ## model prediction
        predicted_label = model.predict(in_data)
        predicted_label = pd.DataFrame(predicted_label)

        ## inverse data scaling
        in_set = np.reshape(in_data, (in_data.shape[0], in_data.shape[1]))
        in_set_df = pd.DataFrame(in_set)
        in_label_df= pd.DataFrame(label_data)
        in_set = pd.concat((in_set_df,in_label_df) , axis=1)
        in_set = scaler.inverse_transform(in_set)
        inversed_in_label = in_set[:, -1]

        inversed_predict_label = pd.concat((in_set_df,predicted_label) , axis=1)
        inversed_predict_label = scaler.inverse_transform(inversed_predict_label)
        inversed_predict_label = inversed_predict_label[: , -1]

        model_score = mean_squared_error( inversed_in_label, inversed_predict_label)

        l1 = "model evaluation on unscaled (original) data: " + dir_names[i] + " : \n" +  "MSE : "+str(model_score) + " \n"
        l2 = "min value in original input data and target respectively :" + str(np.amin(in_set)) + " , " + str(np.amin(inversed_in_label)) + "\n"
        l3 = "max value in original input data and target respectively :" + str(np.amax(in_set)) + " , " + str(np.amax(inversed_in_label)) + "\n++++++++++++++++++++++++++\n"
        print(l1)
        print(l2)
        print(l3)
        f.write(l1)
        f.write(l2)
        f.write(l3)

        ## end of inverse data scaling

        ### model evaluation in data:
        model_evaluation = model.evaluate(in_data, label_data, batch_size=model_batch_size, verbose=0)
        print("model evaluation on scaled data: ",dir_names[i], " : \n", model.metrics_names, model_evaluation)
        print("min value in input data and target :", np.amin(in_data), " , ", np.amin(label_data))
        print("max value in input data and target :", np.amax(in_data), " , ", np.amax(label_data), "\n*********************************************\n")

        ## plotting
        fig = plt.figure()
        # plt.plot(test_labels.to_numpy(), label='test actual')
        # plt.plot(test_predict , label="test predict")
        plt.plot(inversed_in_label, label='actual target series')
        plt.plot(inversed_predict_label, label="predicted label series")

        plt.title(dir_names[i])
        plt.ylabel('CPU')
        plt.xlabel('time')
        plt.legend(loc='upper left')
        plt.savefig("./plots/"+str(i)+".png")
        #plt.show()
    f.close()

test_all_files()

# data = window_data_general("./real_data_prepared/epouta/e101_epouta_csc_fi.csv" , 3)
# label_data = data['Cpu_t+'+ str(3)]
# in_data = data.drop(['Cpu_t+' + str(3)], axis=1)
# n = int(float(data.shape[0]) * 0.8)
# train_data = in_data[:n]
# train_labels = label_data[:n]
# test_data = in_data[n:]
# test_labels = label_data[n:]
#
# test_data = test_data.to_numpy()
# test_labels = test_labels.to_numpy()
# test_data = np.reshape(test_data, (test_data.shape[0], 3, 1))
#
# model = load_model("best_model_lstm.h5")
# test_predict = model.predict(test_data)
# print(test_labels)
# print(test_predict)
#
# plt.plot(test_labels, label='Ytest actual')
# plt.plot(test_predict , label="Ytest predict")
# plt.legend(loc='upper left')
# plt.show()
