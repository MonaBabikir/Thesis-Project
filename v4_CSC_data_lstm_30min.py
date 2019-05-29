import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense , LSTM
from keras.callbacks import ModelCheckpoint

from v4_CSC_data_preprocess import window_data , window_data_general , time_cpu_data
from v4_CSC_data_preprocess_30min import window_data_general_30min
from forecasting_metrics import rrse

np.random.seed(7) # to make the result reproducable.

state= False
lock_back = 3
no_features = 1
#data = window_data_general("./real_data_prepared/epouta/e23_epouta_csc_fi.csv" , lock_back)
#data = window_data_general("./real_data_prepared/epouta_30min/e23_epouta_csc_fi.csv" , lock_back)

data1, data2 , data3 , data = window_data_general_30min("./real_data_prepared/epouta/e23_epouta_csc_fi.csv" , lock_back)

## data scaling

# data 1
scaler1 = MinMaxScaler()
data1 = scaler1.fit_transform(data1)
data1 = pd.DataFrame(data1)
data1.columns = ['Cpu_t+0', 'Cpu_t+1', 'Cpu_t+2', 'Cpu_t+3']
# data.columns = ['Cpu_t+0', 'Cpu_t+1' , 'Cpu_t+2', 'Cpu_t+3' , 'Cpu_t+4' , 'Cpu_t+5']

#data 2
scaler2 = MinMaxScaler()
data2 = scaler2.fit_transform(data2)
data2 = pd.DataFrame(data2)
data2.columns = ['Cpu_t+0', 'Cpu_t+1', 'Cpu_t+2', 'Cpu_t+3']
# data.columns = ['Cpu_t+0', 'Cpu_t+1' , 'Cpu_t+2', 'Cpu_t+3' , 'Cpu_t+4' , 'Cpu_t+5']

#data 3
scaler3 = MinMaxScaler()
data3 = scaler3.fit_transform(data3)
data3 = pd.DataFrame(data3)
data3.columns = ['Cpu_t+0', 'Cpu_t+1', 'Cpu_t+2', 'Cpu_t+3']
# data.columns = ['Cpu_t+0', 'Cpu_t+1' , 'Cpu_t+2', 'Cpu_t+3' , 'Cpu_t+4' , 'Cpu_t+5']

#data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)
data.columns = ['Cpu_t+0', 'Cpu_t+1', 'Cpu_t+2', 'Cpu_t+3']
# data.columns = ['Cpu_t+0', 'Cpu_t+1' , 'Cpu_t+2', 'Cpu_t+3' , 'Cpu_t+4' , 'Cpu_t+5']


##end of data scaling


label_data = data['Cpu_t+'+ str(lock_back)]
in_data = data.drop(['Cpu_t+' + str(lock_back)], axis=1)

label_data_1 = data1['Cpu_t+'+ str(lock_back)]
in_data_1 = data1.drop(['Cpu_t+' + str(lock_back)], axis=1)

label_data_2 = data2['Cpu_t+'+ str(lock_back)]
in_data_2 = data2.drop(['Cpu_t+' + str(lock_back)], axis=1)

label_data_3 = data3['Cpu_t+'+ str(lock_back)]
in_data_3 = data3.drop(['Cpu_t+' + str(lock_back)], axis=1)



## train data split in order
# n = int(float(data.shape[0]) * 0.2)
# train_data = in_data[:n]
# train_labels = label_data[:n]
# test_data = in_data[n:]
# test_labels = label_data[n:]

n1 = int(float(data1.shape[0]) * 0.8)
train_data_1 = in_data_1[:n1]
train_labels_1 = label_data_1[:n1]
test_data_1 = in_data_1[n1:]
test_labels_1 = label_data_1[n1:]

n2 = int(float(data2.shape[0]) * 0.8)
train_data_2 = in_data_2[:n2]
train_labels_2 = label_data_2[:n2]
test_data_2 = in_data_2[n2:]
test_labels_2 = label_data_2[n2:]

n3 = int(float(data3.shape[0]) * 0.2)
train_data_3 = in_data_3[:n3]
train_labels_3 = label_data_3[:n3]
test_data_3 = in_data_3[n3:]
test_labels_3 = label_data_3[n3:]





## train data split in randomly
#train_data, test_data, train_labels, test_labels = train_test_split(in_data, label_data, test_size=0.2, random_state=42)

# print(train_data)
# print(train_data.shape)
#
# print(test_data)
# print(test_data.shape)

#
# ## min/max values for the target
# print("\n min value in the train target and test target :", np.amin(train_labels) , " , " , np.amin(test_labels))
# print("\n max value in the train target and test target :", np.amax(train_labels) , " , " , np.amax(test_labels))

# ##
#
#
# # print(cpu_df.shape)
#
# ## end of data loading
#
# ## model building
model_batch_size= 32

def build_lstm_model( train_set , test_set , input_data):
    train_set = train_set.to_numpy()
    test_set = test_set.to_numpy()
    input_data = input_data.to_numpy()
    # train_set = np.reshape(train_set, (train_set.shape[0], 1, train_set.shape[1]))
    # test_set = np.reshape(test_set, (test_set.shape[0], 1, test_set.shape[1]))
    # input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

    train_set = np.reshape(train_set, (train_set.shape[0], lock_back, no_features))
    test_set = np.reshape(test_set, (test_set.shape[0], lock_back, no_features))
    input_data = np.reshape(input_data, (input_data.shape[0], lock_back, no_features))

    # train_set = np.reshape(train_set, (1, lock_back, no_features))
    # test_set = np.reshape(test_set, (1, lock_back, no_features))
    # input_data = np.reshape(input_data, (1, lock_back, no_features))

    NN_model = Sequential()

    NN_model.add(LSTM(5, input_shape=(train_set.shape[1], train_set.shape[2]) , stateful=state , activation='tanh' , return_sequences=True ))  # input_dim = train_data.shape[1] , batch_input_shape=(model_batch_size, , return_sequences=True , activation='relu'
    NN_model.add(LSTM(5 , activation='tanh' ) )
    #.add(LSTM(5, activation='tanh'))
    # NN_model.add(LSTM(100))

    NN_model.add(Dense(1))

    # ## model compilation
    NN_model.compile(loss='mean_squared_error' ,optimizer='sgd')  # 'mean_absolute_error', 'mean_squared_error' , , metrics=['accuracy']


    return NN_model , train_set , test_set , input_data


def build_lstm_model_30min( train_set1 , test_set1 , input_data1 , train_set2 , test_set2 , input_data2 , train_set3 , test_set3 , input_data3 , input_data):
    train_set1 = train_set1.to_numpy()
    test_set1 = test_set1.to_numpy()
    input_data1 = input_data1.to_numpy()

    train_set2 = train_set2.to_numpy()
    test_set2 = test_set2.to_numpy()
    input_data2 = input_data2.to_numpy()

    train_set3 = train_set3.to_numpy()
    test_set3 = test_set3.to_numpy()
    input_data3 = input_data3.to_numpy()

    input_data = input_data.to_numpy()

    train_set1 = np.reshape(train_set1, (train_set1.shape[0], lock_back, no_features))
    test_set1 = np.reshape(test_set1, (test_set1.shape[0], lock_back, no_features))
    input_data1 = np.reshape(input_data1, (input_data1.shape[0], lock_back, no_features))

    train_set2 = np.reshape(train_set2, (train_set2.shape[0], lock_back, no_features))
    test_set2 = np.reshape(test_set2, (test_set2.shape[0], lock_back, no_features))
    input_data2 = np.reshape(input_data2, (input_data2.shape[0], lock_back, no_features))

    train_set3 = np.reshape(train_set3, (train_set3.shape[0], lock_back, no_features))
    test_set3 = np.reshape(test_set3, (test_set3.shape[0], lock_back, no_features))
    input_data3 = np.reshape(input_data3, (input_data3.shape[0], lock_back, no_features))

    input_data = np.reshape(input_data, (input_data.shape[0], lock_back, no_features))


    NN_model = Sequential()

    NN_model.add(LSTM(5, input_shape=(lock_back, no_features) , stateful=state , activation='tanh' , return_sequences=True ))  # input_dim = train_data.shape[1] , batch_input_shape=(model_batch_size, , return_sequences=True , activation='relu'
    NN_model.add(LSTM(5 , activation='tanh' ) )
    #.add(LSTM(5, activation='tanh'))
    # NN_model.add(LSTM(100))

    NN_model.add(Dense(1))

    # ## model compilation
    NN_model.compile(loss='mean_squared_error' ,optimizer='sgd')  # 'mean_absolute_error', 'mean_squared_error' , , metrics=['accuracy']


    return NN_model , train_set1 , test_set1 , input_data1 , train_set2 , test_set2 , input_data2 , train_set3 , test_set3 , input_data3 , input_data




# NN_model , train_data , test_data , in_data = build_lstm_model(train_data , test_data , in_data)
NN_model , train_data_1 , test_data_1 , in_data_1 , train_data_2 , test_data_2 , \
        in_data_2 , train_data_3 , test_data_3 , in_data_3  ,in_data \
        = build_lstm_model_30min( train_data_1 , test_data_1 , in_data_1 , train_data_2 , test_data_2 ,
                                                                      in_data_2 , train_data_3 , test_data_3 , in_data_3 , in_data)

# print("after reshape : ", train_data.shape , test_data.shape)


Model_Checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# ## model fitting
##lstm:
#for stateful :
# epochs_no = 1000
# for i in range(epochs_no):
#     loss_his = []
#     val_loss_his = []
#     model_training = NN_model.fit(train_data, train_labels, epochs=1, batch_size=model_batch_size, validation_data=(test_data, test_labels) , callbacks=[Model_Checkpoint] , shuffle=False) #validation_split = 0.2
#     loss_his.append(model_training.history['loss'])
#     val_loss_his.append(model_training.history['val_loss'])
#     NN_model.reset_states()

### for rrse loss function:
train_labels_1 = train_labels_1.to_numpy()
train_labels_2 = train_labels_2.to_numpy()
train_labels_3 = train_labels_3.to_numpy()


#model_batch_size = train_data.shape[0]

model_training1 = NN_model.fit(train_data_1, train_labels_1, epochs=1000, batch_size=model_batch_size, validation_data=(test_data_1, test_labels_1) , callbacks=[Model_Checkpoint] , shuffle=False) #validation_split = 0.2
model_training2 = NN_model.fit(train_data_2, train_labels_2, epochs=1000, batch_size=model_batch_size, validation_data=(test_data_2, test_labels_2) , callbacks=[Model_Checkpoint] , shuffle=False)
model_training3 = NN_model.fit(train_data_3, train_labels_3, epochs=1000, batch_size=model_batch_size, validation_data=(test_data_3, test_labels_3) , callbacks=[Model_Checkpoint] , shuffle=False)

# print("Data saved in history: \n", print(model_training.history.keys()))
# print("Model Training History: \n" , model_training.history , "\n")

NN_model.save("final_model.h5")

# ## model evaluation
# model_evaluation_test = NN_model.evaluate(test_data, test_labels, batch_size=model_batch_size , verbose=0)
# model_evaluation_train = NN_model.evaluate(train_data, train_labels, batch_size=model_batch_size , verbose=0)
# model_evaluation_alldata = NN_model.evaluate(in_data, label_data, batch_size=model_batch_size , verbose=0)
# print("trainset evaluation _on the final model: \n",NN_model.metrics_names,  model_evaluation_train,"\n")
# print("Model Evaluation (testset evaluation) _on the final model: \n",NN_model.metrics_names,  model_evaluation_test,"\n")
# print("the whole model evaluation _on the final model: \n" , NN_model.metrics_names,  model_evaluation_alldata , "\n")
#
# saved_best_model = load_model('best_model.h5')
# train_loss = saved_best_model.evaluate(train_data, train_labels, verbose=0 , batch_size=model_batch_size)
# test_loss = saved_best_model.evaluate(test_data, test_labels, verbose=0 , batch_size=model_batch_size)
# print("train loss evaluation _on the best model: \n",NN_model.metrics_names,  train_loss,"\n")
# print("test loss evaluation _on the best model: \n",NN_model.metrics_names,  test_loss,"\n")


## inverse of data scaling
train_set = np.reshape(train_data_1, (train_data_1.shape[0], train_data_1.shape[1]))
train_set_df = pd.DataFrame(train_set)
train_label_df= pd.DataFrame(train_labels_1)
train_set = pd.concat((train_set_df,train_label_df) , axis=1)

test_set = np.reshape(test_data_1, (test_data_1.shape[0], test_data_1.shape[1]))
test_set_df = pd.DataFrame(test_set)
test_label_df= pd.DataFrame(test_labels_1.to_numpy())
test_set = pd.concat((test_set_df, test_label_df) , axis=1)


train_set = scaler1.inverse_transform(train_set)
test_set = scaler1.inverse_transform(test_set)

inversed_train_label = train_set[:,-1]
#inversed_train_in = train_set[: , :-1]
inversed_test_label = test_set[: ,-1]
#inversed_test_in = train_set[: , :-1]

predicted_train_label = NN_model.predict(train_data_1)
predicted_train_label = pd.DataFrame(predicted_train_label)
predicted_test_label = NN_model.predict(test_data_1)
predicted_test_label = pd.DataFrame(predicted_test_label)

inversed_predicted_train_label = pd.concat((train_set_df , predicted_train_label) , axis=1)
inversed_predicted_train_label = scaler.inverse_transform(inversed_predicted_train_label)
inversed_predicted_train_label = inversed_predicted_train_label[: , -1]

inversed_predicted_test_label = pd.concat((test_set_df , predicted_test_label) , axis=1)
inversed_predicted_test_label = scaler.inverse_transform(inversed_predicted_test_label)
inversed_predicted_test_label = inversed_predicted_test_label[: , -1]

print("\n min value in the train target and test target after rescaling:", np.amin(inversed_train_label) , " , " , np.amin(inversed_test_label))
print("\n max value in the train target and test target after rescaling:", np.amax(inversed_train_label) , " , " , np.amax(inversed_test_label))

train_score = mean_squared_error(inversed_train_label, inversed_predicted_train_label)
test_score = mean_squared_error(inversed_test_label, inversed_predicted_test_label)

print("1 After inverse scaling Train MSE: ", train_score , ", Test MSE: ", test_score)


#########################################################################################################
train_set = np.reshape(train_data_2, (train_data_2.shape[0], train_data_2.shape[1]))
train_set_df = pd.DataFrame(train_set)
train_label_df= pd.DataFrame(train_labels_2)
train_set = pd.concat((train_set_df,train_label_df) , axis=1)

test_set = np.reshape(test_data_2, (test_data_2.shape[0], test_data_2.shape[1]))
test_set_df = pd.DataFrame(test_set)
test_label_df= pd.DataFrame(test_labels_2.to_numpy())
test_set = pd.concat((test_set_df, test_label_df) , axis=1)


train_set = scaler2.inverse_transform(train_set)
test_set = scaler2.inverse_transform(test_set)

inversed_train_label = train_set[:,-1]
#inversed_train_in = train_set[: , :-1]
inversed_test_label = test_set[: ,-1]
#inversed_test_in = train_set[: , :-1]

predicted_train_label = NN_model.predict(train_data_2)
predicted_train_label = pd.DataFrame(predicted_train_label)
predicted_test_label = NN_model.predict(test_data_2)
predicted_test_label = pd.DataFrame(predicted_test_label)

inversed_predicted_train_label = pd.concat((train_set_df , predicted_train_label) , axis=1)
inversed_predicted_train_label = scaler.inverse_transform(inversed_predicted_train_label)
inversed_predicted_train_label = inversed_predicted_train_label[: , -1]

inversed_predicted_test_label = pd.concat((test_set_df , predicted_test_label) , axis=1)
inversed_predicted_test_label = scaler.inverse_transform(inversed_predicted_test_label)
inversed_predicted_test_label = inversed_predicted_test_label[: , -1]

print("\n min value in the train target and test target after rescaling:", np.amin(inversed_train_label) , " , " , np.amin(inversed_test_label))
print("\n max value in the train target and test target after rescaling:", np.amax(inversed_train_label) , " , " , np.amax(inversed_test_label))

train_score = mean_squared_error(inversed_train_label, inversed_predicted_train_label)
test_score = mean_squared_error(inversed_test_label, inversed_predicted_test_label)

print("2 After inverse scaling Train MSE: ", train_score , ", Test MSE: ", test_score)


#############################################################################################################
train_set = np.reshape(train_data_3, (train_data_3.shape[0], train_data_3.shape[1]))
train_set_df = pd.DataFrame(train_set)
train_label_df= pd.DataFrame(train_labels_3)
train_set = pd.concat((train_set_df,train_label_df) , axis=1)

test_set = np.reshape(test_data_3, (test_data_3.shape[0], test_data_3.shape[1]))
test_set_df = pd.DataFrame(test_set)
test_label_df= pd.DataFrame(test_labels_3.to_numpy())
test_set = pd.concat((test_set_df, test_label_df) , axis=1)


train_set = scaler3.inverse_transform(train_set)
test_set = scaler3.inverse_transform(test_set)

inversed_train_label = train_set[:,-1]
#inversed_train_in = train_set[: , :-1]
inversed_test_label = test_set[: ,-1]
#inversed_test_in = train_set[: , :-1]

predicted_train_label = NN_model.predict(train_data_3)
predicted_train_label = pd.DataFrame(predicted_train_label)
predicted_test_label = NN_model.predict(test_data_3)
predicted_test_label = pd.DataFrame(predicted_test_label)

inversed_predicted_train_label = pd.concat((train_set_df , predicted_train_label) , axis=1)
inversed_predicted_train_label = scaler.inverse_transform(inversed_predicted_train_label)
inversed_predicted_train_label = inversed_predicted_train_label[: , -1]

inversed_predicted_test_label = pd.concat((test_set_df , predicted_test_label) , axis=1)
inversed_predicted_test_label = scaler.inverse_transform(inversed_predicted_test_label)
inversed_predicted_test_label = inversed_predicted_test_label[: , -1]

print("\n min value in the train target and test target after rescaling:", np.amin(inversed_train_label) , " , " , np.amin(inversed_test_label))
print("\n max value in the train target and test target after rescaling:", np.amax(inversed_train_label) , " , " , np.amax(inversed_test_label))

train_score = mean_squared_error(inversed_train_label, inversed_predicted_train_label)
test_score = mean_squared_error(inversed_test_label, inversed_predicted_test_label)

print("3 After inverse scaling Train MSE: ", train_score , ", Test MSE: ", test_score)


## end of inverse of data scaling



#
# ### neural network loss(MSE) plotting (https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
plt.subplot(1,3,1)
plt.plot(model_training1.history['loss'] , label='train')
plt.plot(model_training1.history['val_loss'] , label='test')
#plt.plot(loss_his , label='train')
#plt.plot(val_loss_his , label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')

plt.subplot(1,3,2)
plt.plot(model_training2.history['loss'] , label='train')
plt.plot(model_training2.history['val_loss'] , label='test')
#plt.plot(loss_his , label='train')
#plt.plot(val_loss_his , label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')

plt.subplot(1,3,3)
plt.plot(model_training3.history['loss'] , label='train')
plt.plot(model_training3.history['val_loss'] , label='test')
#plt.plot(loss_his , label='train')
#plt.plot(val_loss_his , label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()




### plot training/ test prediction


# train_predict = NN_model.predict(train_data ,  batch_size=model_batch_size)
# test_predict = NN_model.predict(test_data , batch_size=model_batch_size)

fig = plt.figure()
plt.subplot(1,2, 1)
# plt.plot(train_labels  , label='train actual')
# plt.plot(train_predict , label='train predict')
plt.plot(inversed_train_label  , label='train actual')
plt.plot(inversed_predicted_train_label , label='train predict')


plt.subplot(1, 2,2)
# plt.plot(test_labels.to_numpy(), label='test actual')
# plt.plot(test_predict , label="test predict")
plt.plot(inversed_test_label, label='test actual')
plt.plot(inversed_predicted_test_label , label="test predict")

plt.title(' Actual and Predicted ')
plt.ylabel('CPU')
plt.xlabel('time')
plt.legend(loc='upper left')
plt.show()




###another plotting
data_predict = NN_model.predict(in_data)

data_in = np.reshape(in_data, (in_data.shape[0], in_data.shape[1]))
data_in = pd.DataFrame(data_in)

data_predict = pd.concat((data_in , pd.DataFrame(data_predict)) , axis=1)
data_predict = scaler.inverse_transform(data_predict)
data_predict = data_predict[: , -1]

data_target = pd.concat((data_in , label_data) , axis=1)
data_target = scaler.inverse_transform(data_target)
data_target = data_target[: , -1]

plt.plot(data_target  , label='actual timeseries')
plt.plot(data_predict , label='predicted timeseries')
plt.title(' Actual and Predicted timeseries')
plt.ylabel('CPU')
plt.xlabel('time')
plt.legend(loc='upper right')
plt.show()
