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
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense , LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint

from v4_CSC_data_preprocess import window_data , window_data_general , time_cpu_data

np.random.seed(7) # to make the result reproducable.

state= False
lock_back = 1
no_features = 1
# data = window_data_general("./real_data_prepared/epouta/e101_epouta_csc_fi.csv" , lock_back)
# label_data = data['Cpu_t+'+ str(lock_back)]
# in_data = data.drop(['Cpu_t+' + str(lock_back)], axis=1)

data = time_cpu_data("./real_data_prepared/epouta/e101_epouta_csc_fi.csv")
label_data = data['Cpu']
in_data  = data['int_time']

## train data split in order
n = int(float(data.shape[0]) * 0.8)
train_data = in_data[:n]
train_labels = label_data[:n]
test_data = in_data[n:]
test_labels = label_data[n:]

## train data split in randomly
#train_data, test_data, train_labels, test_labels = train_test_split(in_data, label_data, test_size=0.2, random_state=42)

print(train_data)
print(train_data.shape)

print(test_data)
print(test_data.shape)

#plt.plot(train_labels)
#plt.show()
#
#
# train_data, test_data = train_test_split(data, test_size=0.2)
# #train_target = train_data["Power"]
# #train_data.drop(["Power"] , axis=1 , inplace=True)
# train_data , train_target = np.hsplit(train_data, [-1])
# #test_target = test_data["Power"]
# #test_data.drop(["Power"] , axis=1 , inplace=True)
# test_data , test_target = np.hsplit(test_data, [-1])
#
# ## min/max values for the target
print("\n min value in the train target and test target :", np.amin(train_labels) , " , " , np.amin(test_labels))
print("\n max value in the train target and test target :", np.amax(train_labels) , " , " , np.amax(test_labels))
# ##
#
# ## data scaling
# scaler1 = StandardScaler()
# scaler2 = StandardScaler()
# train_data = scaler1.fit_transform(train_data)
# test_data = scaler1.fit_transform(test_data)
# train_target =scaler2.fit_transform(train_target)
# test_target = scaler2.fit_transform(test_target)
# ## end of data scaling
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

    NN_model = Sequential()

    NN_model.add(LSTM(100, input_shape=(train_set.shape[1], train_set.shape[2]) , stateful=state , return_sequences=True))  # input_dim = train_data.shape[1] , batch_input_shape=(model_batch_size,
    NN_model.add(LSTM(100 , return_sequences=True) )
    NN_model.add(LSTM(100))
    NN_model.add(Dense(1))

    # ## model compilation
    NN_model.compile(loss='mean_squared_error',optimizer='adam')  # 'mean_absolute_error', 'mean_squared_error' , , metrics=['accuracy']


    return NN_model , train_set , test_set , input_data


NN_model , train_data , test_data , in_data = build_lstm_model(train_data , test_data , in_data)


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

model_training = NN_model.fit(train_data, train_labels, epochs=1000, batch_size=model_batch_size, validation_data=(test_data, test_labels) , callbacks=[Model_Checkpoint] , shuffle=False) #validation_split = 0.2

print("Data saved in history: \n", print(model_training.history.keys()))
print("Model Training History: \n" , model_training.history , "\n")
#
# ## model evaluation
model_evaluation_test = NN_model.evaluate(test_data, test_labels, batch_size=model_batch_size , verbose=0)
model_evaluation_train = NN_model.evaluate(train_data, train_labels, batch_size=model_batch_size , verbose=0)
model_evaluation_alldata = NN_model.evaluate(in_data, label_data, batch_size=model_batch_size , verbose=0)
print("trainset evaluation _on the final model: \n",NN_model.metrics_names,  model_evaluation_train,"\n")
print("Model Evaluation (testset evaluation) _on the final model: \n",NN_model.metrics_names,  model_evaluation_test,"\n")
print("the whole model evaluation _on the final model: \n" , NN_model.metrics_names,  model_evaluation_alldata , "\n")

saved_best_model = load_model('best_model.h5')
train_loss = saved_best_model.evaluate(train_data, train_labels, verbose=0 , batch_size=model_batch_size)
test_loss = saved_best_model.evaluate(test_data, test_labels, verbose=0 , batch_size=model_batch_size)
print("train loss evaluation _on the best model: \n",NN_model.metrics_names,  train_loss,"\n")
print("test loss evaluation _on the best model: \n",NN_model.metrics_names,  test_loss,"\n")


### evaluation using scikit-Learn cross validation (https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)
def build_lstm_model_cv():
    NN_model = Sequential()

    #NN_model.add(LSTM(10, kernel_initializer='normal', input_shape=(train_data.shape[1], train_data.shape[2])))  # input_dim = train_data.shape[1]
    NN_model.add(LSTM(5, batch_input_shape=(model_batch_size, train_data.shape[1], train_data.shape[2]), stateful=state))
    # NN_model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(1))

    # ## model compilation
    NN_model.compile(loss='mean_squared_error',optimizer='adam')  # 'mean_absolute_error', 'mean_squared_error' , , metrics=['accuracy']


    return NN_model



# seed = 7
# np.random.seed(seed)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=build_lstm_model_cv, epochs=1000, batch_size=model_batch_size, verbose=0)
#
# kfold = KFold(n_splits=10, random_state=seed)
# scores = cross_val_score(estimator, in_data, label_data, cv=kfold)
# #print("Results: %.10f (%.10f) MSE" % (scores.mean(), scores.std()))
# print("Results: %.10f MSE" % (np.mean(scores)))
# print(scores , "\n")

#### with standerdize data :
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=build_model, epochs=500, batch_size=model_batch_size, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# scores = cross_val_score(pipeline, in_data, label_data, cv=kfold)
# #print("Standardized: %.10f (%.10f) MSE" % (scores.mean(), scores.std()))
# print("Standardized: %.10f MSE" % (np.mean(scores)))
# print(scores , "\n")


## some prediction
# print("\nfirst data example : ", train_data.iloc[1000] , "  ," , train_labels.iloc[1000])
# model_prediction_1 = NN_model.predict(np.array([train_data.iloc[1000]]))
# print("model prediction : ", model_prediction_1)
#
# print("\nfirst data example : ", test_data.iloc[700] , "  ," , test_labels.iloc[700])
# model_prediction_2 = NN_model.predict(np.array([test_data.iloc[700]]))
# print("model prediction : ", model_prediction_2)
#
# print("\nfirst data example : ", test_data.iloc[50] , "  ," , test_labels.iloc[50])
# model_prediction_2 = NN_model.predict(np.array([test_data.iloc[50]]))
# print("model prediction : ", model_prediction_2)


#
# ### neural network loss(MSE) plotting (https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
plt.plot(model_training.history['loss'] , label='train')
plt.plot(model_training.history['val_loss'] , label='test')
#plt.plot(loss_his , label='train')
#plt.plot(val_loss_his , label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()

### plotting data shape
#plt.plot(data['Cpu'])
# plt.plot(data['DateTime'] , data['Cpu'])
# plt.title("CPU Time Series")
# plt.ylabel('CPU Reading')
# plt.xlabel('DataTime')
# plt.show()


### plot training/ test prediction
train_predict = NN_model.predict(train_data ,  batch_size=model_batch_size)
test_predict = NN_model.predict(test_data , batch_size=model_batch_size)

fig = plt.figure()
plt.subplot(1,2, 1)
plt.plot(train_labels  , label='train actual')
plt.plot(train_predict , label='train predict')

plt.subplot(1, 2,2)
plt.plot(test_labels, label='test actual')
plt.plot(test_predict , label="test predict")

plt.title(' Actual and Predicted ')
plt.ylabel('CPU')
plt.xlabel('time')
plt.legend(loc='upper left')
plt.show()