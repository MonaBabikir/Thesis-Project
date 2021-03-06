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

from v4_CSC_data_preprocess import window_data , window_data_general

np.random.seed(7) # to make the result reproducable.

## uploading dataset
# data = pd.read_csv("./real_data_prepared/epouta/e101_epouta_csc_fi.csv", header=None)
# data.columns = ['Time' , 'Cpu', 'Power']
# print(list(data))
# print(len(data))
#
# data['DateTime'] = pd.to_datetime(data.Time)
#
# data.drop(['Power', 'Time'], axis=1, inplace=True)
# print(data.head)
# print(data.dtypes)
# print(data.iloc[4415].DateTime.day)
#
# label_data = data['Cpu']
# in_data = data['DateTime']

lock_back = 3
data = window_data_general("./real_data_prepared/epouta/e23_epouta_csc_fi.csv" , lock_back)

## data scaling
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)
data.columns = ['Cpu_t+0', 'Cpu_t+1', 'Cpu_t+2', 'Cpu_t+3']
##end of data scaling

label_data = data['Cpu_t+'+ str(lock_back)]
in_data = data.drop(['Cpu_t+' + str(lock_back)], axis=1)

## train data split in order
# n = int(float(data.shape[0]) * 0.8)
# train_data = in_data[:n]
# train_labels = label_data[:n]
# test_data = in_data[n:]
# test_labels = label_data[n:]

## train data split in randomly
train_data, test_data, train_labels, test_labels = train_test_split(in_data, label_data, test_size=0.2, random_state=42)

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
def build_model():
    NN_model = Sequential()

    NN_model.add(Dense(10, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu')) #input_dim = train_data.shape[1]
    #NN_model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(1, kernel_initializer='normal',activation='relu'))

    # ## model compilation
    NN_model.compile(loss='mean_squared_error', optimizer='adam') #'mean_absolute_error', 'mean_squared_error' , , metrics=['accuracy']

    return NN_model

def build_lstm_model( train_set , test_set , input_data):
    train_set = train_set.to_numpy()
    test_set = test_set.to_numpy()
    input_data = input_data.to_numpy()
    train_set = np.reshape(train_set, (train_set.shape[0], 1, train_set.shape[1]))
    test_set = np.reshape(test_set, (test_set.shape[0], 1, test_set.shape[1]))
    input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

    NN_model = Sequential()

    NN_model.add(LSTM(5, batch_input_shape=(1, train_set.shape[1], train_set.shape[2]) , stateful=True))  # input_dim = train_data.shape[1]
    # NN_model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(1))

    # ## model compilation
    NN_model.compile(loss='mean_squared_error',optimizer='adam')  # 'mean_absolute_error', 'mean_squared_error' , , metrics=['accuracy']


    return NN_model , train_set , test_set , input_data



NN_model = build_model()
#NN_model , train_data , test_data , in_data = build_lstm_model(train_data , test_data , in_data)


Model_Checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# ## model fitting
## normal:
model_training = NN_model.fit(train_data, train_labels, epochs=1000, batch_size=32, validation_data=(test_data, test_labels) , callbacks=[Model_Checkpoint]) #validation_split = 0.2
##lstm:
# epochs_no = 1000
# for i in range(epochs_no):
#     loss_his = []
#     val_loss_his = []
#     model_training = NN_model.fit(train_data, train_labels, epochs=1, batch_size=1, validation_data=(test_data, test_labels) , callbacks=[Model_Checkpoint] , shuffle=False) #validation_split = 0.2
#     loss_his.append(model_training.history['loss'])
#     val_loss_his.append(model_training.history['val_loss'])
#     NN_model.reset_states()
# print("Data saved in history: \n", print(model_training.history.keys()))
# print("Model Training History: \n" , model_training.history , "\n")
#
# ## model evaluation
model_evaluation_test = NN_model.evaluate(test_data, test_labels, batch_size=32 , verbose=0)
model_evaluation_train = NN_model.evaluate(train_data, train_labels, batch_size=32 , verbose=0)
model_evaluation_alldata = NN_model.evaluate(in_data, label_data, batch_size=32 , verbose=0)
print("trainset evaluation _on the final model: \n",NN_model.metrics_names,  model_evaluation_train,"\n")
print("Model Evaluation (testset evaluation) _on the final model: \n",NN_model.metrics_names,  model_evaluation_test,"\n")
print("the whole model evaluation _on the final model: \n" , NN_model.metrics_names,  model_evaluation_alldata , "\n")

saved_best_model = load_model('best_model.h5')
train_loss = saved_best_model.evaluate(train_data, train_labels, verbose=0 , batch_size=32)
test_loss = saved_best_model.evaluate(test_data, test_labels, verbose=0 , batch_size=32)
print("train loss evaluation _on the best model: \n",NN_model.metrics_names,  train_loss,"\n")
print("test loss evaluation _on the best model: \n",NN_model.metrics_names,  test_loss,"\n")



### using sklearn MLP
# mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=500, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=.1)
# # Train the classifier
# x_train = train_data.to_numpy()
# y_train = train_labels.to_numpy()
# x_test = test_data.to_numpy()
# y_test = test_labels.to_numpy()
#
# mlp.fit(x_train, y_train)
# # Record the final results for the trained data
# print("Training set score: %f" % mlp.score(x_train, y_train))
# # Test the classifier with the test data
# print("Test set score: %f\n\n###################" % mlp.score(x_test, y_test))

## inverse of data scaling
# train_set = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]))
# train_set_df = pd.DataFrame(train_set)
train_set_df = pd.DataFrame(train_data.to_numpy())
train_label_df= pd.DataFrame(train_labels.to_numpy())
train_set = pd.concat((train_set_df,train_label_df) , axis=1)

# test_set = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]))
# test_set_df = pd.DataFrame(test_set)
test_set_df = pd.DataFrame(test_data.to_numpy())
test_label_df= pd.DataFrame(test_labels.to_numpy())
test_set = pd.concat((test_set_df, test_label_df) , axis=1)

train_set = scaler.inverse_transform(train_set)
test_set = scaler.inverse_transform(test_set)

inversed_train_label = train_set[:,-1]
#inversed_train_in = train_set[: , :-1]
inversed_test_label = test_set[: ,-1]
#inversed_test_in = train_set[: , :-1]

predicted_train_label = NN_model.predict(train_data)
predicted_train_label = pd.DataFrame(predicted_train_label)
predicted_test_label = NN_model.predict(test_data)
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

print("After inverse scaling Train MSE: ", train_score , ", Test MSE: ", test_score)
## end of inverse of data scaling


### evaluation using scikit-Learn cross validation (https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)
def build_lstm_model_cv():
    NN_model = Sequential()

    #NN_model.add(LSTM(10, kernel_initializer='normal', input_shape=(train_data.shape[1], train_data.shape[2])))  # input_dim = train_data.shape[1]
    NN_model.add(LSTM(5, batch_input_shape=(1, train_data.shape[1], train_data.shape[2]), stateful=True))
    # NN_model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(1))

    # ## model compilation
    NN_model.compile(loss='mean_squared_error',optimizer='adam')  # 'mean_absolute_error', 'mean_squared_error' , , metrics=['accuracy']


    return NN_model



# seed = 7
# np.random.seed(seed)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=build_model, epochs=1000, batch_size=32, verbose=0)
#
# kfold = KFold(n_splits=10, random_state=seed)
# scores = cross_val_score(estimator, in_data, label_data, cv=kfold)
# #print("Results: %.10f (%.10f) MSE" % (scores.mean(), scores.std()))
# print("Results: %.10f MSE" % (np.mean(scores)))
# print(scores , "\n")

#### with standerdize data :
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=build_model, epochs=500, batch_size=32, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# scores = cross_val_score(pipeline, in_data, label_data, cv=kfold)
# #print("Standardized: %.10f (%.10f) MSE" % (scores.mean(), scores.std()))
# print("Standardized: %.10f MSE" % (np.mean(scores)))
# print(scores , "\n")

# train_predict = NN_model.predict(train_data)
# test_predict = NN_model.predict(test_data)
#
# train_predict = scaler2.inverse_transform(train_predict)
# train_y = scaler2.inverse_transform(train_target)
#
# test_predict = scaler2.inverse_transform(test_predict)
# test_y = scaler2.inverse_transform(test_target)
#
# train_loss = mean_squared_error(train_predict , train_y)
# test_loss = mean_squared_error(test_predict, test_y)
#
# print("\n train data MSE : ", train_loss)
# print("\n test data MSE : ", test_loss)


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
#
# plt.plot(model_training.history['acc'])
# plt.plot(model_training.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()


### plotting data shape
#plt.plot(data['Cpu'])
# plt.plot(data['DateTime'] , data['Cpu'])
# plt.title("CPU Time Series")
# plt.ylabel('CPU Reading')
# plt.xlabel('DataTime')
# plt.show()

# plt.plot(label_data , label='actual series')
# plt.title('Actual Series')
# plt.ylabel('CPU')
# plt.xlabel('time')
# plt.show()

### plot training/ test prediction
train_predict = NN_model.predict(train_data )
test_predict = NN_model.predict(test_data )

train_labels = train_labels.to_numpy()
test_labels = test_labels.to_numpy()

fig = plt.figure()
plt.subplot(1,2, 1)
plt.plot(train_labels  , label='train actual')
plt.plot(train_predict , label='train predict')
#plt.yticks(np.arange(0, 0.8, step=0.1))

plt.subplot(1, 2,2)
plt.plot(test_labels, label='test actual')
plt.plot(test_predict , label="test predict")
#plt.yticks(np.arange(0, 0.8, step=0.1))

plt.title(' Actual and Predicted ')
plt.ylabel('CPU')
plt.xlabel('time')
plt.legend(loc='upper left')
plt.show()


#### try another plotting
data_predict = NN_model.predict(in_data)
plt.plot(label_data  , label='actual timeseries')
plt.plot(data_predict , label='predicted timeseries')
plt.title(' Actual and Predicted timeseries')
plt.ylabel('CPU')
plt.xlabel('time')
plt.legend(loc='upper right')
plt.show()
