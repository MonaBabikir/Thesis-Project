import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense , LSTM , Dropout

from keras import optimizers

np.random.seed(7) # to make the result reproducable.

ins_name = "d40"
cpu_data = pd.read_csv("./real_data_prepared_Upp/CPU/" + ins_name + ".csv")
mem_data = pd.read_csv("./real_data_prepared_Upp/Mem/" + ins_name + ".csv")
disk_space_data = pd.read_csv("./real_data_prepared_Upp/Disk_space/" + ins_name + ".csv")
disk_io_data = pd.read_csv("./real_data_prepared_Upp/Disk_io/" + ins_name + ".csv")
# temp_data = pd.read_csv("./real_data_prepared_Upp/Temp/" + ins_name + ".csv")
temp_data = pd.read_csv("./real_data_prepared_Upp/Temp_mean/" + ins_name + ".csv")

data = pd.concat([cpu_data , mem_data , disk_space_data , disk_io_data , temp_data ] , axis=1)
data.columns = ["Time" , "CPU" , "Time2" , "Mem" , "Time3" ,"Disk_Space" , "Time4" , "Disk_IO" , "Time5" , "Temp"]
data.drop(["Time" ,"Time2" , "Time3" , "Time4" , "Time5"] , axis=1 , inplace= True)

# plt.plot(data["Temp"] , label='actual Temp data')
# plt.show()
################################################ scaling on training data only.
# label_data = data["Power"]
# in_data = data.drop(["Power"] , axis=1)
#
# train_data, test_data, train_labels, test_labels = train_test_split(in_data, label_data, test_size=0.2 , random_state=42) #, random_state=42
# scaler = StandardScaler()
# train_set = pd.concat([train_data , train_labels] , axis=1)
# train_set = scaler.fit_transform(train_set)
# train_set = pd.DataFrame(train_set)
# train_set.columns = ["Net" , "CPU" , "Power"]
# train_labels =  train_set["Power"]
# train_data = train_set.drop(["Power"] , axis=1)
#
# test_set = pd.concat([test_data , test_labels] , axis=1)
# test_set = scaler.transform(test_set)
# test_set = pd.DataFrame(test_set)
# test_set.columns = ["Net" , "CPU" , "Power"]
# test_labels = test_set["Power"]
# test_data = test_set.drop(["Power"] , axis=1)
#
# train_labels = train_labels.to_numpy()
# train_data = train_data.to_numpy()
# test_labels = test_labels.to_numpy()
# test_data = test_data.to_numpy()
#
#
# ### after scaling:
# data = scaler.transform(data)
# data = pd.DataFrame(data)
# data.columns = ["Net" , "CPU" , "Power"]
# label_data = data["Power"]
# in_data = data.drop(["Power"] , axis=1)
#

################################################

####ttttttttttttttttttttttttttt scaling on whole data set

#data.drop([ "Disk_Space"], axis=1 , inplace=True)

print(data)

print("\n min value in label data", np.amin(data['Temp']))
print(data.shape)

## data scaling
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)
data.columns = ["CPU" , "Mem" ,"Disk_Space" , "Disk_IO" , "Temp"]
#data.columns = ["CPU" , "Mem" , "Disk_IO" ,"Temp"]
##end of data scaling



label_data = data["Temp"]
in_data = data.drop(["Temp"] , axis=1)

label_data = label_data.to_numpy()
in_data = in_data.to_numpy()

## train data split in randomly
print(type(label_data))
train_data, test_data, train_labels, test_labels = train_test_split(in_data, label_data, test_size=0.2 , random_state=42) #, random_state=42

# #####ttttttttttttttttt

print(train_data)
print(train_data.shape)



# ## min/max values for the target
print("\n min value in the train target and test target :", np.amin(train_labels) , " , " , np.amin(test_labels))
print("\n max value in the train target and test target :", np.amax(train_labels) , " , " , np.amax(test_labels))


# ## model building
def build_model():
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


NN_model = build_model()

#Model_Checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# ## model fitting
## normal:
model_training = NN_model.fit(train_data, train_labels, epochs=5000, batch_size=32, validation_data=(test_data, test_labels) ) #validation_split = 0.2 , , callbacks=[Model_Checkpoint]

# ## model evaluation
# model_evaluation_test = NN_model.evaluate(test_data, test_labels, batch_size=32 , verbose=0)
# model_evaluation_train = NN_model.evaluate(train_data, train_labels, batch_size=32 , verbose=0)
# model_evaluation_alldata = NN_model.evaluate(in_data, label_data, batch_size=32 , verbose=0)
# print("trainset evaluation _on the final model: \n",NN_model.metrics_names,  model_evaluation_train,"\n")
# print("Model Evaluation (testset evaluation) _on the final model: \n",NN_model.metrics_names,  model_evaluation_test,"\n")
# print("the whole model evaluation _on the final model: \n" , NN_model.metrics_names,  model_evaluation_alldata , "\n")
#
# saved_best_model = load_model('best_model.h5')
# train_loss = saved_best_model.evaluate(train_data, train_labels, verbose=0 , batch_size=1)
# test_loss = saved_best_model.evaluate(test_data, test_labels, verbose=0 , batch_size=1)
# print("train loss evaluation _on the best model: \n",NN_model.metrics_names,  train_loss,"\n")
# print("test loss evaluation _on the best model: \n",NN_model.metrics_names,  test_loss,"\n")

## inverse of data scaling
train_set_df = pd.DataFrame(train_data)
train_label_df= pd.DataFrame(train_labels)
train_set = pd.concat((train_set_df,train_label_df) , axis=1)

test_set_df = pd.DataFrame(test_data)
test_label_df= pd.DataFrame(test_labels)
test_set = pd.concat((test_set_df, test_label_df) , axis=1)


train_set = scaler.inverse_transform(train_set)
test_set = scaler.inverse_transform(test_set)

inversed_train_label = train_set[:,-1]
inversed_test_label = test_set[: ,-1]

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

train_score_mae = mean_absolute_error(inversed_train_label, inversed_predicted_train_label)
test_score_mae = mean_absolute_error(inversed_test_label, inversed_predicted_test_label)

print("After inverse scaling Train MAE: ", train_score_mae , ", Test MAE: ", test_score_mae)

## end of inverse of data scaling



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

# plt.plot(inversed_train_label)
# plt.show()
# plt.plot(inversed_predicted_train_label)
# plt.show()


### plot training/ test prediction
train_predict = NN_model.predict(train_data )
test_predict = NN_model.predict(test_data )

fig = plt.figure()
plt.subplot(1,2, 1)
# plt.plot(train_labels  , label='train actual')
# plt.plot(train_predict , label='train predict')
plt.plot(inversed_train_label  , label='train actual')
plt.plot(inversed_predicted_train_label , label='train predict')
#plt.yticks(np.arange(0, 0.8, step=0.1))

plt.subplot(1, 2,2)
# plt.plot(test_labels.to_numpy(), label='test actual')
# plt.plot(test_predict , label="test predict")
plt.plot(inversed_test_label, label='test actual')
plt.plot(inversed_predicted_test_label , label="test predict")
#plt.yticks(np.arange(0, 0.8, step=0.1))

plt.title(' Actual and Predicted ')
plt.ylabel('Temperature')
plt.xlabel('time')
plt.legend(loc='lower left')
plt.show()


### plotting data as sequence,
data_predict = NN_model.predict(in_data)
data_predict = pd.DataFrame(data_predict)
in_data = pd.DataFrame(in_data)
inversed_data_predict = pd.concat((in_data , data_predict) , axis=1)
inversed_data_predict = scaler.inverse_transform(inversed_data_predict)
inversed_data_predict = inversed_data_predict[: , -1]

inversed_data_label = scaler.inverse_transform(data)
inversed_data_label = inversed_data_label[: , -1]

plt.plot(inversed_data_label , label='actual Temp data')
plt.show()

plt.plot(inversed_data_label , label='actual Temp data')
plt.plot(inversed_data_predict , label='predicted Temp data')
plt.title(' Actual and Predicted ')
plt.ylabel('Temperature')
plt.xlabel('time')
plt.legend(loc='lower right')
plt.show()