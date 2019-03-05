from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

## Get the Dataset
#data = pd.read_csv("1.csv", error_bad_lines=False)
data = pd.read_excel('combined_data.xlsx', sheet_name='Sheet1')
print(data)
#data.drop(["\tNetwork transmitted throughput [KB/s]" , "\tNetwork received throughput [KB/s]" ,
#                  "\tDisk write throughput [KB/s]" , "\tDisk read throughput [KB/s]" , "Timestamp [ms]"], axis=1 , inplace = True)

train_set, test_set = train_test_split(data, test_size=0.2)
print(data.shape, " ", train_set.shape , " ", test_set.shape)
print(list(data)) # print columns names
#print(data.iloc[1]) # print the value of one row.
#print(data.dtypes) # print data types of each column

## specifying training and test lables
train_target = train_set["\tCPU usage [MHZ]"]
train_set.drop("\tCPU usage [MHZ]" , axis=1 , inplace = True)

test_target = test_set["\tCPU usage [MHZ]"]
test_set.drop("\tCPU usage [MHZ]" , axis=1 , inplace = True)


##some plots
plt.plot(data["\tCPU capacity provisioned [MHZ]"], 'r--' , data["\tCPU usage [MHZ]"], 'bs'
         , data["\tMemory capacity provisioned [KB]"] , 'g^' , data["\tMemory usage [KB]"] , 'ro')
#plt.show()
##end of plots

print("input features : ", list(train_set))
## Selection of best feature using RFE (Recursive Feature Elimination) "https://machinelearningmastery.com/feature-selection-machine-learning-python/"
# model = LinearRegression()
# rfe = RFE(model)
# selction = rfe.fit(train_set, train_target)
# print("Num Features: ", selction.n_features_)
# print("Selected Features: ",  selction.support_)
# print("Feature Ranking: " , selction.ranking_)

## Dimentionality reduction using PCA, "https://www.kaggle.com/xyz8983/pca-2-layer-neural-network-in-python" ,
pca = PCA()
# pca_fit = pca.fit(train_set, 3)
# train_set = pca_fit.transform(train_set)
# test_set = pca_fit.transform(test_set)

train_set = pca.fit_transform(train_set)
test_set = pca.transform(test_set)

# #print("train and test new component: ", list(train_set), "\n" ,list(test_set))
print("set shape after pca :", train_set.shape , "  ", test_set.shape)

## dropping irrelevant features
#train_set.drop(["Timestamp [ms]", "\tCPU cores" , "\tCPU capacity provisioned [MHZ]" , "\tMemory usage [KB]" , "\tDisk read throughput [KB/s]"], axis=1, inplace=True)
#test_set.drop(["Timestamp [ms]", "\tCPU cores" , "\tCPU capacity provisioned [MHZ]" , "\tMemory usage [KB]" , "\tDisk read throughput [KB/s]"], axis=1, inplace=True)

#print("input features after dropping some: ", list(train_set))

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
plot_model(NN_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

## Compile the network : To configure network for training
# "https://keras.io/models/model/#compile" description of 'compile' function.
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


## Train the model
# "https://keras.io/models/sequential/" description of 'fit' function which is used for model training.
model_training = NN_model.fit(train_set, train_target, epochs=1500, batch_size=32, validation_split = 0.2)
print("Model Training History: \n" , model_training.history , "\n")

## Evaluate the
model_evaluation = NN_model.evaluate(test_set, test_target, batch_size=32 , verbose=1)
print("Model Evaluation: \n",NN_model.metrics_names,  model_evaluation,"\n")


## Use the model for Prediction
model_prediction_1 = NN_model.predict(np.array([[1376314846 ,2, 5851.999076 , 2.4 , 3145728.0, 383776.5333333333, 2.466666666666667 , 20.6 , 5.2 , 5.333333333333333]])) # actual value = 140.447977824
print("model prediction : ", model_prediction_1)

model_prediction_2 = NN_model.predict(np.array([[1376315746 ,8, 20799.994544 , 1.5583333333333333 , 5605376.0, 134217.6, 0.0 , 15.266666666666667 , 4.8 , 0.06666666666666667]])) # actual value = 324.13324831066666
print("model prediction : ", model_prediction_2)

model_prediction_3 = NN_model.predict(np.array([[0.0 ,0.0, 0.0 , 0.0 , 0.0, 0.0, 0.0 , 0.0 , 0.0 , 0.0]])) # actual value =
print("model prediction : ", model_prediction_3)

