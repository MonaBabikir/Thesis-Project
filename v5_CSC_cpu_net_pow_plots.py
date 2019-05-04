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

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

cpu_pow_data = pd.read_csv("./real_data_prepared/epouta/e23_epouta_csc_fi.csv", header=None)
cpu_pow_data.columns = ["Time" , "CPU" , "Power"]
cpu_data = cpu_pow_data["CPU"]
pow_data = cpu_pow_data["Power"]
net_data = pd.read_csv("./real_data_prepared/epouta_net/e23_epouta_csc_fi.csv", header=None)

# data = pd.concat([net_data , cpu_data , pow_data], axis=1)
# data.columns = ["Time" , "Net" , "CPU" , "Power"]
# data.drop(["Time"] , axis=1 , inplace=True)
net_data.columns = ["Time" , "Net"]
net_data = net_data["Net"]

NN_model = load_model("./models_reg_pow/final_model_e23_epouta_csc_fi.csv.h5")

data = pd.concat([net_data , cpu_data , pow_data] , axis=1)
## data scaling
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)
data.columns = ["Net" , "CPU" , "Power"]
##end of data scaling

in_data = data.drop(["Power"], axis=1)
pred = NN_model.predict(in_data)
inv_pred = pd.concat([in_data , pd.DataFrame(pred)] , axis=1)
inv_pred = scaler.inverse_transform(inv_pred)
inv_pred = inv_pred[: , -1]

# print(inv_pred.shape)
# print(inv_pred)
# print(cpu_data.to_numpy())
# print(net_data.to_numpy())



plot_data = [
    go.Contour(
        # z=[[10, 10.625, 12.5, 15.625, 20],
        #    [5.625, 6.25, 8.125, 11.25, 15.625],
        #    [2.5, 3.125, 5., 8.125, 12.5],
        #    [0.625, 1.25, 3.125, 6.25, 10.625],
        #    [0, 0.625, 2.5, 5.625, 10]],
        # z=[1 , 2, 4, 3 , 6, 7 , 9],
        # x=[-9, -6, -5 , -3, -1],
        # y=[0, 1, 4, 5, 7]

        z= inv_pred[0:400]
        ,
        x= cpu_data.to_numpy()[0:400],
        y= net_data.to_numpy()[0:400]
    )]

#plotly.offline.plot(plot_data)
#############################################################################
# scaler2 = StandardScaler()
def f(x , y): # x: CPU , y: Net
    # print(x)
    x = pd.DataFrame(np.array([x]))
    y = pd.DataFrame(np.array([y]))
    in_data = pd.concat([y , x] , axis=1)
    # in_data = scaler2.fit_transform(in_data)
    z = NN_model.predict(in_data)

    z = pd.DataFrame(z)
    xyz = pd.concat([y ,x , z] , axis=1)
    xyz = scaler.inverse_transform(xyz)
    xyz = pd.DataFrame(xyz)
    xyz.columns = ["y" , "x" , "z"]
    x = xyz["x"]
    y = xyz["y"]
    z = xyz["z"]
    return x ,y ,z

vf = np.vectorize(f)

##scaled data
# x = data["CPU"][0:2000]
# y = data["Net"][0:2000]

##### with random numbers
x = np.random.uniform(0.0 , 100.0 , 10000)
y = np.random.uniform(11000 , 500000000 , 10000)
zr = np.random.uniform(0.0 , 400.0 , 10000)

xdf = pd.DataFrame(x)
ydf = pd.DataFrame(y)
zdf = pd.DataFrame(zr)

xyz = pd.concat([ydf ,xdf , zdf] , axis=1)

#scalerxy = StandardScaler()
xyz = scaler.transform(xyz)
xyz = pd.DataFrame(xyz)
xyz.columns = [ "Net" , "CPU", "Power"]

x = xyz["CPU"]
y = xyz["Net"]

##### end of random numbers

# print(x)
# print(y)
X, Y = np.meshgrid(x, y)
# print(X.shape)

X , Y , z = vf(X , Y)
# print(X)
# print(Y)
# print(z)

plt.contourf( X , Y , z)
plt.colorbar()
plt.show()


# print(f(-0.89115062 , -0.62514045))

