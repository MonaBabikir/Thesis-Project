#####
#This code for learning and practice purposes and it is taken from ( https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33 )
#####

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
#import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
#from xgboost import XGBRegressor


def get_data():
    #training data
    train_data_path = "train.csv"
    train = pd.read_csv(train_data_path)

    #test data
    test_data_path = "test.csv"
    test = pd.read_csv(test_data_path)

    return train , test


#load training and test data
train_data , test_data = get_data()


#this may be useful when computing the columns without nan values therefore if any column that has nan values either in the training or test data will be ignored.
def get_combined_data():
  #reading train data
  train , test = get_data()

  target = train.SalePrice
  train.drop(['SalePrice'],axis = 1 , inplace = True)

  combined = train.append(test)
  combined.reset_index(inplace=True)
  combined.drop(['index', 'Id'], inplace=True, axis=1)
  return combined, target

#Combine train and test data to process them together
combined, target = get_combined_data()


# function to remove nan columns (columns with non empty values)
def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type :
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


num_cols = get_cols_with_no_nans(combined , 'num')
cat_cols = get_cols_with_no_nans(combined , 'no_num')
print ('Number of numerical columns with no nan values :',len(num_cols))
print ('Number of nun-numerical columns with no nan values :',len(cat_cols))

#create a new dataframe that contains only numerical and categorical data
combined = combined[num_cols + cat_cols]

#one hot encoded for categorical features
def oneHotEncode(df,colNames):
    for col in colNames:
        if( df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)

            #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df

print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
combined = oneHotEncode(combined, cat_cols)
print('There are {} columns after encoding categorical features'.format(combined.shape[1]))


#splitting the data to training and test set for further processing
def split_combined():
    global combined
    train = combined[:1460]
    test = combined[1460:]

    return train, test


train, test = split_combined()


###############################
# training Model
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


## Define a checkpoint callback :
# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]

#Training the model
# NN_model.fit(train, target, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
NN_model.fit(train, target, epochs=50, batch_size=32, validation_split = 0.2)

from keras import backend as K
inp = NN_model.input
outputs = [layer.output for layer in NN_model.layers] ##outputs of each layer
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]

print("\n\n layers output:\n")
print(outputs)
print("\n", functors)


# Load wights file of the best model :
#after the training we see the value where the model doesn't improve after it an then choose that file to be the weights of the model.
# this should be separated from the raining to choose the file after that.

# wights_file = 'Weights-440--18738.19831.hdf5' # choose the best checkpoint
# NN_model.load_weights(wights_file) # load it
# NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
