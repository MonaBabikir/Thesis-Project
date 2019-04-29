## https://www.kaggle.com/inspector/keras-hyperopt-example-sketch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error , mean_absolute_error

from keras.callbacks import Callback

import ml_metrics

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials


np.random.seed(7) # to make the result reproducable.

cpu_pow_data = pd.read_csv("./real_data_prepared/epouta/e101_epouta_csc_fi.csv", header=None)
cpu_pow_data.columns = ["Time" , "CPU" , "Power"]
cpu_data = cpu_pow_data["CPU"]
pow_data = cpu_pow_data["Power"]
net_data = pd.read_csv("./real_data_prepared/epouta_net/e101_epouta_csc_fi.csv", header=None)

data = pd.concat([net_data , cpu_data , pow_data], axis=1)
data.columns = ["Time" , "Net" , "CPU" , "Power"]
data.drop(["Time"] , axis=1 , inplace=True)


## data scaling
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)
data.columns = ["Net" , "CPU" , "Power"]
##end of data scaling


label_data = data["Power"]
in_data = data.drop(["Power"] , axis=1)


label_data = label_data.to_numpy()
in_data = in_data.to_numpy()

## train data split in order
# n = int(float(data.shape[0]) * 0.8)
# train_data = in_data[:n]
# train_labels = label_data[:n]
# test_data = in_data[n:]
# test_labels = label_data[n:]

## train data split in randomly
# print(type(label_data))
train_data, test_data, train_labels, test_labels = train_test_split(in_data, label_data, test_size=0.2, random_state=42)



space = {'choice':

             hp.choice('num_layers',
                       [
                           {'layers': 'two',

                            }
                           ,

                           {'layers': 'three',

                            'units3': hp.choice('units3', [20 , 50 ,100 , 200 , 300]),
                            # 'dropout3': hp.choice('dropout3', [0.25, 0.5, 0.75])

                            }
                           ,

                           {'layers': 'four',

                            'units34': hp.choice('units34', [20 , 50 ,100 , 200 , 300]),

                            'units4': hp.choice('units4', [20 , 50 ,100 , 200 , 300]),
                            # 'dropout3': hp.choice('dropout3', [0.25, 0.5, 0.75])

                            }

                       ]),

         'units1': hp.choice('units1', [20 , 50 ,100 , 200 , 300]),
         'units2': hp.choice('units2', [20 , 50 ,100 , 200 , 300]),

         # 'dropout1': hp.choice('dropout1', [0.25, 0.5, 0.75]),
         # 'dropout2': hp.choice('dropout2', [0.25, 0.5, 0.75]),

         'batch_size': hp.choice('batch_size', [32 ,16]),

         'nb_epochs': 500,
         'optimizer': hp.choice('optimizer', ['adam']),
         'activation': hp.choice('activation', ['relu' ,'tanh'])

         }


# Objective function that hyperopt will minimize


def objective(params):
    import ml_metrics

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta
    from keras.layers.normalization import BatchNormalization
    from keras.callbacks import Callback

    print('Params testing: ', params)
    print('\n ')
    model = Sequential()
    model.add(Dense(params['units1'], input_dim=train_data.shape[1], kernel_initializer="normal"))
    model.add(Activation(params['activation']))
    # model.add(Dropout(params['dropout1']))
    # model.add(BatchNormalization())

    model.add(Dense(params['units2'], kernel_initializer="normal"))
    model.add(Activation(params['activation']))
    # model.add(Dropout(params['dropout2']))
    # model.add(BatchNormalization())

    if params['choice']['layers'] == 'three':
        model.add(Dense(params['choice']['units3'], kernel_initializer="normal"))
        model.add(Activation(params['activation']))
        # model.add(Dropout(params['choice']['dropout3']))
        # model.add(BatchNormalization())
        patience = 25
    elif params['choice']['layers'] == 'four':
        model.add(Dense(params['choice']['units34'], kernel_initializer="normal"))
        model.add(Activation(params['activation']))
        model.add(Dense(params['choice']['units4'], kernel_initializer="normal"))
        model.add(Activation(params['activation']))
        # model.add(Dropout(params['choice']['dropout3']))
        # model.add(BatchNormalization())
        patience = 25
    else:
        patience = 15

    model.add(Dense(1, kernel_initializer="normal"))  # End in a single output node for regression style output
    model.compile(loss='mean_absolute_error', optimizer=params['optimizer'])

    # object of class for call back early stopping
    # val_call = clsvalidation_kappa(validation_data=(X_val, y_val), patience=patience,
    #                                filepath='"../input/best.h5')  # instantiate object

    # Model_Checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # includes the call back object
    model.fit(train_data, train_labels , epochs=params['nb_epochs'], batch_size=params['batch_size'], verbose=0) # callbacks=[Model_Checkpoint]

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

    predicted_train_label = model.predict(train_data)
    predicted_train_label = pd.DataFrame(predicted_train_label)
    predicted_test_label = model.predict(test_data)
    predicted_test_label = pd.DataFrame(predicted_test_label)

    inversed_predicted_train_label = pd.concat((train_set_df, predicted_train_label), axis=1)
    inversed_predicted_train_label = scaler.inverse_transform(inversed_predicted_train_label)
    inversed_predicted_train_label = inversed_predicted_train_label[:, -1]

    inversed_predicted_test_label = pd.concat((test_set_df, predicted_test_label), axis=1)
    inversed_predicted_test_label = scaler.inverse_transform(inversed_predicted_test_label)
    inversed_predicted_test_label = inversed_predicted_test_label[:, -1]

    # print("\n min value in the train target and test target after rescaling:", np.amin(inversed_train_label), " , ",np.amin(inversed_test_label))
    # print("\n max value in the train target and test target after rescaling:", np.amax(inversed_train_label), " , ",np.amax(inversed_test_label))

    train_score = mean_squared_error(inversed_train_label, inversed_predicted_train_label)
    test_score = mean_squared_error(inversed_test_label, inversed_predicted_test_label)

    # print("After inverse scaling Train MSE: ", train_score, ", Test MSE: ", test_score)

    ## end of inverse of data scaling

    # predict the test set
    # preds = model.predict(X_test, batch_size=32, verbose=0)
    #
    # predClipped = np.clip(np.round(preds.astype(int).ravel()), 1, 8)  # simple rounding of predictionto int
    # score = ml_metrics.quadratic_weighted_kappa(y_test.values.ravel(), predClipped)

    # return {'loss': score * (-1), 'status': STATUS_OK, 'rounds': val_call.best_rounds}
    return {'loss': test_score, 'status': STATUS_OK}


trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=100)

print("\n")
print(best)
print("\n")
print(trials.best_trial)
print("\n")
print(trials.results)


####
# tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
#                             'iteration': tpe_trials.idxs_vals[0]['x'],
#                             'x': tpe_trials.idxs_vals[1]['x']})
#
# tpe_results.head()