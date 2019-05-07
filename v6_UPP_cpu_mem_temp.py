# import glob
# import csv
# import os
# import numpy as np
# import sklearn
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
# import itertools
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_squared_error , mean_absolute_error
# from sklearn.model_selection import train_test_split
import pandas as pd
# from sklearn.preprocessing import StandardScaler , MinMaxScaler
# from keras.models import Sequential, load_model
# from keras.layers import Dense , LSTM
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.pipeline import Pipeline
# from keras.callbacks import ModelCheckpoint

import json

def prepare_cpu(cpu_file_path):
    # f = open("./real_data/uppmax/cpu", "r")
    f = open(cpu_file_path, "r")
    cpu_data = json.load(f)

    # indented_cpu_data = json.dumps(cpu_data, indent=4) ## can be used to print indented json data
    print(len(cpu_data['data']['result']))

    # print(cpu_data['data']['result'][0])

    # c= 0
    # for i in cpu_data['data']['result']:
    #
    #     if i['metric']['instance'] == 'd102:19999': #"instance":"d102:19999"
    #         #print(i)
    #         c +=1

    # print(c)

    ####### to seperate instances observations from each other. and save it to file.
    # ins_dic = {}
    # # c = 0
    # for i in cpu_data['data']['result']:
    #
    #     if i['metric']['instance'] in ins_dic:
    #         ins_dic[i['metric']['instance']][i['metric']['dimension']] = i['values']
    #     else:
    #         ins_dic[i['metric']['instance']] = {}
    #         ins_dic[i['metric']['instance']][i['metric']['dimension']] = i['values']
    #     # c +=1
    #
    #     # if c == 5:
    #     #     break
    #
    # f = open('dict_cpu.txt', 'w')
    # f.write(str(ins_dic))
    # f.close()
    # # print(ins_list)
    # print(len(ins_dic))

    ############ end of instance seperation

    ######### build DF and csv file for each instance
    # c=0
    # for key , val in ins_dic.items():
    #     print(len(ins_dic))
    #     print(key)
    #     print(len(val))
    #     # print(val)
    #     ins_df = pd.DataFrame()
    #     ins_cols = []
    #     for k ,v in val.items():
    #         # print(k)
    #         cpu_comp_df = pd.DataFrame(v)
    #         ins_df = pd.concat([ins_df , cpu_comp_df[1]] , axis=1)
    #         ins_cols.append(k)
    #
    #     ins_df.columns = ins_cols
    #     ins_df.to_csv("./df_"+ key.split(':')[0]+".csv")
    #
    #     c +=1
    #     if c ==1:
    #         break

    ######## end of building DF

    ###################
    ins_dic = {}
    ## next loop create dictinary where the 'keys' are instance name and 'values' are all instance value from different dimension
    for i in cpu_data['data']['result']:

        if i['metric']['instance'] in ins_dic:
            ins_dic[i['metric']['instance']].extend(i['values'])
        else:
            ins_dic[i['metric']['instance']] = i['values']

    print("Number of machine : " , len(ins_dic))
    #### to write the ins_dec with instance name as key and all values from different dimensions as value.
    # f = open('dict2.txt', 'w')
    # f.write(str(ins_dic))
    # f.close()

    # c = 0
    for ins_name, val in ins_dic.items():
        ins_df = pd.DataFrame(val)
        ins_df.columns = ["TimeStamp", "CPU"]
        ins_df["CPU"] = ins_df["CPU"].astype(float)
        ins_df = ins_df.groupby(by='TimeStamp')['CPU'].sum()
        ins_df.to_csv("./real_data_prepared_Upp/CPU/" + ins_name.split(':')[0] + ".csv" , header=True)

        # c += 1
        # if c == 1:
        #     break

    # print(ins_list)
    # print(len(ins_dic))
    # print(len(ins_dic['d102:19999']))  # 1035 * 17
    ###################


def prepare_temp(temp_file_path):
    f = open(temp_file_path, "r")
    cpu_data = json.load(f)

    print(len(cpu_data['data']['result']))

    ####*************** to know different dimension:
    # ins_dic = {}
    #
    # for i in cpu_data['data']['result']:
    #
    #     if i['metric']['instance'] in ins_dic:
    #         ins_dic[i['metric']['instance']][i['metric']['dimension']] = i['values']
    #     else:
    #         ins_dic[i['metric']['instance']] = {}
    #         ins_dic[i['metric']['instance']][i['metric']['dimension']] = i['values']
    #
    #
    # f = open('dict_temp.txt', 'w')
    # f.write(str(ins_dic))
    # f.close()
    ####*****************

    ins_dic = {}
    ## next loop create dictinary where the 'keys' are instance name and 'values' are all instance value from different dimension
    for i in cpu_data['data']['result']:

        if i['metric']['instance'] in ins_dic:
            ins_dic[i['metric']['instance']].extend(i['values'])
        else:
            ins_dic[i['metric']['instance']] = i['values']

    print("Number of machine : ", len(ins_dic))

    for ins_name, val in ins_dic.items():
        ins_df = pd.DataFrame(val)
        ins_df.columns = ["TimeStamp", "Temp"]
        ins_df["Temp"] = ins_df["Temp"].astype(float)
        ins_df = ins_df.groupby(by='TimeStamp')['Temp'].sum()
        ins_df.to_csv("./real_data_prepared_Upp/Temp/" + ins_name.split(':')[0] + ".csv" , header=True)


def prepare_feature(feature_file_path , feature):
    f = open(feature_file_path, "r")
    cpu_data = json.load(f)

    print(len(cpu_data['data']['result']))

    ####*************** to know different dimension:
    # ins_dic = {}
    #
    # for i in cpu_data['data']['result']:
    #
    #     if i['metric']['instance'] in ins_dic:
    #         ins_dic[i['metric']['instance']][i['metric']['dimension']] = i['values']
    #     else:
    #         ins_dic[i['metric']['instance']] = {}
    #         ins_dic[i['metric']['instance']][i['metric']['dimension']] = i['values']
    #
    #
    # f = open('dict_'+feature.lower()+'.txt', 'w')
    # f.write(str(ins_dic))
    # f.close()
    ####*****************

    ins_dic = {}
    ## next loop create dictinary where the 'keys' are instance name and 'values' are all instance value from different dimension
    for i in cpu_data['data']['result']:

        if i['metric']['instance'] in ins_dic:
            ins_dic[i['metric']['instance']].extend(i['values'])
        else:
            ins_dic[i['metric']['instance']] = i['values']

    print("Number of machine : ", len(ins_dic))

    for ins_name, val in ins_dic.items():
        ins_df = pd.DataFrame(val)
        ins_df.columns = ["TimeStamp", feature]
        ins_df[feature] = ins_df[feature].astype(float)
        ins_df = ins_df.groupby(by='TimeStamp')[feature].sum()
        ins_df.to_csv("./real_data_prepared_Upp/"+ feature +"/" + ins_name.split(':')[0] + ".csv", header=True)


#prepare_cpu("./real_data/uppmax/cpu")

#prepare_temp("./real_data/uppmax/temperature")

prepare_feature("./real_data/uppmax/disk_io" , "Disk_io")
