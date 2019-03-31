### converting timeseries data to supervised learning problem (https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)


import os
import pandas as pd
from datetime import datetime

## data preprocessing to put all the data together
def aggregate_data():
    data_path = "./real_data/"
    dir_names = os.listdir(data_path)
    agg_data = []
    for i in range(len(dir_names)):
        file_path = data_path +  dir_names[i]
        cpu_df = pd.read_csv(file_path +"/cpu.csv", delimiter=';')
        net_df = pd.read_csv(file_path +"/net.csv", delimiter=';')
        power_df = pd.read_csv(file_path +"/power.csv", delimiter=';')
        data = pd.concat([cpu_df, net_df, power_df], axis=1)
        agg_data.append(data)

    agg_data = pd.concat(agg_data)
    agg_data.to_csv("./aggregated_data.csv")

#aggregate_data()
## end of preprocessing

# data = pd.read_csv("./real_data/10-27/cpu.csv" , delimiter=';')
# print(list(data))
# print(len(data))
# print(len(data['Series'].unique()))
# print((data['Series'].unique())[1].split('.'))
# x = (data['Series'].unique())[1].split('.')[4] == (data['Series'].unique())[10].split('.')[4]
# print(x)
#
#
# print(data['Time'][0] == data['Time'][146])
# print(len(data['Time'].unique()))
#
# df = pd.DataFrame(data['Series'])
# df = df.applymap(lambda x: x.split('.')[2])
# print(df.head)
# print(len(df))
#
# data = pd.concat([df , data] ,axis=1)
#
# data.columns = ['Series' , 'Full_Series', 'Time', 'Value']
# data.drop(['Full_Series'], axis=1, inplace=True)
# group = data.groupby('Series')
# print(list(data), "  ", len(data))
#
# print(group.describe())
#
# for series, df_series in data.groupby('Series'):
#     print(series)
#     #print(df_series)
#     agg_df = df_series.groupby(by='Time')['Value'].sum()
#     print(len(agg_df))
#
#     with open('./real_data_prepared/' + series + '.csv' , 'a') as f:
#         agg_df.to_csv(f, header=False)



def onefile_prepare(cpu_path, power_path):
    #cpu_data = pd.read_csv("./real_data/10-27/cpu.csv" , delimiter=';')
    #power_data = pd.read_csv("./real_data/10-27/power.csv", delimiter=';')
    cpu_data = pd.read_csv(cpu_path, delimiter=';')
    power_data = pd.read_csv(power_path, delimiter=';')

    #print(list(data))
    #print(len(data))

    cpu_df = pd.DataFrame(cpu_data['Series'])
    power_df = pd.DataFrame(power_data['Series'])

    cpu_df = cpu_df.applymap(lambda x: x.split('.')[2])
    power_df = power_df.applymap(lambda x: x.split('.')[2])
    #print(df.head)
    #print(len(df))

    cpu_data = pd.concat([cpu_df, cpu_data], axis=1)
    power_data = pd.concat([power_df, power_data], axis=1)

    cpu_data.columns = ['Series', 'Full_Series', 'Time', 'Value']
    cpu_data.drop(['Full_Series'], axis=1, inplace=True)

    power_data.columns = ['Series', 'Full_Series', 'Time', 'Value']
    power_data.drop(['Full_Series'], axis=1, inplace=True)

    cpu_data_list = []
    for series, df_series in cpu_data.groupby('Series'):
        #print(series)
        # print(df_series)
        agg_df = df_series.groupby(by='Time')['Value'].sum()
        #print(len(agg_df))

        cpu_data_list.append(agg_df)
        # with open('./real_data_prepared/power/' + series + '.csv', 'a') as f:
        #     agg_df.to_csv(f, header=False)

    power_data_list = []
    series_list = []
    for series, df_series in power_data.groupby('Series'):
        #print(series)
        series_list.append(series)
        # print(df_series)
        agg_df = df_series.groupby(by='Time')['Value'].sum()
        #print(len(agg_df))

        power_data_list.append(agg_df)
        # with open('./real_data_prepared/power/' + series + '.csv', 'a') as f:
        #     agg_df.to_csv(f, header=False)

    if len(cpu_data_list) == len(power_data_list) == len(series_list):
        for i in range(len(cpu_data_list)):
            c_p_agg_df = pd.concat([cpu_data_list[i] , power_data_list[i]], axis=1)
            #print("length of aggregated = ", len(c_p_agg_df))
            with open('./real_data_prepared/epouta/' + series_list[i] + '.csv', 'a') as f:
                c_p_agg_df.to_csv(f, header=False)


    else:
        print("\nSorry there is a problems with the files :( \n")


def allfiles_prepare():
    data_path = "./real_data/epouta/"
    dir_names = os.listdir(data_path)
    for i in range(len(dir_names)):
        print(" In directory : ", dir_names[i])
        file_path = data_path +  dir_names[i]
        cpu_path = file_path + "/cpu.csv"
        power_path = file_path +"/power.csv"
        net_path = file_path +"/net.csv"

        onefile_prepare(cpu_path, power_path)


#allfiles_prepare()
#test = pd.read_csv("./real_data/10-27/net.csv" , delimiter=';')

def netfiles_prepare(): ## to be combined later with allfiles_prepare
    data_path = "./real_data/epouta/"
    dir_names = os.listdir(data_path)
    for i in range(len(dir_names)):
        print(" In directory : ", dir_names[i])
        file_path = data_path + dir_names[i]
        net_path = file_path +"/net.csv"

        net_data = pd.read_csv(net_path ,delimiter=';') #, dtype={2:'float64'}
        print(len(net_data))
        print(list(net_data))
        #print(test.iloc[12587])
        value_df = pd.DataFrame(net_data['Value'])
        print(value_df.iloc[12829])
        def r(x):
            x = str(x)
            x = x.replace(',', '')
            x= float(x)
            #print(x)
            return x

        value_df = value_df.applymap(lambda x: r(x))
        print(value_df.iloc[12829])

        net_data = pd.concat([net_data , value_df] ,axis=1)
        net_data.columns = ['Series' , 'Time' , 'OldValue' , 'Value']
        net_data.drop(['OldValue'] , axis=1 , inplace=True)

        net_series_df = pd.DataFrame(net_data['Series'])

        net_series_df = net_series_df.applymap(lambda x: x.split('.')[0])

        net_data = pd.concat([net_series_df, net_data], axis=1)

        net_data.columns = ['Series', 'Full_Series', 'Time', 'Value']
        net_data.drop(['Full_Series'], axis=1, inplace=True)

        print(list(net_data))
        #print(net_data.head)

        for series, df_series in net_data.groupby('Series'):
            # print(series)
            # print(df_series)
            agg_df = df_series.groupby(by='Time')['Value'].sum()
            # print(len(agg_df))

            with open('./real_data_prepared/epouta/net/' + series + '.csv', 'a') as f:
                 agg_df.to_csv(f, header=False)


#netfiles_prepare()


def window_data(file_path):
    #data = pd.read_csv("./real_data_prepared/epouta/e101_epouta_csc_fi.csv", header=None)
    data = pd.read_csv(file_path, header=None)
    data.columns = ['Time', 'Cpu_t', 'Power']
    data.drop(['Power' , 'Time'] , axis=1 , inplace=True)


    data['Cpu_t+1'] = data['Cpu_t'].shift(-1)
    data['Cpu_t+2'] = data['Cpu_t+1'].shift(-1)
    data['Cpu_t+3'] = data['Cpu_t+2'].shift(-1)
    #data['Cpu_t+4'] = data['Cpu_t+3'].shift(-1)
    data.dropna(inplace=True)
    #print(data)
    return data

def window_data_general(file_path , lock_back):
    #data = pd.read_csv("./real_data_prepared/epouta/e101_epouta_csc_fi.csv", header=None)
    data = pd.read_csv(file_path, header=None)
    data.columns = ['Time', 'Cpu_t+0', 'Power']
    data.drop(['Power' , 'Time'] , axis=1 , inplace=True)

    for i in range(lock_back):
         data['Cpu_t+'+ str(i+1)] = data['Cpu_t+'+ str(i)].shift(-1)

    data.dropna(inplace=True)
    #print(data)
    return data


#window_data_general("./real_data_prepared/epouta/e101_epouta_csc_fi.csv" , 3 )

def time_cpu_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data.columns = ['Time', 'Cpu_t', 'Power']
    data.drop(['Power'], axis=1, inplace=True)
    print(type(data['Time']))
    #print(data['Time'])
    df = data['Time']
    #data['new Time'] = pd.to_datetime((data['Time']))
    data['new Time'] = df.apply(pd.to_datetime , errors='coerce', utc=True ) #, errors='coerce'

    print(data.iloc[56])
    print(data.iloc[57])

time_cpu_data("./real_data_prepared/epouta/e101_epouta_csc_fi.csv")






