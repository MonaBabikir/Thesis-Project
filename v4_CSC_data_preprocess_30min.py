import os
import pandas as pd

def onefile_prepare_30min(file_path , file_name):
    # data = pd.read_csv("./real_data_prepared/epouta/e23_epouta_csc_fi.csv")
    data = pd.read_csv(file_path)
    data = pd.DataFrame(data)
    data.columns = ["Time" , "CPU" , "Power"]

    newtime_df = pd.to_datetime(data["Time"])
    newtime_df = pd.DataFrame(newtime_df)
    newtime_df = newtime_df.applymap(lambda x: x if x.minute == 00 or x.minute == 30 else 0)
    newtime_df.columns= ["Time"]

    data["Time"] = data["Time"].astype(str)
    newtime_df["Time"] = newtime_df["Time"].astype(str)

    fdata = pd.merge(newtime_df, data , on="Time" , how='inner')

    # print(fdata)
    # print(len(fdata))

    fdata.to_csv("./real_data_prepared/epouta_30min/" + file_name ,  header=False , index=False)

#onefile_prepare_30min()


def allfiles_prepare():
    print("START PROCESSING ALL FILES ...")
    data_path = "./real_data_prepared/epouta/"
    file_names = os.listdir(data_path)
    for i in range(len(file_names)):
        print(" In File : ", file_names[i])
        file_path = data_path +  file_names[i]

        onefile_prepare_30min(file_path , file_names[i])

    print("END PROCESSING ALL FILES ...")


#allfiles_prepare()

def window_data_general_30min(file_path, lock_back):
    data = pd.read_csv(file_path, header=None)
    n1 = 386  # (1 - 772)
    n2 = 1538  # (772 - 3076)

    data1 = pd.DataFrame(data[:n1])
    data2 = pd.DataFrame(data[n1: n2])
    data3 = pd.DataFrame(data[n2:])

    # print(data1)
    # print("***********")
    # print(data2)
    # print("***********")
    # print(data3)
    # print("***********")

    data.columns = ['Time', 'Cpu_t+0', 'Power']
    data.drop(['Power', 'Time'], axis=1, inplace=True)

    data1.columns = ['Time', 'Cpu_t+0', 'Power']
    data1.drop(['Power', 'Time'], axis=1, inplace=True)
    data2.columns = ['Time', 'Cpu_t+0', 'Power']
    data2.drop(['Power', 'Time'], axis=1, inplace=True)
    data3.columns = ['Time', 'Cpu_t+0', 'Power']
    data3.drop(['Power', 'Time'], axis=1, inplace=True)

    for i in range(lock_back):
        data['Cpu_t+' + str(i + 1)] = data['Cpu_t+' + str(i)].shift(-1)

        data1['Cpu_t+'+ str(i+1)] = data1['Cpu_t+'+ str(i)].shift(-1)
        data2['Cpu_t+' + str(i + 1)] = data2['Cpu_t+' + str(i)].shift(-1)
        data3['Cpu_t+' + str(i + 1)] = data3['Cpu_t+' + str(i)].shift(-1)


    data1.dropna(inplace=True)
    data2.dropna(inplace=True)
    data3.dropna(inplace=True)

    return data1, data2 , data3 , data


window_data_general_30min("./real_data_prepared/epouta/e23_epouta_csc_fi.csv", 3)