import pandas as pd

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
        # ins_df = ins_df.groupby(by='TimeStamp')['Temp'].sum()
        ins_df = ins_df.groupby(by='TimeStamp')['Temp'].mean()
        ins_df.to_csv("./real_data_prepared_Upp/Temp_mean/" + ins_name.split(':')[0] + ".csv" , header=True)


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


def prepare_disk_io(diskio_filepath):
    f = open(diskio_filepath, "r")
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
        ins_df.columns = ["TimeStamp", "Disk_io"]
        disk_df = ins_df["Disk_io"].astype(float)
        disk_df = pd.DataFrame(disk_df)
        disk_df = disk_df.applymap(lambda x: -1*x if x <0 else x) ## to convert all values to positive value
        ins_df["Disk_io"] = disk_df
        ins_df = ins_df.groupby(by='TimeStamp')["Disk_io"].sum()
        ins_df.to_csv("./real_data_prepared_Upp/" + "Disk_io" + "/" + ins_name.split(':')[0] + ".csv", header=True)


def prepare_disk_space(diskspace_filepath):
    f = open(diskspace_filepath, "r")
    space_data = json.load(f)

    print(len(space_data['data']['result']))

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
    for i in space_data['data']['result']:

        if i['metric']['dimension'] == 'avail':
            if i['metric']['instance'] in ins_dic:
                ins_dic[i['metric']['instance']].extend(i['values'])
            else:
                ins_dic[i['metric']['instance']] = i['values']

    print("Number of machine : ", len(ins_dic))

    for ins_name, val in ins_dic.items():
        ins_df = pd.DataFrame(val)
        ins_df.columns = ["TimeStamp", "Disk_space"]
        ins_df["Disk_space"] = ins_df["Disk_space"].astype(float)
        ins_df = ins_df.groupby(by='TimeStamp')["Disk_space"].sum()
        ins_df.to_csv("./real_data_prepared_Upp/" + "Disk_space" + "/" + ins_name.split(':')[0] + ".csv", header=True)


#prepare_cpu("./real_data/uppmax/cpu")

#prepare_temp("./real_data/uppmax/temperature")

# prepare_feature("./real_data/uppmax/disk_io" , "Disk_io")

#prepare_disk_io("./real_data/uppmax/disk_io")

#prepare_disk_space("./real_data/uppmax/disk_space")

# f = open("./real_data/uppmax/disk_space", "r")
# cpu_data = json.load(f)
# print(len(cpu_data['data']['result']))
#
# c= 0
# for i in cpu_data['data']['result']:
#     if i['metric']['instance'] == 'd13:19999': #"instance":"d102:19999"
#         # print(i['metric'])
#         print(i)
#         c +=1
#
# print(c)

##putting all data together
# ins_name = "d40"
# cpu_data = pd.read_csv("./real_data_prepared_Upp/CPU/" + ins_name + ".csv")
# mem_data = pd.read_csv("./real_data_prepared_Upp/Mem/" + ins_name + ".csv")
# disk_space_data = pd.read_csv("./real_data_prepared_Upp/Disk_space/" + ins_name + ".csv")
# disk_io_data = pd.read_csv("./real_data_prepared_Upp/Disk_io/" + ins_name + ".csv")
# temp_data = pd.read_csv("./real_data_prepared_Upp/Temp_mean/" + ins_name + ".csv")
#
# data = pd.concat([cpu_data , mem_data , disk_space_data , disk_io_data , temp_data ] , axis=1)
# data.columns = ["Time" , "CPU" , "Time2" , "Mem" , "Time3" ,"Disk_Space" , "Time4" , "Disk_IO" , "Time5" , "Temp"]
# data.drop(["Time" ,"Time2" , "Time3" , "Time4" , "Time5"] , axis=1 , inplace= True)
# data.to_csv("data_docker.csv", index=False)
