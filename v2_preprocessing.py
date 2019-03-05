"https://github.com/kwananth/VMWorkloadPredictor/blob/master/massage_uniform_20_classes.py"

import glob
import csv
import os
import ntpath

path = "./fastStorage/*.csv"
#path = "G://Master_Uppsala//semester_4//Code//fastStorage//2013-8/*.csv"
output_path = "./output/"

sample_size_average = 30 # 30 entries = 9000 ms = 9 second average

for fname in glob.glob(path):
    head, tail = ntpath.split(fname)
    #print(tail)

    with open(fname, 'r') as infh:
        next(infh)
        reader = csv.reader(infh, delimiter=';')

        #output = open(output_path + fname[fname[2:].find('/')+3:], "w+")
        output = open(output_path + tail, "w+")

        timestamp_list = []
        cpu_usage_list = []
        mem_usage_percent_list = []
        disk_read_list = []
        disk_write_list = []
        net_in_list = []
        net_out_list = []

        counter = 0

        for row in reader:

            timestamp = int(row[0])
            cpu_usage = float(row[4])
            mem_capacity = float(row[5])
            mem_usage = float(row[6])
            disk_read = float(row[7])
            disk_write = float(row[8])
            net_in = float(row[9])
            net_out = float(row[10])

            # Append to lists
            timestamp_list.append(timestamp)
            cpu_usage_list.append(cpu_usage)

            if (mem_capacity != 0):
                mem_usage_percent_list.append((mem_usage/mem_capacity)*100.0)
            else:
                mem_usage_percent_list.append(0.0)

            disk_read_list.append(disk_read)
            disk_write_list.append(disk_write)
            net_in_list.append(net_in)
            net_out_list.append(net_out)

            counter+= 1

            if counter >= sample_size_average:
                # Get the averages
                timestamp_avg = sum(timestamp_list)/len(timestamp_list)
                cpu_avg = sum(cpu_usage_list)/len(cpu_usage_list)
                mem_avg = sum(mem_usage_percent_list)/len(mem_usage_percent_list)
                disk_read_avg = sum(disk_read_list)/len(disk_read_list)
                disk_write_avg = sum(disk_write_list)/len(disk_write_list)
                net_in_avg = sum(net_in_list)/len(net_in_list)
                net_out_avg = sum(net_out_list)/len(net_out_list)

                class_num = -1

                if cpu_avg < 5.0:
                    class_num = 0
                elif cpu_avg < 10.0:
                    class_num = 1
                elif cpu_avg < 15.0:
                    class_num = 2
                elif cpu_avg < 20.0:
                    class_num = 3
                elif cpu_avg < 25.0:
                    class_num = 4
                elif cpu_avg < 30.0:
                    class_num = 5
                elif cpu_avg < 35.0:
                    class_num = 6
                elif cpu_avg < 40.0:
                    class_num = 7
                elif cpu_avg < 45.0:
                    class_num = 8
                elif cpu_avg < 50.0:
                    class_num = 9
                elif cpu_avg < 55.0:
                    class_num = 10
                elif cpu_avg < 60.0:
                    class_num = 11
                elif cpu_avg < 65.0:
                    class_num = 12
                elif cpu_avg < 70.0:
                    class_num = 13
                elif cpu_avg < 75.0:
                    class_num = 14
                elif cpu_avg < 80.0:
                    class_num = 15
                elif cpu_avg < 85.0:
                    class_num = 16
                elif cpu_avg < 90.0:
                    class_num = 17
                elif cpu_avg < 95.0:
                    class_num = 18
                else:
                    class_num = 19

                counter = 0
                timestamp_list = []
                cpu_usage_list = []
                mem_usage_percent_list = []
                disk_read_list = []
                disk_write_list = []
                net_in_list = []
                net_out_list = []

                output.write(str(timestamp_avg) + ';' + str(cpu_avg) + ';' + str(mem_avg) + ';' + str(disk_read_avg) + ';' + str(disk_write_avg) + ';' + str(net_in_avg) + ';' + str(net_out_avg) + ';' + str(class_num) + '\n')



###################### Preprocessing according to the online source "preprocess.py"
import glob
import csv
import os

path = "./output/*.csv"
output_path = "./ml_data/"

features_multiplier = 10 # 10 * 7 = 70 features

for fname in glob.glob(path):
    with open(fname, 'r') as infh:
        reader = csv.reader(infh, delimiter=';')

        head, tail = ntpath.split(fname)
        output = open(output_path + tail, "w+")

        timestamp_list = []
        cpu_usage_list = []
        mem_usage_percent_list = []
        disk_read_list = []
        disk_write_list = []
        net_in_list = []
        net_out_list = []

        counter = 0

        csv_file_list = []

        for row in reader:
            timestamp_avg = float(row[0])
            cpu_avg = float(row[1])
            mem_avg = float(row[2])
            disk_read_avg = float(row[3])
            disk_write_avg = float(row[4])
            net_in_avg = float(row[5])
            net_out_avg = float(row[6])
            class_num = float(row[7])

            '''
            csv_file_list.append(
                str(timestamp_avg) + ';' + str(cpu_avg) + ';' + str(mem_avg) + ';' + str(disk_read_avg) + ';' + str(
                    disk_write_avg) + ';' + str(net_in_avg) + ';' + str(net_out_avg))
            '''

            entry_list = [timestamp_avg, cpu_avg, mem_avg, disk_read_avg, disk_write_avg, net_in_avg, net_out_avg, class_num]
            csv_file_list.append(entry_list)

        print(len(csv_file_list))

        i = 0
        while i < len(csv_file_list) - features_multiplier:
            upperbound = i + features_multiplier

            j = i
            while j < upperbound:
                output.write(str(csv_file_list[j][0]) + ';' + str(csv_file_list[j][1]) + ';' + str(csv_file_list[j][2]) + ';' + str(csv_file_list[j][3]) + ';' + str(
                    csv_file_list[j][4]) + ';' + str(csv_file_list[j][5]) + ';' + str(csv_file_list[j][6]) + ';')
                j += 1
            #output.write(str(csv_file_list[j][7]) + '\n')
            output.write(str(csv_file_list[j][1]) + '\n')

            i += 1