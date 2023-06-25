'''
Author: wyd1216.git&&yudongwang117@icloud.com
Date: 2022-10-18 20:10:02
LastEditors: Yudong Wang yudongwang117@icloud.com
LastEditTime: 2022-10-18 20:13:30
FilePath: /image_processing/image_processing/table_info_check.py
Description: 

Copyright (c) 2022 by Yudong Wang yudongwang117@icloud.com, All Rights Reserved. 
'''
import pandas as pd
import numpy as np
import os


def generete_list_dict(column, all_df):
    column_list = []
    column_dict = {}

    for i, row in all_df.iterrows():
        column_list.append(row[column])

        if row[column] not in column_dict:
            column_dict[row[column]] = 1
        else:
            column_dict[row[column]] += 1
    return column_list, column_dict


def returnSum(myDict):

    sum = 0
    for i in myDict:
        sum = sum + myDict[i]

    return sum


def statistic(dict_cur):
    dict_cur_sort = sorted(dict_cur.items(), key=lambda e: e[0], reverse=False)
    statistic_list = []
    total_num = returnSum(dict_cur)
    proportion_sum = 0
    for i, each in enumerate(dict_cur_sort):
        each_list = list(each)
        proportion = each_list[1] / total_num * 100
        each_list.append(proportion)

        proportion_sum += proportion
        each_list.append(proportion_sum)
        statistic_list.append(each_list)
    print("total_num:{}".format(total_num))
    return statistic_list


def statistic_mean(dict_list):
    dict_array = np.array(sorted(dict_list))

    mean_value = np.mean(dict_array)
    min_value = np.min(dict_array)
    max_value = np.max(dict_array)
    median_value = np.median(dict_array)

    lower_q = np.quantile(dict_array, 0.25, interpolation='lower')  #下四分位数
    higher_q = np.quantile(dict_array, 0.75, interpolation='higher')  #上四分位数

    print("vmin".ljust(15), "{}".format(round(min_value, 2)).ljust(10))
    print("lower quartile".ljust(15), "{}".format(round(lower_q, 2)).ljust(10))
    print("mean".ljust(15), "{}".format(round(mean_value, 2)).ljust(10))
    print("median".ljust(15), "{}".format(round(median_value, 2)).ljust(10))
    print("higher quartile".ljust(15), "{}".format(round(higher_q,
                                                         2)).ljust(10))
    print("vmax".ljust(15), "{}".format(round(max_value, 2)).ljust(10))


def show_statistic(statistic_list):
    for i in statistic_list:
        print("{}".format(i[0]).ljust(25), "{}".format(i[1]).ljust(5),
              "{}".format(round(i[2], 2)).ljust(10),
              "{}".format(round(i[3], 2)).ljust(10))
