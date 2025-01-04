import sys
from math import floor
from venv import create
import numpy as np
import copy
import csv
import json

def normal_access_patterns(access):
    print(len(access))
    total_access_patterns =0
    zero = [0] * len(access)
    for i in range(-1,len(access)):
        temp_access = []
        # print(access[:i+1] + zero[:len(access) - i-1])
        # print(temp_access)
        total_access_patterns+=1

    print("Created ",total_access_patterns," total access patterns")

def by_two_access_pattern(access, acs_bound):
    print(len(access)+1)
    total_access_patterns =0
    zero = [0]*len(access)
    created_access_patterns = []
    for i in range(0,len(access),2):
        temp_list = (access[:i+1] + zero[:len(access) - i-1])
        created_access_patterns.append(temp_list)
        created_access_patterns =(np.unique(created_access_patterns,axis=0)).tolist()
        total_access_patterns=len(created_access_patterns)
    
    divisor =2
    new_access_pattern = access
    while total_access_patterns <len(access)+1:
        for i in range(len(created_access_patterns)-1,0,-1):
            temp_list = copy.deepcopy(created_access_patterns[i])
            for j in range(0,len(temp_list),1):
                if temp_list[j] > acs_bound:
                    temp_list[j] = floor(temp_list[j]/divisor)
            created_access_patterns.append(temp_list)
            created_access_patterns =(np.unique(created_access_patterns,axis=0)).tolist()
            total_access_patterns=len(created_access_patterns)
            if total_access_patterns == len(access)+1:
                break
        divisor = divisor*2

    print("Created ",total_access_patterns," total access patterns")
    return created_access_patterns


def by_four_access_pattern(access):
    print(len(access))
    total_access_patterns =0
    for i in range(0,len(access),4):
        temp_access = []
        temp_access2 = []
        temp_access3 = []
        temp_access4 = []
        for j in range(len(access)):
            if j<=i :
                temp_access.append(access[j])
                if access[j]>64:
                    temp_access2.append(floor(access[j]/2))
                    temp_access3.append(floor(access[j] / 4))
                    temp_access4.append(floor(access[j] / 8))
                else:
                    temp_access2.append(access[j])
                    temp_access3.append(access[j])
                    temp_access4.append(access[j])
            else:
                temp_access.append(0)
                temp_access2.append(0)
                temp_access3.append(0)
                temp_access4.append(0)
        print(temp_access)
        print(temp_access2)
        print(temp_access3)
        print(temp_access4)
        total_access_patterns+=4


    print("Created ",total_access_patterns," total access patterns")

def is_list_dominated(start_list, comparison_list):
    if len(start_list)!=len(comparison_list):
        print("Length Mismatch")
        exit(1)
    for i in range(len(start_list)):
        if start_list[i]>comparison_list[i]:
            return False

    return True

def max_access_pattern(access_pattern_list):
    max_sum = 0
    for list in access_pattern_list:
        if sum(list) > max_sum:
            max_sum = sum(list)
    return max_sum


def report(access_per_level, dataset_name, dim, total_access, rounds):
    if dim ==2:
        acs_bound = 16
    elif dim ==3:
        acs_bound = 64
    else: 
        print("Invalid Dimension")
        exit(1)
        
    normal_access_patterns(access_per_level)
    created_accesses = by_two_access_pattern(access_per_level, acs_bound)
    # by_four_access_pattern(access_per_level)

    query_access_list = []
    with open(f'reports/qr_acs_{dim}d_{dataset_name}.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            access_pattern = eval(row[1])
            temp_list = []
            for i in range(len(access_pattern)):
                temp_list.append(access_pattern[i])
            query_access_list.append(temp_list)

    height = []
    access = []

    max_height = len(created_accesses[0])
    max_accesses = max_access_pattern(created_accesses)
    print(max_height)
    print(max_accesses)
    for i in range(len(query_access_list)):
        query = query_access_list[i]
        current_min_accesses=1000000
        desired_j =1000000
        for j in range(len(created_accesses)):
            if is_list_dominated(query,created_accesses[j]):
                temp_sum = sum(created_accesses[j])
                if temp_sum< current_min_accesses:
                    current_min_accesses = temp_sum
                    desired_j = j
        if desired_j ==1000000:
            height.append(max_height)
            access.append(max_accesses)
            # print("Invalid Query")
        else:
            height.append(np.count_nonzero(created_accesses[desired_j]))
            access.append(current_min_accesses)

    print("Modified Height",sum(height)/len(height))
    Round_new = sum(height)/len(height)+1
    print("Rounds", sum(height)/len(height)+1)
    # print("Modified Accesses", sum(access)/len(access))
    print('delta', Round_new-rounds)
    print("Modified Accesses", sum(access)/len(access))
    print('Ratio of Modified Accesses to Full Access Pattern', (sum(access)/len(access))/total_access)


if __name__ == '__main__':
    # max access levels for each dataset from the config file
    max_access_levels =  [[1, 2, 4, 6, 9, 15, 21, 25, 28, 33, 37, 38, 41, 43, 38, 37, 35, 30, 24, 16, 10, 4], 
    [1, 2, 4, 6, 10, 18, 24, 28, 33, 39, 41, 39, 42, 39, 35, 30, 25, 20, 12, 7, 4, 2],
        [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2],
        [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2],
        [1, 2, 4, 6, 9, 13, 17, 25, 36, 50, 62, 77, 88, 97, 96, 93, 90, 69, 43, 20, 8] ,
        [1,2,4,6,9,16,19,21,21,22,24,25,24,23,20,16,15,11,12,10,8,7,7,4,1,0,0,0],
        [1,2,4,6,9,14,16,18,18,18,19,18,16,18,14,13,12,9,6,3,2,2,1,0,0]  ,
            [1,2,4,6,10,16,24,40,62,84,92,85,75,77,66,55,37,28,18,9,4,2,1] ,
            [1,2,4,6,8,11,13,16,22,26,27,31,32,36,36,39,31,30,28,25,19,14,7,0] ,
            [1,2,4,6,9,16,19,25,29,32,38,42,45,46,46,46,40,36,29,25,20,13,7,3] ,
            [1,2,4,6,9,16,20,27,33,36,36,39,42,41,38,34,32,28,24,20,12,4],
                [1, 2, 4, 6, 9, 14, 21, 32, 50, 68, 94, 120, 144, 166, 184, 181, 168, 178, 164, 138, 97, 51, 20, 6],
                [1,2,4,6,9,15,21,32,50,72,101,125,150,173,192,207,222,220,200,181,147,103,57,24,11,4,2]]

    # fill the dataset_names list with the names of the datasets above
    dataset_names = ['cali-1024x1024', 'spitz-1024x1024', 'gowalla', 'amazon-books', 'nh_64', 'gowalla_100k', 'gowalla_50k',
        'gowalla_3d_23k', 'synthetic_2d_1m', 'synthetic_2d_1m_sparse', 'synthetic_2d_1m-1024x1024', 'synthetic_3d_1m_128', 'synthetic_3d_1m_256']
    dims = [2,2,1,1,3,2,2,3,2,2,2,3,3]
    total_access =[398,306, 86,48,906, 239, 160,226,410,569,494,1644,2168 ]
    rounds = [18.0, 14.7, 24.8, 14.6, 13.8, 16.8, 14.4, 9.8, 21.7, 23.7, 22.2, 21.7, 24.8]

    for i in range(len(max_access_levels)):
        if (dims[i]>1):
            print("Dataset: ", dataset_names[i])
            report(max_access_levels[i], dataset_names[i], dims[i], total_access[i], rounds[i])


