# make this code a module in utils.py

import csv
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import math
# seed everything random in utils.py
random.seed(42)

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

# def count_range_nodes(points, *ranges):
#     """
#     Counts the number of points that fall within the specified ranges in each dimension.

#     :param points: List of tuples representing the points in d-dimensional space.
#     :param ranges: Variable number of tuples, each representing the lower and upper bounds for a dimension.
#     :return: The count of points that fall within the specified ranges.
#     """
#     counter = 0

#     for pt in points:
#         in_range = True
#         for dim in range(len(ranges)):
#             if not (ranges[dim][0] <= pt[dim] <= ranges[dim][1]):
#                 in_range = False
#                 break
#         if in_range:
#             counter += 1

#     return counter

def count_range_nodes(points, *ranges):
    """
    Counts the number of points that fall within the specified ranges in each dimension.

    :param points: List of points, where each point is a tuple representing coordinates in d-dimensional space,
                   or an integer for 1-dimensional space.
    :param ranges: A variable number of tuples, each representing the lower and upper bounds for a dimension.
    :return: The count of points that fall within the specified ranges.
    """
    counter = 0

    for pt in points:
        if isinstance(pt, int):
            # 1-dimensional case
            if ranges[0][0] <= pt <= ranges[0][1]:
                counter += 1
        else:
            # Multi-dimensional case
            in_range = True
            for dim in range(len(ranges)):
                if not (ranges[dim][0] <= pt[dim] <= ranges[dim][1]):
                    in_range = False
                    break
            if in_range:
                counter += 1

    return counter
def random_range(range):
    l_bound = random.randint(range[0], range[1])
    u_bound = random.randint(l_bound, range[1])
    return (l_bound, u_bound)   

# a function to generate a random query
def generate_random_query(num_dims, x_range, y_range = None, z_range=None):
    """
    Generates a random query within the specified ranges for each dimension.

    :param num_dims: The number of dimensions in the space.
    :param x_range: A tuple representing the lower and upper bounds for the x-dimension.
    :param y_range: A tuple representing the lower and upper bounds for the y-dimension.
    :param z_range: A tuple representing the lower and upper bounds for the z-dimension.
    :return: A tuple representing the query ranges for each dimension.
    """
    if num_dims == 1:
        return random_range(x_range)
    elif num_dims == 2:
        return (random_range(x_range), random_range(y_range))
    elif num_dims == 3:
        return (random_range(x_range), random_range(y_range), random_range(z_range))
    else:
        raise ValueError('Only 1D, 2D, and 3D queries are supported.')

# define a function based on the above logic when the condition change based on the input parameter devition,
#  for example the above is 4 devisons, if the input is 2, then it will be 2 divisions (lower than mean, greater thean mean),
#  if the input is 8, then it will be 8 divisions 
# (lower than mean-3*std, between mean-3*std and mean-2*std, between mean-2*std and mean-std,
#  between mean-std and mean, between mean and mean+std, between mean+std and mean+2*std,
#  between mean+2*std and mean+3*std, greater than mean+3*std)
def replace_values(qr_h_1d, mean, std, max, divisions):
    qr_h_1d_ = [[0 for _ in range(len(qr_h_1d[i]))] for i in range(len(qr_h_1d))]
    for i in range(len(qr_h_1d)):
        for j in range(len(qr_h_1d[i])):
            if divisions == 1:
                # replace all values with max
                qr_h_1d_[i][j] = max
            if divisions == 2:
                if qr_h_1d[i][j] <= mean:
                    qr_h_1d_[i][j] = mean
                else:
                    qr_h_1d_[i][j] = max
            elif divisions == 4:
                if qr_h_1d[i][j] <= mean - std:
                    qr_h_1d_[i][j] = mean - std
                elif qr_h_1d[i][j] <= mean:
                    qr_h_1d_[i][j] = mean
                elif qr_h_1d[i][j] <= mean + std:
                    qr_h_1d_[i][j] = mean + std
                else:
                    qr_h_1d_[i][j] = max
            elif divisions == 8:
                if qr_h_1d[i][j] <= mean - 3*std:
                    qr_h_1d_[i][j] = mean - 3*std
                elif qr_h_1d[i][j] <= mean - 2*std:
                    qr_h_1d_[i][j] = mean - 2*std
                elif qr_h_1d[i][j] <= mean - std:
                    qr_h_1d_[i][j] = mean - std
                elif qr_h_1d[i][j] <= mean:
                    qr_h_1d_[i][j] = mean
                elif qr_h_1d[i][j] <= mean + std:
                    qr_h_1d_[i][j] = mean + std
                elif qr_h_1d[i][j] <= mean + 2*std:
                    qr_h_1d_[i][j] = mean + 2*std
                elif qr_h_1d[i][j] <= mean + 3*std:
                    qr_h_1d_[i][j] = mean + 3*std
                else:
                    qr_h_1d_[i][j] = max
    return qr_h_1d_

# overwire the above function when the qr_h_ input is flattened and also parametrize the loop for mean-x*std
def replace_values_2(qr_h, mean, std, max, divisions):
    qr_h_ = [0 for i in range(len(qr_h))]
    for i in range(len(qr_h)):
        if divisions == 1:
            # replace all values with max
            qr_h_[i] = max
        if divisions == 2:
            if qr_h[i] <= mean:
                qr_h_[i] = mean
            else:
                qr_h_[i] = max
        if divisions == 4:
            if qr_h[i] <= mean - std:
                qr_h_[i] = mean - std
            elif qr_h[i] <= mean:
                qr_h_[i] = mean
            elif qr_h[i] <= mean + std:
                qr_h_[i] = mean + std
            else:
                qr_h_[i] = max
        if divisions == 8:
            if qr_h[i] <= mean - 3*std:
                qr_h_[i] = mean - 3*std
            elif qr_h[i] <= mean - 2*std:
                qr_h_[i] = mean - 2*std
            elif qr_h[i] <= mean - std:
                qr_h_[i] = mean - std
            elif qr_h[i] <= mean:
                qr_h_[i] = mean
            elif qr_h[i] <= mean + std:
                qr_h_[i] = mean + std
            elif qr_h[i] <= mean + 2*std:
                qr_h_[i] = mean + 2*std
            elif qr_h[i] <= mean + 3*std:
                qr_h_[i] = mean + 3*std
            else:
                qr_h_[i] = max
    return qr_h_    

def report_h_dev(qr_h, mean, std, max, dev, dim):
    qr_h_ = replace_values(qr_h, mean, std, max, dev)
    # qr_h_flat = [item for sublist in qr_h_ for item in sublist]
    qr_h_avg = [np.round(np.mean(np.array(qr_h_[i]))) for i in range(len(qr_h_))]
    print(f'average of the qr_h_{dim}d_{dev}:', np.round(np.mean(qr_h_avg), decimals=2))
    print(f'std of the qr_h_{dim}d_{dev}:', np.round(np.std(qr_h_avg), decimals=2))
    plt.hist(qr_h_avg, bins=dev, alpha=0.5, label='1D')
    plt.savefig(f'plots/qr_h_{dim}d_avg_{dev}.png')
    plt.clf()

def plot_histogram(qr_h_avg,bins, dim, name = None, label = 'avg'):
    plt.hist(qr_h_avg, bins=bins, alpha=0.5, label=f'{dim}D {label}')
    plt.legend(loc='upper right')
    plt.title(f'Histogram of {dim}D heights')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'plots/{name}_{dim}d_.png')
    plt.clf()  # Clear the current figure for the next plot

def load_csv_data(file_path):
    # loading the qr_h_{dim}d.csv file like the qr_h_{dim}d.csv file
    qr_h = []
    with open(f'reports/{file_path}') as file:
        reader = csv.reader(file)
        for row in reader:
            qr_h.append(row[1])

    # create a new list to store the qr_h values
    qr_h_values = []
    for qr in qr_h:
        qr = qr.replace('{', '').replace('}', '').split(',')
        if qr[0] == '':
            continue
            # append empty value to the qr_h_values
        qr = [int(q.split(':')[1]) for q in qr]
        qr_h_values.append(qr)

    return qr_h_values

import ast

def load_csv_data_acs_dims(file_path):

    qr_acs_dims = []
    with open(f'reports/{file_path}') as file:
        reader = csv.reader(file)
        for row in reader:
            qr_acs_dims.append(row[1])
    
    # create a new list to store the qr_acs_dims values
    qr_acs_dims_values = []
    for qr in qr_acs_dims:
        # Convert the dictionary string to a dictionary
        data_dict = ast.literal_eval(qr)
        # Extract the values and convert them to a list of tuples
        list_of_tuples = list(data_dict.values())
        qr_acs_dims_values.append(list_of_tuples)

    return qr_acs_dims_values


def barplot(data, dim, name, label='avg'):
    x = np.arange(len(data))
    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(x, data, bar_width, alpha=opacity, color='b', label= label)

    plt.xlabel('Level')
    plt.ylabel('Number of accessed nodes')
    # plt.title(f'Average and maximum of the number of accessed nodes in each level of the tree in {dim}D')
    plt.xticks(x, [str(i) for i in range(len(data))])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{name}_{dim}d.png')
    plt.clf()

def write_csv(data, name):
    with open(f'reports/{name}.csv', 'w') as file:
        writer = csv.writer(file)
        for i in range(len(data)):
            writer.writerow([i, data[i]])

# load the 2d dataset from the json/pickle file
def load_dataset(path, type_ = 'csv'):
    if type_ == 'csv' or type_ == 'txt':
        with open(path) as f:
            data = json.load(f)

    elif type_ == 'pkl' or type_ == 'pickle':
        with open(path, 'rb') as f:
            data = pickle.load(f)

    # check if pts is a dictionary
    if isinstance(data, dict):
        pts = list(data.keys())
    else:
        pts = data

    num_dims = len(pts[0])
    pts = list(set(map(tuple, pts))) # Remove duplicates
    # plot the points
    # plt.scatter([pt[0] for pt in pts], [pt[1] for pt in pts])

    # calcuate unique x and unique y values
    unique_x = set([pt[0] for pt in pts])
    unique_y = set([pt[1] for pt in pts])

    # print the unique x and y values
    print('unique x values:', len(unique_x))
    print('unique y values:', len(unique_y))

    print('size of the dataset:', len(pts))
    # sorting 2d points
    pts = sorted(pts, key=lambda x: (x[0], x[1]))

    # print the x bound and y bound of the dataset
    print('x bound:', (pts[0][0], pts[-1][0]))
    x_range = (pts[0][0], pts[-1][0])
    # sort the points by y values
    pts = sorted(pts, key=lambda x: (x[1], x[0]))

    y_range = (pts[0][1], pts[-1][1])
    print('y bound:', (pts[0][1], pts[-1][1]))

    pts = sorted(pts, key=lambda x: (x[0], x[1])) 

    # check if pts is a dictionary
    if isinstance(data, dict):
    # create pts_dict from the pts, and associate each point from the assoicated value in data dictionary
        pts_dict = {pt: data[pt] for pt in pts}

    else:
    # crete pts_dict from the pts, and associate each point with 1
        pts_dict = {pt: 1 for pt in pts}
    
    return pts, pts_dict, x_range, y_range


def load_dataset_3d(path, type_ = 'csv'):
    
    if type_ == 'csv' or type_ == 'txt':
        with open(path) as f:
            data = json.load(f)

    elif type_ == 'pkl' or type_ == 'pickle':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    
    # check if pts is a dictionary
    if isinstance(data, dict):
        pts = list(data.keys())
    else:
        pts = data
    
    num_dims = len(pts[0])
    pts = list(set(map(tuple, pts))) # Remove duplicates

    print('size of the dataset:', len(pts))

    # sorting 3d points
    pts = sorted(pts, key=lambda x: (x[0], x[1], x[2]))

    # calculating the unique x, y, z values
    unique_x = sorted(list(set([pt[0] for pt in pts])))
    unique_y = sorted(list(set([pt[1] for pt in pts])))
    unique_z = sorted(list(set([pt[2] for pt in pts])))

    x_range = (unique_x[0], unique_x[-1])
    y_range = (unique_y[0], unique_y[-1])
    z_range = (unique_z[0], unique_z[-1])


    # print the length of the unique x, y, z values
    print('unique x values:', len(unique_x))
    print('unique y values:', len(unique_y))
    print('unique z values:', len(unique_z))

    # print the range of the x, y, z values
    print('-'*50 + 'domain range values' + '-'*50)
    print('x range:', x_range)
    print('y range:', y_range)
    print('z range:', z_range)

    # check if pts is a dictionary
    if isinstance(data, dict):
    # create pts_dict from the pts, and associate each point from the assoicated value in data dictionary
        pts_dict = {pt: data[pt] for pt in pts}

    else:
    # crete pts_dict from the pts, and associate each point with 1
        pts_dict = {pt: 1 for pt in pts}
    
    return pts, pts_dict, x_range, y_range, z_range

# write a theoritical max calculation function for access per level, n is the list of n values in each dimension
def theorical_max(n):
    d = len(n)

    if d == 1:
        return 4
    
    max_ = 1
    for i in range(d):
        p = 1
        for j in range(i):
            p *= (math.ceil((math.log2(n[j])-1)))
        if i>0 :
            max_ += (math.pow(2,i))* p

    max_ *= 4
    return max_

def generate_max_access_list(dataset_name):
    # Define the dataset array
    # Updated dataset heights
    dataset_heights = {
        # 2D datasets
        'cali-1024x1024': 22,
        'spitz-1024x1024': 22,
        'gowalla_100k': 28,
        'gowalla_50k': 25,
        'synthetic_2d_1m': 24,
        'synthetic_2d_1m_sparse': 24,
        'synthetic_2d_1m-1024x1024': 22,
        
        # 3D datasets
        'nh_64': 21,
        'gowalla_3d_23k': 23,
        'synthetic_3d_1m_128': 24,
        'synthetic_3d_1m_256': 27
    }
    
    # Updated dataset info
    dataset_info = [
        # 2D datasets
        {"dataset": "cali-1024x1024", "theoretical_max": 76},
        {"dataset": "spitz-1024x1024", "theoretical_max": 76},
        {"dataset": "gowalla_100k", "theoretical_max": 100},
        {"dataset": "gowalla_50k", "theoretical_max": 92},
        {"dataset": "synthetic_2d_1m", "theoretical_max": 84},
        {"dataset": "synthetic_2d_1m_sparse", "theoretical_max": 84},
        {"dataset": "synthetic_2d_1m-1024x1024", "theoretical_max": 76},
        
        # 3D datasets
        {"dataset": "nh_64", "theoretical_max": 444},
        {"dataset": "gowalla_3d_23k", "theoretical_max": 524},
        {"dataset": "synthetic_3d_1m_128", "theoretical_max": 628},
        {"dataset": "synthetic_3d_1m_256", "theoretical_max": 844}
    ]

    # the max access list from the experiments of running 10000 queries for each dataset
    qr_acs_max = {
    # 2D datasets
    "cali-1024x1024": [1, 2, 4, 6, 10, 16, 17, 20, 23, 24, 28, 32, 33, 33, 32, 28, 20, 13, 10, 6, 3, 2],
    "spitz-1024x1024": [1, 2, 4, 6, 10, 17, 22, 24, 26, 27, 24, 24, 18, 15, 11, 8, 7, 5, 5, 3, 1, 1],
    "gowalla_100k": [1, 2, 4, 6, 10, 17, 20, 25, 24, 26, 28, 30, 27, 28, 28, 25, 20, 20, 17, 11, 12, 9, 9, 10, 6, 2, 1, 0],
    "gowalla_50k": [1, 2, 4, 6, 10, 16, 16, 18, 20, 21, 24, 23, 20, 20, 26, 26, 24, 18, 14, 9, 7, 5, 2, 1, 0],
    "synthetic_2d_1m": [1, 2, 4, 6, 8, 12, 16, 19, 22, 26, 31, 33, 37, 40, 41, 41, 42, 38, 33, 28, 22, 17, 9, 4],
    "synthetic_2d_1m_sparse": [1, 2, 4, 6, 10, 17, 24, 29, 33, 38, 40, 44, 46, 49, 49, 48, 44, 39, 33, 28, 22, 15, 9, 4],
    "synthetic_2d_1m-1024x1024": [1, 2, 4, 6, 10, 17, 22, 27, 33, 35, 38, 40, 44, 43, 43, 40, 36, 34, 28, 20, 12, 4],

    # 3D datasets
    "gowalla_3d_23k": [1, 2, 4, 6, 10, 16, 24, 31, 47, 62, 75, 82, 75, 68, 63, 54, 43, 37, 26, 14, 7, 2, 0],
    "synthetic_3d_1m_128": [1, 2, 4, 6, 10, 15, 24, 39, 59, 86, 118, 144, 159, 174, 188, 184, 204, 208, 188, 148, 99, 54, 26, 8],
    "synthetic_3d_1m_256": [1, 2, 4, 6, 10, 18, 28, 43, 67, 95, 134, 176, 215, 258, 291, 305, 317, 317, 307, 285, 234, 164, 79, 38, 16, 8, 3],
    "nh_64": [1, 2, 4, 6, 10, 16, 26, 40, 53, 58, 64, 74, 69, 58, 49, 35, 27, 19, 13, 6, 3]
        }

    
    # Find the height for the given dataset
    h = dataset_heights.get(dataset_name)
    
    # Find the theoretical max for the given dataset
    theoretical_max = None
    for data in dataset_info:
        if data["dataset"] == dataset_name:
            theoretical_max = data["theoretical_max"]
            break
    
    if theoretical_max is None:
        return "Dataset not found"
    
    # initialize a list with size h
    result = [0 for _ in range(h)]
    
    # First half: start with 1, multiply by 2 until close to max
    half_size = h // 2
    value = 1
    for i in range(half_size):
        result[i] = value
        # If doubling would exceed theoretical max, stop increasing
        if value * 2 < theoretical_max:
            value *= 2
        else:
            value = theoretical_max
    
    # Check if dataset is 3D (to set initial value for second half)
    is_3d_dataset = any(dataset_name == d["dataset"] for d in dataset_info[7:])  # Check if dataset is in 3D section
    
    # Second half: set initial value based on dataset dimension
    value = 16 if is_3d_dataset else 8
    
    for i in range(h - half_size):
        # fill the result list from the end
        result[h - i - 1] = value
        # If doubling would exceed theoretical max, stop increasing
        if value * 2 < theoretical_max:
            value *= 2
        else:
            value = theoretical_max
    
    # print(f"Generated max access list for {dataset_name}: {result}")
    # print the experimental max access list for the dataset
    # print(f"Experimental max access list for {dataset_name}: {qr_acs_max.get(dataset_name, 'Not found')}")
    
    # compare the generated max access list with the experimental data
    if dataset_name in qr_acs_max:
        # check if they have the same length
        if len(qr_acs_max[dataset_name]) != h:
            print(f"Warning: Length mismatch for dataset {dataset_name}. Expected {h}, got {len(qr_acs_max[dataset_name])}.")
            return result
        experimental_max = qr_acs_max[dataset_name]
        for i in range(h):
            if result[i] < experimental_max[i]:
                print(f"Warning: Generated max access list for {dataset_name} at level {i} is less than experimental data. Adjusting...")
    
    return result

        



    