import sys
import os
import argparse

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from termcolor import colored
from utils import report_h_dev, plot_histogram, load_csv_data, next_power_of_2, barplot, load_csv_data_acs_dims
from config import dataset_config



def height_report_query(file_path, dim):
    
    #print the name of the file being loaded
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    print(colored(f'loading {file_name}...', 'blue'))

    # loading the qr_h_3d.csv 
    qr_h_values = load_csv_data(file_path= file_path)
    qr_h = qr_h_values

    qr_h_avg = [np.mean(np.array(qr_h_values[i])) for i in range(len(qr_h_values))]
    qr_h_max = [np.max(np.array(qr_h_values[i])) for i in range(len(qr_h_values))]
    # reading all the values in the qr_h_values list and store them in a new list
    qr_h_values = [val for qr in qr_h_values for val in qr]

    # convert the qr_h_values to a numpy array
    qr_h_values = np.array(qr_h_values)

    mean = np.round(np.mean(qr_h_values), decimals=2)
    std = np.round(np.std(qr_h_avg), decimals=2)
    max = np.max(qr_h_max)
    l =[1,2,4,8]
    for i in range(len(l)):
        print(f'calling report_h_dev for {dim}D with dev=', l[i])
        report_h_dev(qr_h, mean, std, max, l[i], dim)
        print('\n')

    # print the average of the qr_h_3d_values and round it to 2 decimal places
    print(f'average of the qr_h_{dim}d_values:', np.round(np.mean(qr_h_values), decimals=2))
    # print the std of the qr_h_3d_values and round it to 2 decimal places
    print(f'std of the qr_h_{dim}d_values:', np.round(np.std(qr_h_values), decimals=2))
    # plot_histogram(qr_h_avg, max, dim, name=f'qr_h_avg', label='avg')


def nodes_avg_pow2(data, dim):
    data_= [0 for i in range(len(data))]
    for i in range(len(data)):
        data_[i] = next_power_of_2(len(data[i]))
    # print the average of the data_
    print(f'average of the nodes in {dim}D:', np.round(np.mean(data_), decimals=2))
    # print the std of the data_
    print(f'std of the nodes in {dim}D:', np.round(np.std(data_), decimals=2))             

# write a function to compute the average of the nodes in 1D, 2D, and 3D
def nodes_avg(data, dim):
    # print the average of the data
    print(f'average of the nodes in {dim}D:', np.round(np.mean([len(data[i]) for i in range(len(data))]), decimals=2))
    # print the std of the data
    print(f'std of the nodes in {dim}D:', np.round(np.std([len(data[i]) for i in range(len(data))]), decimals=2))

# a function that calculates the average and maximum of the number of accessed nodes in each level of the tree
def level_access_data(data, dim):
    data = np.array(data)

    avg = np.round(np.mean(data, axis=0), decimals=2)
    max_ = np.max(data, axis=0)
    std = np.round(np.std(data, axis=0), decimals=2)

    print (f'average of the qr_acs_{dim}d:', avg)
    print (f'max of the qr_acs_{dim}d:', max_)
    print (f'std of the qr_acs_{dim}d:', std)

    return avg, max_, std   

def max_over_tuple_accesslists(list_of_lists):
    # Transpose the nested lists using zip
    transposed = zip(*list_of_lists)
    # Find max values over columns
    result = [tuple(map(max, zip(*pairs))) for pairs in transposed]
    print (f'max over tuple accesslists of dimensions over queries:', result)
    return result

def std_max_levels(data, dim, max_access_level):
    data = np.array(data)

    # identify the levels with the maximum number of accessed nodes
    max_l = [i for i in range(len(max_access_level)) if max_access_level[i] == max(max_access_level)]

    max_l_queries={}
    for j in range(len(max_l)):
        acl = [data[i][max_l[j]] for i in range(len(data))]
        # calculate the std of acl 
        std = np.round(np.std(acl), decimals=2)
        max_l_queries[max_l[j]] = std
    
    # calculate the average of max_l_queries values
    avg = np.round(np.mean(list(max_l_queries.values())), decimals=2)
    print(f'average of the std of the accesslists over queries for levels with max values:', avg)
    
    return avg


def report(report_path_h, report_path_acs, dim, report_path_acs_dim=None, report_path_acs_lev_2d=None):
    print(colored(f"{'-'*25}QUERY RESULTS STATISTICS REPORT FOR {report_path_h.split('/')[-1].split('.')[0]}{'-'*25}", 'green'))

    print(colored('STOPPING HEIGHT REPORT: ', 'blue'))
    height_report_query(report_path_h, dim)

    print(colored('-'*50, 'blue'))
    print(colored('RETURNED NODES STATISTICS: ', 'blue'))
    data = load_csv_data(report_path_h)
    print('Power of 2 of the Average of the returned nodes in: ') 
    nodes_avg_pow2(data, dim)
    print('Average of the returned nodes in: ') 
    nodes_avg(data, dim)


    print(colored('-'*50, 'blue'))
    print(colored('ACCESS PER LEVEL STATISTICS: ', 'blue'))
    access_data = load_csv_data(report_path_acs)
    avg, max_, std = level_access_data(access_data, dim)

    if report_path_acs_dim:
        print(colored('Max ACCESS PER LEVEL STATISTICS: for acs_dims ', 'blue'))
        print(colored('-'*50+'max over tuple accesslists of dimensions over queries'+ '-'*50, 'green'))
        access_dims = load_csv_data_acs_dims(report_path_acs_dim)
        result = max_over_tuple_accesslists(access_dims)

    if report_path_acs_lev_2d:
        print(colored('ACCESS PER LEVEL STATISTICS: for acs_lev_2d ', 'blue'))
        print(colored('-'*50+'max over accesslists over queries per dimension'+ '-'*50, 'green'))
        access_lev_2d = load_csv_data_acs_dims(report_path_acs_lev_2d)
        result = max_over_tuple_accesslists(access_lev_2d)

    # print(colored('-'*50+'std over accesslists over queries'+ '-'*50, 'green'))
    # print(np.round(np.mean(std, axis=0), decimals=2))

def std_over_maxlevels():

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

    # for each dataset in the dataset_names list, call the std_max_levels function considering the dimension of the dataset
    for i in range(len(dataset_names)):
        print(colored(f'loading {dataset_names[i]}...', 'blue'))
        report_path_access = f'qr_acs_{dims[i]}d_{dataset_names[i]}.csv'
        std_max_levels(load_csv_data(report_path_access), dims[i], max_access_levels[i])



if __name__ == '__main__':
    dim = 1
    # Load the configuration from the file
    dataset_config.load_from_file()
    # get dim as argparse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=dim, help='Dimension of the dataset')
    parser.add_argument('--ds', type=str, default=dataset_config.dataset, help='Name of the dataset')
    parser.add_argument('--stm', type=bool, default=False, help='std over max levels')
    args = parser.parse_args()
    dim = args.dim
    dsname = args.ds
    print(f'dimension of the dataset: {dim}D')
    report_path_height = f'qr_h_{dim}d_{dsname}.csv'
    report_path_access = f'qr_acs_{dim}d_{dsname}.csv'
    report_path_acs_dim = f'qr_acs_dims_{dim}d_{dsname}.csv'
    report_path_acs_lev_2d = f'qr_acs_lev_{dim}d_{dsname}.csv'
    print(colored(f'HEIGHT AND ACCESS LEVEL REPORTS FOR THE DATASET:{dsname} ', 'green'))
    report(report_path_height, report_path_access, dim, report_path_acs_dim)



    print(colored('-'*50, 'blue'))
    if args.stm:
        # for all the datasets in dataset_config, calculate the std of the accesslists over queries for levels with max values
        print(colored('STD OF THE ACCESSLISTS OVER QUERIES FOR LEVELS WITH MAX VALUES: ', 'blue'))
        std_over_maxlevels()

    # for dataset in dataset_names:
    #     print(colored(f'loading {dataset}...', 'blue'))
    #     report_path_access = f'qr_acs_{dim}d_{dataset}.csv'
    # std_max_levels(load_csv_data(report_path_access), dim)



    # if dim == 3:
    #     print(colored('-'*50, 'blue'))
    #     print(colored('ACCESS PER LEVEL STATISTICS for 3d: ', 'blue'))
    #     access_data = load_csv_data(report_path_access)
    #     avg, max, std = level_access_data_3d(access_data, dim)


# height_report_query('rep/qr_h_2d_spitz.csv', 2)
# height_report_query('rep/qr_h_3d_nh.csv', 3)

# load csv data for 1D, 2D, and 3D
# data_1d = load_csv_data('rep/qr_h_1d_books.csv')
# data_2d = load_csv_data('rep/qr_h_2d_spitz.csv')
# data_3d = load_csv_data('rep/qr_h_3d_nh.csv')

# nodes_avg_pow2(data_1d, 1)
# nodes_avg_pow2(data_2d, 2)
# nodes_avg_pow2(data_3d, 3)

# nodes_avg(data_1d, 1)
# nodes_avg(data_2d, 2)
# nodes_avg(data_3d, 3)

# Calculating the average and maximum of the number of accessed nodes in each level of the tree
# access_data_1d = load_csv_data('rep/qr_acs_1d_books.csv')
# access_data_2d = load_csv_data('rep/qr_acs_2d_spitz.csv')
# access_data_3d = load_csv_data('rep/qr_acs_3d_nh.csv')


# avg_1d, max_1d, std_1d = level_access_data(access_data_1d, 1)
# avg_2d, max_2d, std_2d = level_access_data(access_data_2d, 2)
# avg_3d, max_3d, std_3d = level_access_data(access_data_3d, 3)

# draw a barplot of the average and maximum of the number of accessed nodes in each level of the tree
# barplot(avg_1d, 1, 'qr_acs_avg', label='Average_1d')
# barplot(max_1d, 1, 'qr_acs_max', label='Maximum_1d')

# barplot(avg_2d, 2, 'qr_acs_avg', label='Average_2d')
# barplot(max_2d, 2, 'qr_acs_max', label='Maximum_2d')

# barplot(avg_3d, 3, 'qr_acs_avg', label='Average_3d')
# barplot(max_3d, 3, 'qr_acs_max', label='Maximum_3d')


