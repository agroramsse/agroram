import sys
import os

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from termcolor import colored
from utils import report_h_dev, plot_histogram, load_csv_data, next_power_of_2, barplot
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
    plot_histogram(qr_h_avg, max, dim, name=f'qr_h_avg', label='avg')


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
    max = np.max(data, axis=0)
    std = np.round(np.std(data, axis=0), decimals=2)

    print (f'average of the qr_acs_{dim}d:', avg)
    print (f'max of the qr_acs_{dim}d:', max)
    print (f'std of the qr_acs_{dim}d:', std)

    return avg, max, std   


def report(report_path_h, report_path_acs, dim):
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
    avg, max, std = level_access_data(access_data, dim)


if __name__ == '__main__':
    dim = 2
    report_path_height = 'rep/qr_h_3d_gowalla.csv'
    report_path_access = 'rep/qr_acs_3d_gowalla.csv'
    report(report_path_height, report_path_access, dim)


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


