
import csv
import os
import sys
import argparse

from config import dataset_config
from termcolor import colored
from utils import load_dataset, load_dataset_3d
import math

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def data_bucket(m, n, data, alpha=0.5):
    "m: domain range (min, max), n: number of data points in the domain range, data: list of data points, alpha: bucket size parameter"
    bucket_size={}
    num_buckets=1
    for dim in range(len(m)):
        bucket_size[dim]=math.ceil((m[dim][1]-m[dim][0])**alpha)
        num_buckets*=math.ceil((m[dim][1]-m[dim][0])/(bucket_size[dim]))
    print('Bucket Size:', bucket_size)

    # Maxumum bucket size: count values in each bucket and get the max
    max_bucket_size = 0
    i=0
    to_continue=True
    while to_continue:
        bucket_min={}
        bucket_max={}
        bucket_size_current=0
        for dim in range(len(m)):
            bucket_min[dim]=m[dim][0]+i*bucket_size[dim]
            if bucket_min[dim]>m[dim][1]:
                to_continue=False
            bucket_max[dim]=m[dim][0]+(i+1)*bucket_size[dim]-1
        if not to_continue:
            break
        for d in data:
            for dim in range(len(m)):
                if bucket_min[dim] <= d[dim] < bucket_max[dim]:
                    bucket_size_current+=1
        max_bucket_size = max(max_bucket_size, bucket_size_current)
        i=i+1
        #print(bucket)
        

    print('Maximum Bucket Size:', max_bucket_size)
    print('Number of buckets: ', num_buckets)
    # # Calculate the bucketed_MMAP size
    print('The bucketed_MMAP size...', num_buckets*max_bucket_size)

    return num_buckets*max_bucket_size



def emt_1d():

    # Load the configuration from the file
    dataset_config.load_from_file()



    dataset_path = dataset_config.dataset_path
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    # print (colored('\033[1mAmazon book 1D dataset\033[0m', 'blue'))
    data = []
    # Loading the 1D dataset
    print(colored('-'*50, color= 'red'), 
        colored(f' \033[1D Dataset: EMT Bucketed_MMAP Size\033[0m for {dataset_name}: ', color= 'red'),
        colored('-'*50, color= 'red'))

    with open(dataset_config.dataset_path, 'r') as file:
    #with open('datasets/amazon-books.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(int(row[0]))

    # remove duplicates from the data
    data = list(set(data))
    data.sort()

    print('The data loaded from csv- size:',len(data))

    print('The range of values in the data list:',data[0], data[-1])

    # Dataset Parameters
    alpha = 0.5
    N = len(data)
    M = (data[0], data[-1])
    print('Dataset Size:', N)
    print('domain range', M)

    bucket_size = int((M[1]-M[0])**alpha)
    print('Bucket Size:', bucket_size)

    # Maxumum bucket size: count values in each bucket and get the max
    max_bucket_size = 0
    for i in range(M[0], M[1], bucket_size):

        bucket = [d for d in data if i <= d < i + bucket_size]
        # print(bucket)
        max_bucket_size = max(max_bucket_size, len(bucket))

    print('Maximum Bucket Size:', max_bucket_size)
    print('Number of buckets: ', ((M[1]-M[0])/(bucket_size)))
    # Calculate the bucketed_MMAP size
    print('The bucketed_MMAP size...', round(((M[1]-M[0])/(bucket_size)) * max_bucket_size))


def emt_2d():

    print(colored('-'*100, 'green'))
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    # *Loading 2D dataset 

    dataset_path = dataset_config.dataset_path
    print(colored(f'Loading 2D dataset from {dataset_path}...', 'blue'))
    # read the dataset_path extension and store as string in a type_ variable
    type_ = dataset_path.split('.')[-1]
    # from the dataset_path, get the dataset name
    dataset_name = dataset_path.split('/')[-1].split('.')[0]

    print(colored(f'\033[1m{dataset_name} 2d dataset\033[0m', 'blue'))

    pts, pts_dict, x_range, y_range = load_dataset(dataset_path, type_=type_)

    # dataset parameters
    alpha = 0.5
    N = len(pts)
    M = (x_range, y_range)
    print('Dataset Size:', N)
    print('Domain Range:', M)
    # print(colored('-'*100, 'green'))

    # for each dimension call the data_bucket function
    data_bucket_MMAP_size = 0
    data_bucket_MMAP_size += data_bucket(M, N, [p for p in pts], alpha=alpha)
    
    print(f'The bucketed_MMAP size for 2D dataset{dataset_name}:', round(data_bucket_MMAP_size))


 
def emt_3d():
    dataset_path = dataset_config.dataset_path
    print(colored(f'Loading 3D dataset from {dataset_path}...', 'blue'))
    # read the dataset_path extension and store as string in a type_ variable
    type_ = dataset_path.split('.')[-1]
    # from the dataset_path, get the dataset name
    dataset_name = dataset_path.split('/')[-1].split('.')[0]

    print(colored(f'\033[1m{dataset_name} 3d dataset\033[0m', 'blue'))

    pts, pts_dict, x_range, y_range, z_range = load_dataset_3d(dataset_path, type_=type_)
    

    # dataset parameters
    alpha = 0.5
    N = len(pts)
    M = (x_range, y_range, z_range)
    print('Dataset Size:', N)
    print('Domain Range:', M)
    # print(colored('-'*100, 'green'))
    # for each dimension call the data_bucket function
    data_bucket_MMAP_size = 0
    data_bucket_MMAP_size = data_bucket(M, N, [p for p in pts], alpha=alpha)

    print(f'The bucketed_MMAP size for 3D dataset{dataset_name}:', round(data_bucket_MMAP_size))




datasets_1d = ['amazon-books', 'gowalla']
datasets_2d = ['spitz-1024x1024', 'cali-1024x1024', 'gowalla_100k', 'gowalla_50k','synthetic_2d_1m', 'synthetic_2d_1m_sparse', 'synthetic_2d_1m-1024x1024' ]
datasets_3d =  ['nh_64','gowalla_3d_23k', 'synthetic_3d_1m_128', 'synthetic_3d_1m_256']


parser = argparse.ArgumentParser(description='Run experiments for different datasets.')
parser.add_argument('-d', '--dataset', type=str, default='all', help='Dataset name')

# parse the arguments
args = parser.parse_args()
dataset_config.dataset = args.dataset

# if dataset is not provided, run for all datasets
if args.dataset != 'all':
    datasets_1d = [args.dataset] if args.dataset in datasets_1d else []
    datasets_2d = [args.dataset] if args.dataset in datasets_2d else []
    datasets_3d = [args.dataset] if args.dataset in datasets_3d else []
# if dataset is not found, exit
if not datasets_1d and not datasets_2d and not datasets_3d:
    print(f'Dataset {args.dataset} not found.')
    sys.exit(1)
# if dataset is found, run the experiment
# if not os.path.exists('log'):
#     os.makedirs('log')


for dataset in datasets_1d:
    dataset_config.dataset = dataset
    dataset_config.update()
    dataset_config.save_to_file()
    print(f'Calculating the EMT MMAP-Bucket for {dataset}...')
    emt_1d()
    print(f'DONE for {dataset}...')

for dataset in datasets_2d:
    dataset_config.dataset = dataset
    dataset_config.update()
    dataset_config.save_to_file()

    print(f'running experiment for {dataset}...')
    emt_2d()
    print(f'completed experiment for {dataset}...')

for dataset in datasets_3d:
    dataset_config.dataset = dataset
    dataset_config.update()
    dataset_config.save_to_file()
    print(f'running experiment for {dataset}...')
    emt_3d()
    print(f'completed experiment for {dataset}...')
