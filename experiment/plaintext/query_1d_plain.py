import sys
import os
import time
import csv


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import query_config, dataset_config
from utils import count_range_nodes, generate_random_query, write_csv
from termcolor import colored
from segtree import SegmentTree1D

# ! Set the number of query samples and dataset name before running the experiment in the config.py file

print(colored('-'*50, color= 'red'), 
      colored(f' \033[1D Dataset: loading and creating the Segment Tree\033[0m', color= 'red'),
      colored('-'*50, color= 'red'))

dataset_path = dataset_config.dataset_path
dataset_name = dataset_path.split('/')[-1].split('.')[0]
# print (colored('\033[1mAmazon book 1D dataset\033[0m', 'blue'))
data = []
# open file with relative path
print(colored(f'Loading the {dataset_name} 1D dataset for the 1D segment tree...', 'blue'))

with open(dataset_config.dataset_path, 'r') as file:
#with open('datasets/amazon-books.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(int(row[0]))

# data.sort()

# remove duplicates from the data
data = list(set(data))
# sort the data
data.sort()
print('The data loaded from csv- size:',len(data))

print('The range of values in the data list:',data[0], data[-1])

# defining the domain values range
domain = (data[0], data[-1])
print('domain range', domain)

# make dictionary from the data, assicate each value with 1
data_dict = {val: 1 for val in data}

# print(data_dict)
print(colored('Creating a segment tree from the data_dict...', 'blue'))
seg_tree = SegmentTree1D(data_dict)

# Query the sum of values in range [9504, 154656]:
# print(f"Sum of values in range [9504, 154656]: {seg_tree.query(9504, 158976)}")

print('Number of nodes:', seg_tree.count_nodes())
# print('Number of leaves:', seg_tree.count_leaves())
print('Height of the tree:', seg_tree.height())
print('Is the tree balanced:', seg_tree.is_balanced())

# query the segment tree with a sample range
# print(f"Sum of values in range [9504, 76896]: {seg_tree.query(9504, 165000)}")
# print('Number of returned nodes:', len(seg_tree.qr_result))

print(colored('segment_tree was created successfully!', 'green'))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
print('-'*100)
#---------------------------------------------------------------------------------
print(colored('-' * 50, 'green'))
sample_num = query_config.sample_num
# generating sample_num random queries in range data[0], data[-1]
print(colored(f'Generating {sample_num} random queries in the domain range: {data[0]}-{data[-1]}...', 'blue'))
import random
from tqdm import tqdm

queries = []
for i in range(sample_num):
    queries.append(generate_random_query(num_dims = 1, x_range = (data[0], data[-1])))

print(colored(f'{sample_num} random queries generated', 'green'))
print(colored('-' * 50, 'green')) 
# query the segment tree with the random queries
# per query, get the number of returned nodes and access
print(colored('Querying the segment tree with {sample_num} random queries...','blue'))

qr_h_, qr_acs = [], []
for i in tqdm(range(len(queries))):
    seg_tree.query(queries[i][0], queries[i][1])
    qr_h_.append(seg_tree.qr_h)
    # number of accesses per query in each level of the tree
    qr_acs.append(seg_tree.access)


# write the qr_h_ to a file
write_csv(qr_h_, f'qr_h_1d_{dataset_config.dataset}')
# write the qr_acs to a file
write_csv(qr_acs, f'qr_acs_1d_{dataset_config.dataset}')

print(colored(f'{sample_num} random queries completed successfully', 'green'))
print(colored('-' * 50, 'green'))   
 
# calling the script results.py to generate the report with the argument dim=1
os.system('python experiment/results.py --dim 1')

