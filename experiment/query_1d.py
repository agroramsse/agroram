import sys
import os
import time


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import query_config, dataset_config

from termcolor import colored

# ! Set the number of query samples and dataset name before running the experiment in the config.py file

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import count_range_nodes, generate_random_query, write_csv

from config import query_config

# utility functions
# def next_power_of_2(x):
#     return 1 if x == 0 else 2 ** (x - 1).bit_length()

# Loading the amazon book 1D dataset for the 1D segment tree
import csv
# from segmentree import SegmentTree
from segtree import SegmentTree1D

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

data.sort()

print('The data loaded from csv- size:',len(data))

print('The range of values in the data list:',data[0], data[-1])

# defining the domain values range
domain = (data[0], data[-1])
print('domain range', domain)

# print first 10 elements of the data
# print('First 10 elements of the data:', data[:20])

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
#---------------------------------------------------------------------------------


#*-------------------------ORAM-----------------------------
# * amazon book dataset: ORAM
print(colored('-'*50, color= 'red'), 
      colored('\033[1mORAM\033[0m', color= 'red'), 
      colored('-'*50, color= 'red'))

from b4_oram import OMapE
from o_segmenttree import OSegmentTree
import os

print(colored('Building the O Segment Tree...', 'blue'))

o_segmenttree = OSegmentTree("./", accesses_per_level=dataset_config.access_per_level)


# removing heap files in the directory
[os.remove(file) for file in os.listdir() if file.startswith('heap')]


print('-'*50,colored(f'\033[1m1D {dataset_name} dataset: ORAM statistics\033[0m', 'green'),'-'*40)
# store the segment tree node identifiers and their corresponding data in a dictionary, the following list contains elements in the form of Node(tag=2046, identifier=2046, data=8)
# tree_data = seg_tree.tree.all_nodes()
# # the data is the value of the node
# data_dict = {node.identifier: node.data for node in tree_data}

# pad the above data_dict to the next power of 2 of the size of the data_dict
#TODO: this doesn't do anything right now
# max_pow2 = next_power_of_2(len(data_dict))

# print('size of the data before padding', data_dict.__len__())
# # get the max key of the data_dict
# max_key = max(data_dict.keys())
# # pad the data_dict to the next power of 2 size of the data_dict with the keys incrementing from the max key of the data_dict
# data_dict = {k: 1 for k in range(max_key + 1, max_pow2 + max_key + 1)}

# print('size of the data after padding', data_dict.__len__())

# omap.apply(data_dict)
# (nodes, returned_nodes, access) = omap.search(1)
# print(nodes)
# print(returned_nodes)
# print(access)

# query the segment tree with a sample range
t_start = time.time()
o_segmenttree.apply(seg_tree.tree)
print("Time to create oblivious structure ", time.time() - t_start)
print(colored('Test OMAP with the sample range sum query: sum(9504, 165000) ','blue'))
print(f"Sum of values in range [9504, 76896]: {seg_tree.query(9504, 165000)}")
print("Accessed",seg_tree.accessed_nodes)
print('Number of returned nodes:', len(seg_tree.qr_result))
# calling omap search on qr_result
print('Calling omap search on qr_result:')

start = time.time()
(nodes_, access_, time_per_depth_) = o_segmenttree.search(seg_tree.accessed_nodes)

    # print('length of the nodes list: ', len(nodes))
    # print('returned nodes', len(returned_nodes))
    # print('number of accesses', len(access))
print("Time to run query",time.time()-start)
print("Parallel time to run query",sum(time_per_depth_))


# print the nodes, returned nodes and access lists lengths for the sample range query
print('length of the nodes list: ', len(nodes_))
print('number of accesses', len(access_))

print(colored('Test passed', 'green'))

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

nodes_, returned_nodes_, access_, qr_h_, qr_acs, times, parallel_times, plaintext_access_ = [], [], [], [], [], [], [], []
for i in tqdm(range(len(queries))):
    seg_tree.query(queries[i][0], queries[i][1])
    qr_h_.append(seg_tree.qr_h)
    # number of accesses per query in each level of the tree
    qr_acs.append(seg_tree.access)
    returned_nodes_.append(len(seg_tree.qr_result))
    start = time.time()
    (nodes, access, time_per_depth_) = o_segmenttree.search(seg_tree.accessed_nodes)

    query_depth = 0
    for a in access:
        query_depth += (len(access[a])>0)

    plaintext_access_.append(len(seg_tree.accessed_nodes))
    times.append(time.time()-start)
    nodes_.append(len(nodes))
    access_.append(query_depth)
    parallel_times.append(sum(time_per_depth_))

# write the qr_h_ to a file
write_csv(qr_h_, 'qr_h_1d')
# write the qr_acs to a file
write_csv(qr_acs, 'qr_acs_1d')

print(colored(f'{sample_num} random queries completed successfully', 'green'))
print(colored('-' * 50, 'green'))   
# print the average of the values in the nodes, returned nodes and access lists
print('average of the number of nodes accessed:', sum(nodes_)/sample_num)
print('average returned nodes', sum(returned_nodes_)/sample_num)
print('average of the depth:', sum(access_)/sample_num)
print('average plaintext access', sum(plaintext_access_)/sample_num)
print('average of the number of round trips:', sum(access_)/sample_num+1)
print('average time:', sum(times)/sample_num)
print('average parallel time', sum(parallel_times)/sample_num)
storage_size = 0
for file in os.listdir():
    if file.startswith('heap'):
        storage_size = storage_size+os.stat(file).st_size
print('storage_size:', storage_size/10**6, "MB")



print(colored('-' * 50, 'green')) 

#  remove the heap files in the directory
print(colored('Removing heap files in the directory...', 'blue'))
[os.remove(file) for file in os.listdir() if file.startswith('heap')]
print(colored('Heap files removed', 'green'))
