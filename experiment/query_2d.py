import sys
import os
import time
import pickle
# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import csv

from termcolor import colored
from tqdm import tqdm

from segtree import SegmentTree2D, merge_dim_trees
from utils import count_range_nodes, generate_random_query, write_csv, load_dataset
from config import query_config, dataset_config

# ! Set the number of query samples and dataset name before running the experiment in the config.py file

print(colored('-'*100, 'green'))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# *Loading 2D dataset for the 2D segment tree

dataset_path = dataset_config.dataset_path

# read the dataset_path extension and store as string in a type_ variable
type_ = dataset_path.split('.')[-1]
# from the dataset_path, get the dataset name
dataset_name = dataset_path.split('/')[-1].split('.')[0]

print(colored(f'\033[1m{dataset_name} 2d dataset\033[0m', 'blue'))

pts, pts_dict, x_range, y_range = load_dataset(dataset_path, type_=type_)

print(colored('Creating a segment tree from the data_dict...', 'blue'))
seg_tree_2d = SegmentTree2D(pts_dict)


# Print tree stats
print('Number of nodes:', seg_tree_2d.count_nodes())
print('Height of the tree:', seg_tree_2d.height())
print('Is the tree balanced:', seg_tree_2d.is_balanced())

print(colored('2Dimensional segment_tree was created successfully!', 'green'))
print('-'*100)

range_ = ((200, 700), (200, 700))
print('Number of points in the range: ', count_range_nodes(pts, *range_))    
# query the segment tree with a sample range
print(colored(f'Test the sample range sum query: sum{range_} ','blue'))
print(f"Sum of values in range {range_}]: {seg_tree_2d.query(range_[0][0], range_[0][1], range_[1][0], range_[1][1])}")
print('Number of returned nodes:', len(seg_tree_2d.qr_result))
# print non-empty nodes in the tree
print('Number of non-empty nodes:', seg_tree_2d.count_non_empty_nodes())    

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

# *-------------------------ORAM-----------------------------
print(colored('-'*50, color= 'red'), 
      colored('\033[1mORAM\033[0m', color= 'red'), 
      colored('-'*50, color= 'red'))

from b4_oram import OMapE
from o_segmenttree import OSegmentTree
import os


# print(colored('Building the OMAP...', 'blue'))
# omap = OMapE("./", total_accesses=1)

print(colored('Building the O Segment Tree...', 'blue'))
# access_per_level = [1,2,4,6,10,18,24,28,33,39,41,39,42,39,35,30,25,20,12,7,4,2]
access_per_level = dataset_config.access_per_level
o_segmenttree = OSegmentTree("./", accesses_per_level=access_per_level)

# removing heap files in the directory
[os.remove(file) for file in os.listdir() if file.startswith('heap')]

# * --------------------------------------------------2D dataset: OSegmentTree-----------------------------------------------------

print('-'*50, colored(f'\033[1m2D {dataset_name} dataset ORAM statistics:\033[0m','green'), '-'*50)

# merge the segment tree
print(colored('Merging the segment trees into a single big tree', 'blue'))
merged_sgtree= merge_dim_trees(seg_tree_2d)

# save the merged segment tree as pickle file in temp directory
# with open(f'temp/merged_sgtree_{dataset_name}.pkl', 'wb') as f:
#     pickle.dump(merged_sgtree, f)

# apply the merged segment tree to the ORAM
print(colored('Applying the merged segment tree to the ORAM...', 'blue'))
t_start = time.time()
o_segmenttree.apply(merged_sgtree.tree)
print("Time to create oblivious structure ", time.time() - t_start)
print(colored('Merged segment tree applied to the ORAM: DONE', 'blue'))

range_ = ((0, 5), (448, 460))
# query the segment tree with a sample range
print(colored(f'Test OMAP with the sample range sum query: sum[{range_}] ','blue'))
print('Number of points in the range: ', count_range_nodes(pts, *range_))
print(f"Sum of values in range [(0, 448)-(5, 460)]: {seg_tree_2d.query(0, 5, 448, 460)}")

start = time.time()
(nodes_, access_, time_per_depth_) = o_segmenttree.search(seg_tree_2d.accessed_nodes)

print("Time to run query",time.time()-start)
print("Parallel time to run query",sum(time_per_depth_))
print("Time to run query",time.time()-start)
print('accessed nodes', len(nodes_))
total_depth_accessed = 0
for a in access_:
    total_depth_accessed += (len(access_[a]) > 0)
print('total depth', total_depth_accessed)

print(colored('OMAP Search Test passed!', 'green'))

print(colored('-' * 50, 'green'))
sample_num = query_config.sample_num 
# generating sample_num random queries in range x: (pts[0][0], pts[-1][0]) and y (pts[0][1], pts[-1][1])
print(colored(f'Generating {sample_num} random queries in the domain range: x:{x_range} y:{y_range}', 'blue'))
import random
from tqdm import tqdm

queries = []
for i in tqdm(range(sample_num)):
    queries.append(generate_random_query(2, x_range, y_range))

print(colored(f'{sample_num} random queries generated', 'green'))
print(colored('-' * 50, 'green')) 
# query the segment tree with the random queries
# per query, get the number of returned nodes and access
print(colored(f'Querying the segment tree with {sample_num} random queries...','blue'))

nodes_, returned_nodes_, access_, qr_h_, qr_acs, times, parallel_times, plaintext_access_ = [], [], [], [], [], [], [], []
for i in tqdm(range(len(queries))):
    # query the segment tree with the random queries in the form of query (x_1,x_2, y_1,y_2)
    seg_tree_2d.query(queries[i][0][0], queries[i][0][1], queries[i][1][0], queries[i][1][1])
    qr_h_.append(seg_tree_2d.qr_h)
    returned_nodes_.append(len(seg_tree_2d.qr_result))
    qr_acs.append(seg_tree_2d.access)
    if not seg_tree_2d.accessed_nodes:
        continue
    start = time.time()
    (nodes, access, time_per_depth_) = o_segmenttree.search(seg_tree_2d.accessed_nodes)
    query_depth = 0
    # print(access)
    for a in access:
        query_depth += (len(access[a]) > 0)
    plaintext_access_.append(len(seg_tree_2d.accessed_nodes))
    times.append(time.time() - start)
    nodes_.append(len(nodes))
    access_.append(query_depth)
    parallel_times.append(sum(time_per_depth_))

# write the qr_h_ to a file
write_csv(qr_h_, f'qr_h_2d_{dataset_name}')
write_csv(qr_acs, f'qr_acs_2d_{dataset_name}')

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