import sys
import os
import pickle

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import csv
import time

from termcolor import colored
from segtree import SegmentTree3D, merge_dim_trees
from utils import count_range_nodes, generate_random_query, write_csv, load_dataset_3d
from config import query_config, dataset_config

print(colored('-'*100, 'green'))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# *loading 3D dataset

dataset_path = dataset_config.dataset_path

# read the dataset_path extension and store as string in a type_ variable
type_ = dataset_path.split('.')[-1]
# from the dataset_path, get the dataset name
dataset_name = dataset_path.split('/')[-1].split('.')[0]

print(colored(f'\033[1m{dataset_name} 3d dataset\033[0m', 'blue'))

pts, pts_dict, x_range, y_range, z_range = load_dataset_3d(dataset_path, type_=type_)

print(colored('Creating a segment tree from the 3d points...', 'blue'))
# create a slice of the first 100 points from the points dictionary
# pts_dict = {k: pts_dict[k] for k in list(pts_dict)[:100]}
seg_tree_3d = SegmentTree3D(pts_dict)

# Print tree stats
print('Number of nodes:', seg_tree_3d.count_nodes())
print('Height of the tree:', seg_tree_3d.height())
print('Is the tree balanced:', seg_tree_3d.is_balanced())

# print first 50 elements in the points
# print('First 50 elements in the points:', pts[:50])

range_ = ((0, 40), (0, 40), (0, 40))
print('Number of points in the range: ', count_range_nodes(pts, *range_))    
# query the segment tree with a sample range
print(colored(f'Test the sample range sum query: sum{range_} ','blue'))
# print time taken to query the range
start_time = time.time()
print(f"Sum of values in range {range_}]: {seg_tree_3d.query(range_[0][0], range_[0][1], range_[1][0], range_[1][1], range_[2][0], range_[2][1])}")
end_time = time.time()
# print the time taken to query the range in milliseconds
print('Time taken to query the range:', (end_time - start_time) * 1000, 'ms')
print('Number of returned nodes:', len(seg_tree_3d.qr_result))

# counting non-empty nodes in the tree
print('Number of non-empty nodes:', seg_tree_3d.count_non_empty_nodes())

#*-------------------------ORAM-----------------------------
# create an ORAM instance
print(colored('-'*50, color= 'red'), 
      colored('\033[1mORAM\033[0m', color= 'red'), 
      colored('-'*50, color= 'red'))

from b4_oram import OMapE
from o_segmenttree import OSegmentTree
import os

# print(colored('Building the OMAP...', 'blue'))
# omap = OMapE("./", total_accesses=1)
print(colored('Building the O Segment Tree for 3d dataset...', 'blue'))
# access_per_level = [1,2,4,6,9,13,17,25,36,50,62,77,88,97,96,93,90,69,43,20,8]
o_segmenttree = OSegmentTree("./", accesses_per_level=dataset_config.access_per_level)

# removing heap files in the directory
[os.remove(file) for file in os.listdir() if file.startswith('heap')]

# * --------------------------------------------------3D  dataset: ORAM-----------------------------------------------------
print('-'*50, colored(f'\033[1m3D {dataset_name} dataset ORAM statistics:\033[0m','green'), '-'*50)

# merge the segment tree 3d into the bigtree
print(colored('Merging the segment trees into a single big tree', 'blue'))
merged_sgtree = merge_dim_trees(seg_tree_3d)

# with open(f'temp/merged_sgtree_{dataset_name}.pkl', 'wb') as f:
#     pickle.dump(merged_sgtree, f)

# apply the merged segment tree to the ORAM
print(colored('Applying the merged segment tree to the ORAM...', 'blue'))
t_start = time.time()
o_segmenttree.apply(merged_sgtree.tree)
print("Time to create oblivious structure ", time.time() - t_start)
print(colored('Merged segment tree applied to the ORAM: DONE', 'blue'))

# *Generating queryies and querying the segment tree with the queries, query the omap with the returned nodes
# query the segment tree with a sample range
print(colored(f'Test OMAP with the sample range sum query: sum[{range_}] ','blue'))
print(f"Sum of values in range {range_}: {seg_tree_3d.query(range_[0][0], range_[0][1], range_[1][0], range_[1][1], range_[2][0], range_[2][1])}")
print('Number of returned nodes:', len(seg_tree_3d.qr_result))

print(colored(f'searching the osegment tree with the accessed nodes for the range {range_}', 'green'))
start = time.time()
(nodes_, access_, time_per_depth_) = o_segmenttree.search(seg_tree_3d.accessed_nodes)
print("Time to run query",time.time()-start)
print("Parallel time to run query",sum(time_per_depth_))
print("Time to run query",time.time()-start)
print("Plaintext access",len(seg_tree_3d.accessed_nodes))

print(colored('Test passed', 'green'))
print('accessed nodes', len(nodes_))
total_depth_accessed = 0
for a in access_:
    total_depth_accessed += (len(access_[a]) > 0)
print('total depth', total_depth_accessed)
print(colored('OMAP Search Test passed!', 'green'))

print(colored('-' * 50, 'green')) 
sample_num = query_config.sample_num
# generating sample_num random queries in range x: (pts[0][0], pts[-1][0]) and y (pts[0][1], pts[-1][1])
print(colored(f'Generating {sample_num} random queries in the domain range: x:{x_range} , y:{y_range}, z:{z_range}', 'blue'))
import random
from tqdm import tqdm

queries = []
for i in tqdm(range(sample_num)):
    queries.append(generate_random_query(3, x_range, y_range, z_range))

print(colored(f'{sample_num} random queries generated', 'green'))
print(colored('-' * 50, 'green')) 
# query the segment tree with the random queries
# per query, get the number of returned nodes and access
print(colored(f'Querying the segment tree with {sample_num} random queries...','blue'))

nodes_, returned_nodes_, access_, qr_h_, qr_acs, times, parallel_times, plaintext_access_ = [], [], [], [], [], [], [], []
for i in tqdm(range(len(queries))):
    # query the segment tree with the random queries in the form of query (x_1,x_2, y_1,y_2, z_1, z_2)
    seg_tree_3d.query(queries[i][0][0], queries[i][0][1], queries[i][1][0], queries[i][1][1], queries[i][2][0], queries[i][2][1])
    qr_h_.append(seg_tree_3d.qr_h)
    # number of accesses per query in each level of the tree
    qr_acs.append(seg_tree_3d.access)
    # calling omap search on qr_result
    returned_nodes_.append(len(seg_tree_3d.qr_result))

    if not seg_tree_3d.accessed_nodes:
        continue
    start = time.time()
    (nodes, access, time_per_depth_) = o_segmenttree.search(seg_tree_3d.accessed_nodes)
    query_depth = 0
    for a in access:
        query_depth += (len(access[a]) > 0)

    plaintext_access_.append(len(seg_tree_3d.accessed_nodes))
    times.append(time.time() - start)
    nodes_.append(len(nodes))
    access_.append(query_depth)
    parallel_times.append(sum(time_per_depth_))

# write the qr_h_ to a file
write_csv(qr_h_, f'qr_h_3d_{dataset_name}')
write_csv(qr_acs, f'qr_acs_3d_{dataset_name}')

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
