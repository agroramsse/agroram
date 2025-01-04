import sys
import os
import pickle

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
seg_tree_3d = SegmentTree3D(pts_dict)

# merged_sgtree= merge_dim_trees(seg_tree_3d)
# with open(f'temp/{dataset_name}.pkl', 'wb') as f:
#     pickle.dump(merged_sgtree, f)

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

qr_h_, qr_acs= [], []
for i in tqdm(range(len(queries))):
    # query the segment tree with the random queries in the form of query (x_1,x_2, y_1,y_2, z_1, z_2)
    seg_tree_3d.query(queries[i][0][0], queries[i][0][1], queries[i][1][0], queries[i][1][1], queries[i][2][0], queries[i][2][1])
    qr_h_.append(seg_tree_3d.qr_h)
    # number of accesses per query in each level of the tree
    qr_acs.append(seg_tree_3d.access)

# write the qr_h_ to a file
write_csv(qr_h_, f'qr_h_3d_{dataset_config.dataset}')
write_csv(qr_acs, f'qr_acs_3d_{dataset_config.dataset}')

print(colored(f'{sample_num} random queries completed successfully', 'green'))
print(colored('-' * 50, 'green'))   

# calling the script results.py to generate the report with the argument dim=3
os.system('python experiment/results.py --dim 3')
