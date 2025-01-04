import sys
import os
import time
import pickle
# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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

# create merged segment tree and store as pickle file
# merged_sgtree= merge_dim_trees(seg_tree_2d)

# with open(f'temp/{dataset_name}.pkl', 'wb') as f:
#     pickle.dump(merged_sgtree, f)


# Print tree stats
print('Number of nodes:', seg_tree_2d.count_nodes())
print('Height of the tree:', seg_tree_2d.height())
print('Is the tree balanced:', seg_tree_2d.is_balanced())

print(colored('2Dimensional segment_tree was created successfully!', 'green'))
print('-'*100)

range_ = ((200, 700), (200, 700))
print('Number of points in the range: ', count_range_nodes(pts, *range_))  
# caluclate the sum of values in the range using pts_dict
print(f"Sum of values in range {range_}: {sum([pts_dict[pt] for pt in pts_dict if range_[0][0] <= pt[0] <= range_[0][1] and range_[1][0] <= pt[1] <= range_[1][1]])}")  
# query the segment tree with a sample range
print(colored(f'Test the sample range sum query: sum{range_} ','blue'))
print(f"Sum of values in range {range_}: {seg_tree_2d.query(range_[0][0], range_[0][1], range_[1][0], range_[1][1])}")
print('Number of returned nodes:', len(seg_tree_2d.qr_result))
# print non-empty nodes in the tree
print('Number of non-empty nodes:', seg_tree_2d.count_non_empty_nodes())  

sample_num = query_config.sample_num 
# generating sample_num random queries in range x: (pts[0][0], pts[-1][0]) and y (pts[0][1], pts[-1][1])
print(colored(f'Generating {sample_num} random queries in the domain range: x:{x_range} y:{y_range}', 'blue'))

# exit()
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

qr_h_, qr_acs = [], []
for i in tqdm(range(len(queries))):
    # query the segment tree with the random queries in the form of query (x_1,x_2, y_1,y_2)
    seg_tree_2d.query(queries[i][0][0], queries[i][0][1], queries[i][1][0], queries[i][1][1])
    qr_h_.append(seg_tree_2d.qr_h)
    qr_acs.append(seg_tree_2d.access)

# write the qr_h_ to a file
write_csv(qr_h_, f'qr_h_2d_{dataset_config.dataset}')
write_csv(qr_acs, f'qr_acs_2d_{dataset_config.dataset}')

print(colored(f'{sample_num} random queries completed successfully', 'green'))
print(colored('-' * 50, 'green'))   

# calling the script results.py to generate the report with the argument dim=2
os.system('python experiment/results.py --dim 2')