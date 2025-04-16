import os
import sys
import argparse
# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import query_config, dataset_config
from utils import generate_max_access_list


datasets_1d = ['amazon-books', 'gowalla']
datasets_2d = ['spitz-1024x1024', 'cali-1024x1024', 'gowalla_100k', 'gowalla_50k','synthetic_2d_1m', 'synthetic_2d_1m_sparse', 'synthetic_2d_1m-1024x1024' ]
datasets_3d =  ['nh_64','gowalla_3d_23k', 'synthetic_3d_1m_128', 'synthetic_3d_1m_256']


parser = argparse.ArgumentParser(description='Run experiments for different datasets.')
# add argument for dataset that is optional to provide
parser.add_argument('-q', '--q_num', type=int, default=1000, help='Number of query samples')
parser.add_argument('-d', '--dataset', type=str, default='all', help='Dataset name')

# parse the arguments
args = parser.parse_args()
query_config.sample_num = args.q_num
# save the query config to file
query_config.save_to_file()

dataset_config.dataset = args.dataset

# if dataset is not provided, run for all datasets
if args.dataset == 'all':
    datasets_1d = datasets_1d
    datasets_2d = datasets_2d
    datasets_3d = datasets_3d
else:
    datasets_1d = [args.dataset] if args.dataset in datasets_1d else []
    datasets_2d = [args.dataset] if args.dataset in datasets_2d else []
    datasets_3d = [args.dataset] if args.dataset in datasets_3d else []
# if dataset is not found, exit
if not datasets_1d and not datasets_2d and not datasets_3d:
    print(f'Dataset {args.dataset} not found.')
    sys.exit(1)
# if dataset is found, run the experiment
if not os.path.exists('log'):
    os.makedirs('log')


for dataset in datasets_1d:
    dataset_config.dataset = dataset
    dataset_config.update()
    dataset_config.save_to_file()
    print(f'running experiment for {dataset}...')
    os.system(f'python3 experiment/query_1d.py > log/1d_{dataset}_crypto.log')
    print(f'completed experiment for {dataset}...')

for dataset in datasets_2d:
    dataset_config.dataset = dataset
    dataset_config.access_per_level = generate_max_access_list(dataset)
    dataset_config.update()
    dataset_config.save_to_file()
    print(f'Setting max access list for {dataset}: {dataset_config.access_per_level}')
    print(f'running experiment for {dataset}...')
    os.system(f'python3 experiment/query_2d.py > log/2d_{dataset}_crypto.log')
    print(f'completed experiment for {dataset}...')

for dataset in datasets_3d:
    dataset_config.dataset = dataset
    dataset_config.access_per_level = generate_max_access_list(dataset)
    dataset_config.update()
    dataset_config.save_to_file()
    print(f'Setting max access list for {dataset}: {dataset_config.access_per_level}')
    print(f'running experiment for {dataset}...')
    os.system(f'python3 experiment/query_3d.py > log/3d_{dataset}_crypto.log')
    print(f'completed experiment for {dataset}...')
