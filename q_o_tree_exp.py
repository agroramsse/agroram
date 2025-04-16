import os
import sys

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import query_config, dataset_config
from utils import generate_max_access_list


datasets_2d = ['cali-1024x1024', 'spitz-1024x1024', 'gowalla_100k', 'gowalla_50k','synthetic_2d_1m', 'synthetic_2d_1m_sparse', 'synthetic_2d_1m-1024x1024' ]
datasets_3d =  ['nh_64','gowalla_3d_23k', 'synthetic_3d_1m_128', 'synthetic_3d_1m_256']


for dataset in datasets_2d:
    dataset_config.dataset = dataset
    dataset_config.update()
    dataset_config.save_to_file()
    print(f'running experiment for {dataset}...')
    os.system(f'python3 quadtree.py > log/2d_quadt_{dataset}.log')
    print(f'completed experiment for {dataset}...')

for dataset in datasets_3d:
    dataset_config.dataset = dataset
    dataset_config.update()
    dataset_config.save_to_file()
    print(f'running experiment for {dataset}...')
    os.system(f'python3 octree.py > log/3d_octree_{dataset}.log')
    print(f'completed experiment for {dataset}...')
