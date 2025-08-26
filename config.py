import os
import json

class query_config:
    # number of query samples
    sample_num = 1000

    @classmethod
    def update(cls, sample_num ):
        cls.sample_num = sample_num

    @classmethod
    def save_to_file(cls, filename='query_config.json'):
        config_data = {
            'sample_num': cls.sample_num
        }
        with open(filename, 'w') as f:
            json.dump(config_data, f)

    @classmethod
    def load_from_file(cls, filename='query_config.json'):
        if not os.path.exists(filename):
            return  # or raise an error, depending on how you'd like to handle this
        with open(filename, 'r') as f:
            data = json.load(f)
            cls.sample_num = data.get('sample_num', cls.sample_num)


class dataset_config:
    dataset = 'gowalla'
    dataset_path = ''
    access_per_level = []

    @classmethod
    def update(cls):
        if cls.dataset == 'cali-1024x1024':
            cls.dataset_path = 'datasets/cali-1024x1024.pickle'
            # cls.access_per_level = [1, 2, 4, 6, 9, 15, 21, 25, 28, 33, 37, 38, 41, 43, 38, 37, 35, 30, 24, 16, 10, 4]
        elif cls.dataset == 'spitz-1024x1024':
            cls.dataset_path = 'datasets/spitz-1024x1024.csv'
            # cls.access_per_level = [1, 2, 4, 6, 10, 18, 24, 28, 33, 39, 41, 39, 42, 39, 35, 30, 25, 20, 12, 7, 4, 2]
        elif cls.dataset == 'gowalla':
            cls.dataset_path = 'datasets/gowalla-1d/5.0m-gowalla.csv'
            cls.access_per_level = [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2]
        elif cls.dataset == 'amazon-books':
            cls.dataset_path = 'datasets/amazon-books.csv'
            cls.access_per_level = [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2]
        elif cls.dataset == 'nh_64':
            cls.dataset_path = 'datasets/nh_64.txt'
            # cls.access_per_level = [1, 2, 4, 6, 9, 13, 17, 25, 36, 50, 62, 77, 88, 97, 96, 93, 90, 69, 43, 20, 8]
        elif cls.dataset == 'gowalla_100k':
            cls.dataset_path = 'datasets/gowalla_2d_100k.pkl'
            # cls.access_per_level = [1, 2, 4, 6, 9, 16, 19, 21, 21, 22, 24, 25, 24, 23, 20, 16, 15, 11, 12, 10, 8, 7, 7, 4, 1, 0, 0, 0]
        elif cls.dataset == 'gowalla_50k':
            cls.dataset_path = 'datasets/gowalla_2d_50k.pkl'
            # cls.access_per_level = [1, 2, 4, 6, 9, 14, 16, 18, 18, 18, 19, 18, 16, 18, 14, 13, 12, 9, 6, 3, 2, 2, 1, 0, 0]
        elif cls.dataset == 'gowalla_3d_23k':
            cls.dataset_path = 'datasets/gowalla_3d_23k.pkl'
            # cls.access_per_level = [1, 2, 4, 6, 10, 16, 24, 40, 62, 84, 92, 85, 75, 77, 66, 55, 37, 28, 18, 9, 4, 2, 1]
        elif cls.dataset == 'synthetic_2d_1m':
            cls.dataset_path = 'datasets/synthetic_2d_1m.pkl'
            # cls.access_per_level = [1, 2, 4, 6, 8, 11, 13, 16, 22, 26, 27, 31, 32, 36, 36, 39, 31, 30, 28, 25, 19, 14, 7, 0]
        elif cls.dataset == 'synthetic_2d_1m_sparse':
            cls.dataset_path = 'datasets/synthetic_2d_1m_sparse.pkl'
            # cls.access_per_level = [1, 2, 4, 6, 9, 16, 19, 25, 29, 32, 38, 42, 45, 46, 46, 46, 40, 36, 29, 25, 20, 13, 7, 3]
        elif cls.dataset == 'synthetic_2d_1m-1024x1024':
            cls.dataset_path = 'datasets/synthetic_2d_1m-1024x1024.pkl'
            # cls.access_per_level = [1, 2, 4, 6, 9, 16, 20, 27, 33, 36, 36, 39, 42, 41, 38, 34, 32, 28, 24, 20, 12, 4]
        elif cls.dataset == 'synthetic_3d_1m_sparse':
            cls.dataset_path = 'datasets/synthetic_3d_1m_sparse.pkl'
            cls.access_per_level = []
        elif cls.dataset == 'synthetic_3d_1m_128':
            cls.dataset_path = 'datasets/synthetic_3d_1m_128.pkl'
            # cls.access_per_level = [1, 2, 4, 6, 9, 14, 21, 32, 50, 68, 94, 120, 144, 166, 184, 181, 168, 178, 164, 138, 97, 51, 20, 6]
        elif cls.dataset == 'synthetic_3d_1m_256':
            cls.dataset_path = 'datasets/synthetic_3d_1m_256.pkl'
            # cls.access_per_level = [1, 2, 4, 6, 9, 15, 21, 32, 50, 72, 101, 125, 150, 173, 192, 207, 222, 220, 200, 181, 147, 103, 57, 24, 11, 4, 2]

    @classmethod
    def save_to_file(cls, filename='config.json'):
        config_data = {
            'dataset': cls.dataset,
            'dataset_path': cls.dataset_path,
            'access_per_level': cls.access_per_level
        }
        with open(filename, 'w') as config_file:
            json.dump(config_data, config_file)

    @classmethod
    def load_from_file(cls, filename='config.json'):
        with open(filename, 'r') as config_file:
            config_data = json.load(config_file)
            cls.dataset = config_data['dataset']
            cls.dataset_path = config_data['dataset_path']
            cls.access_per_level = config_data['access_per_level']

# Ensure the initial configuration is set
# dataset_config.update()
# dataset_config.save_to_file()


