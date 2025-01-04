import argparse
class query_config:
    # number of query samples
    sample_num = 500

class dataset_config:
        
    dataset ='cali-1024x1024'

    if dataset == 'cali-1024x1024':
        dataset_path = 'datasets/cali-1024x1024.pickle'
        access_per_level = [1, 2, 4, 6, 9, 15, 21, 25, 28, 33, 37, 38, 41, 43, 38, 37, 35, 30, 24, 16, 10, 4]

    elif dataset == 'spitz-1024x1024':
        dataset_path = 'datasets/spitz-1024x1024.csv'
        access_per_level = [1, 2, 4, 6, 10, 18, 24, 28, 33, 39, 41, 39, 42, 39, 35, 30, 25, 20, 12, 7, 4, 2]

    elif dataset == 'gowalla':
        dataset_path = 'datasets/gowalla-1d/5.0m-gowalla.csv'
        access_per_level = [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2]

    elif dataset == 'amazon-books':
        dataset_path = 'datasets/amazon-books.csv'
        access_per_level = [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2]

    elif dataset == 'nh_64':
        dataset_path = 'datasets/nh_64.txt'
        access_per_level = [1, 2, 4, 6, 9, 13, 17, 25, 36, 50, 62, 77, 88, 97, 96, 93, 90, 69, 43, 20, 8]
    
    elif dataset =='gowalla_100k':
        dataset_path = 'datasets/gowalla_2d_100k.pkl'
        access_per_level = [1,2,4,6,9,16,19,21,21,22,24,25,24,23,20,16,15,11,12,10,8,7,7,4,1,0,0,0]

    elif dataset =='gowalla_50k':
        dataset_path = 'datasets/gowalla_2d_50k.pkl'
        access_per_level = [1,2,4,6,9,14,16,18,18,18,19,18,16,18,14,13,12,9,6,3,2,2,1,0,0]

    elif dataset == 'gowalla_3d_23k':
        dataset_path = 'datasets/gowalla_3d_23k.pkl'
        access_per_level = [1,2,4,6,10,16,24,40,62,84,92,85,75,77,66,55,37,28,18,9,4,2,1]
    
    # dense
    elif dataset == 'synthetic_2d_1m':
        dataset_path = 'datasets/synthetic_2d_1m.pkl'
        access_per_level = [1,2,4,6,8,11,13,16,22,26,27,31,32,36,36,39,31,30,28,25,19,14,7,0]
    
    # spase 2048x2048
    elif dataset == 'synthetic_2d_1m_sparse':
        dataset_path = 'datasets/synthetic_2d_1m_sparse.pkl'
        access_per_level = [1,2,4,6,9,16,19,25,29,32,38,42,45,46,46,46,40,36,29,25,20,13,7,3]
        
    # sparse 1024x1024
    elif dataset == 'synthetic_2d_1m-1024x1024':
        dataset_path = 'datasets/synthetic_2d_1m-1024x1024.pkl'
        access_per_level = [1,2,4,6,9,16,20,27,33,36,36,39,42,41,38,34,32,28,24,20,12,4]

    # sparse 1024x1024x1024
    elif dataset == 'synthetic_3d_1m_sparse':
        dataset_path = 'datasets/synthetic_3d_1m_sparse.pkl'
        access_per_level = []

    # sparse 3d 128x128x128
    elif dataset == 'synthetic_3d_1m_128':
        dataset_path = 'datasets/synthetic_3d_1m_128.pkl'
        access_per_level = [1, 2, 4, 6, 9, 14, 21, 32, 50, 68, 94, 120, 144, 166, 184, 181, 168, 178, 164, 138, 97, 51, 20, 6]
    
    # sparse 256x256x256
    elif dataset == 'synthetic_3d_1m_256':
        dataset_path = 'datasets/synthetic_3d_1m_256.pkl'
        access_per_level = [1,2,4,6,9,15,21,32,50,72,101,125,150,173,192,207,222,220,200,181,147,103,57,24,11,4,2]
