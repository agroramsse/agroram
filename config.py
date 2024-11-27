
class query_config:
    # number of query samples
    sample_num = 10
class dataset_config:
        
    dataset ='gowalla2d100'

    if dataset == 'cali':
        dataset_path = 'datasets/cali-1024x1024.pickle'
        access_per_level = [1, 2, 4, 6, 9, 15, 21, 25, 28, 33, 37, 38, 41, 43, 38, 37, 35, 30, 24, 16, 10, 4]

    elif dataset == 'spitz':
        dataset_path = 'datasets/spitz-1024x1024.csv'
        access_per_level = [1, 2, 4, 6, 10, 18, 24, 28, 33, 39, 41, 39, 42, 39, 35, 30, 25, 20, 12, 7, 4, 2]

    elif dataset == 'gowalla1d':
        dataset_path = 'datasets/gowalla-1d/5.0m-gowalla.csv'
        access_per_level = [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2]

    elif dataset == 'amazon':
        dataset_path = 'datasets/amazon-books.csv'
        access_per_level = [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2]

    elif dataset == 'nh':
        dataset_path = 'datasets/nh_64.txt'
        access_per_level = [1, 2, 4, 6, 9, 13, 17, 25, 36, 50, 62, 77, 88, 97, 96, 93, 90, 69, 43, 20, 8]
    
    elif dataset =='gowalla2d100':
        dataset_path = 'datasets/gowalla_2d_100k.pkl'
        access_per_level = [1,2,4,6,9,16,19,21,21,22,24,25,24,23,20,16,15,11,12,10,8,7,7,4,1,0,0,0]

    elif dataset =='gowalla2d50':
        dataset_path = 'datasets/gowalla_2d_50k.pkl'
        access_per_level = [1,2,4,6,9,14,16,18,18,18,19,18,16,18,14,13,12,9,6,3,2,2,1,0,0]

    elif dataset == 'gowalla3d':
        dataset_path = 'datasets/gowalla_3d_23k.pkl'
        access_per_level = []
    
    elif dataset == 'synthetic_2d_1m':
        dataset_path = 'datasets/synthetic_2d_1mil.pkl'
        access_per_level = [1,2,4,6,8,11,13,16,22,26,27,31,32,36,36,39,31,30,28,25,19,14,7,0]


