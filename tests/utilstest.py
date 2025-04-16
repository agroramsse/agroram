
# unittests/test_utils.py

import sys
import os

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from termcolor import colored

# write a unit test for count_range_nodes for 1d, 2d, and 3d points
def test_count_range_nodes():
    # 1D test
    points = [1, 2, 3, 4, 5]
    ranges = [(2, 4)]
    assert utils.count_range_nodes(points, *ranges) == 3
    print(colored('1D test pass','green'))

    # 2D test
    points = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    ranges = [(2, 4), (3, 5)]
    assert utils.count_range_nodes(points, *ranges) == 3
    print(colored('2D test pass','green'))

    # 3D test
    points = [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]
    ranges = [(2, 4), (3, 5), (4, 6)]
    assert utils.count_range_nodes(points, *ranges) == 3
    print(colored('3D test pass','green'))
    print(colored('-'*50,'green'))
    
    print(colored('All test cases pass','green'))

# write a unit test for random_range
def test_random_range():
    range_ = (1, 10)
    for _ in range(100):
        r = utils.random_range(range_)
        assert r[0] >= range_[0] and r[1] <= range_[1] and r[0] <= r[1]
    print(colored('random_range test pass','green'))

# write a unit test for generate_random_query
def test_generate_random_query():
    x_range = (1, 10)
    y_range = (1, 10)
    z_range = (1, 10)
    for _ in range(100):
        r = utils.generate_random_query(1, x_range)
        assert r[0] >= x_range[0] and r[0] <= x_range[1]

        r = utils.generate_random_query(2, x_range, y_range)
        assert r[0][0] >= x_range[0] and r[0][0] <= x_range[1]
        assert r[0][1] >= y_range[0] and r[0][1] <= y_range[1]

        r = utils.generate_random_query(3, x_range, y_range, z_range)
        assert r[0][0] >= x_range[0] and r[0][1] <= x_range[1]
        assert r[1][0] >= y_range[0] and r[1][1] <= y_range[1]
        assert r[2][0] >= z_range[0] and r[2][1] <= z_range[1]
    print(colored('generate_random_query test pass','green'))

# write a unit test for load_csv_data
def test_load_csv_data():
    data = utils.load_csv_data('qr_acs_1d.csv')
    # verify all elements are the same size
    for i in range(1, len(data)):
        assert len(data[i]) == len(data[i-1])
    assert len(data) == 1000
    print(colored('load_csv_data test pass','green'))

# write a unit test for write_csv function
def test_write_csv():
    data = [{1:1, 2:0, 3:0}, {1:4, 2:4, 3:3}, {1:0, 2:8, 3:0}]
    utils.write_csv(data, 'test')
    data_ = utils.load_csv_data('reports/test.csv')
    print(data_)
    # verify all values assoiated with the keys in the data are the same as the data_ values
    for i in range(len(data)):
        for j in range(len(data[i])):
            assert data[i][j+1] == data_[i][j]
    # remove the test file created
    os.remove('reports/test.csv')
    print(colored('write_csv test pass','green'))

# write a test for the max_access_per_level_theory function
def test_max_access_per_level_theory():
    # Example usage
    n_values = [1015, 961]  # Replace with your values for n_1, n_2, ..., n_i
    d = 2  # Set d (upper limit for i)
    output = utils.max_access_per_level_theory(n_values, d)
    print("Result:", output)



if __name__ == '__main__':
    # test_count_range_nodes()
    # test_random_range()
    # test_generate_random_query()
    # test_load_csv_data()
    # test_write_csv()
    test_max_access_per_level_theory()