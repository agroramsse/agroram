# Privacy Preserving Efficient Aggregate Range Queries on Encrypted Multidimensional Databases 
Implementation of Tree-Based Oblivious Map from Wang et al.
Implementation of multidimensional segment tree for aggregation functions.

In order to run the code, you must have Python 3.10.13 or newer versions installed. You can find the installation source [here][def].<br />
The `req.txt` file located in the main directory lists all the required dependencies for executing the project. You can install these dependencies by using the command `pip install -r req.txt`.


In the `config.py` you can set the number of query samples to run the experiments.

To run the 1d experiments run the `python3.10 experiment/query_1d.py` for the amazon_book 1d dataset.

To run the 2d experiments run the `python3.10 experiment/query_2d.py` for the spitz 2d dataset.

To run the 3d experiments run the `python3.10 experiment/query_3d.py` for nh 3d dataset.

Then in order to output the results analysis of stopping height, and node accesses per level run `python3.10 experiment/results.py`.


In order to run all the experiments on all datasets, and output the results analysis, run `runexp.sh`.
