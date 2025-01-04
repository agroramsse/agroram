# Privacy Preserving Aggregate Range Queries via Oblivious Multidimensional Segment Trees
Implementation of Tree-Based Oblivious Map from Wang et al.
Implementation of multidimensional segment tree for aggregation functions.

In order to run the code, you must have Python 3.10.13 or newer versions installed. <br />
The `req.txt` file located in the main directory lists all the required dependencies for executing the project. You can install these dependencies by using the command `pip install -r req.txt`.


In the `config.py` you can set the number of query samples to run the experiments, and the dataset name.

To run the 1d experiments run the `python3 experiment/query_1d.py`.

To run the 2d experiments run the `python3 experiment/query_2d.py`.

To run the 3d experiments run the `python3 experiment/query_3d.py`.

Then in order to output the results analysis of stopping height, and node accesses per level, and figure6 data  run `python3 experiment/results.py --dim [dim]`, the dim is the dimenion of the dataset you have specified in the config.py.

In order to generate the table6 reports, run the `python3 allowedAccessPatterns.py`.
