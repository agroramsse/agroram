# Privacy Preserving Aggregate Range Queries with Practical Storage
This project is a Python-based project for running encrypted aggregation query experiments on different 1D, 2D, and 3D datasets. The project includes the follwoing implementations:  

- Implementation of Tree-Based Oblivious Map from Wang et al.
- Implementation of multidimensional Segment Tree for aggregation functions.
- Implementation of multidimensional Oblivious Segment Tree.
- Implementation of quadtree and octree for height measurements on the datasets.

In order to run the code, you must have Python 3.10.13 or newer versions installed. <br />
The `req.txt` file located in the main directory lists all the required dependencies for executing the project. You can install these dependencies by using the command `pip install -r req.txt`.


## Usage
From the project root directory, run:
```bash
python crypto_exp.py [--q_num Q_NUM] [--dataset DATASET_NAME] 
```
**Usage Flags**

- **`--q_num Q_NUM`** *(Optional)*  
  Number of query samples. Defaults to **1000** if not specified.

- **`--dataset DATASET_NAME`** *(Optional)*  
  Dataset to run experiments on. Defaults to **all** (runs experiments for all 1D, 2D, and 3D datasets).

## Available Datasets

### 1D Datasets
- `amazon-books`
- `gowalla`

### 2D Datasets
- `spitz-1024x1024`
- `cali-1024x1024`
- `gowalla_100k`
- `gowalla_50k`
- `synthetic_2d_1m`
- `synthetic_2d_1m_sparse`
- `synthetic_2d_1m-1024x1024`

### 3D Datasets
- `nh_64`
- `gowalla_3d_23k`
- `synthetic_3d_1m_128`
- `synthetic_3d_1m_256`

## Examples

1. **Run experiments with default settings** (1000 queries, all datasets):
```bash
   python crypto_exp.py
```

2. **Specify a custom number of queries** (e.g., 500):
```bash
  python crypto_exp.py --q_num 500
```
3. **Run a single dataset** (e.g., `spitz-1024x1024`):
```bash
python crypto_exp.py --dataset spitz-1024x1024
```

4. **Combine both** (e.g., 2000 queries with `amazon-books`):
```bash
python crypto_exp.py --q_num 2000 --dataset amazon-books
```

## Logs

Logs are automatically generated in the `log/` folder:

- **1D datasets**: `log/1d_{dataset}_crypto.log`
- **2D datasets**: `log/2d_{dataset}_crypto.log`
- **3D datasets**: `log/3d_{dataset}_crypto.log`

## Regenerate Table results in the paper

To generate the **Table 8** and **Table 9** data in the paper, run:

```bash
python /log/report_log.py



