# Privacy Preserving Aggregate Range Queries with Practical Storage
This project is a Python-based project for running encrypted aggregation query experiments on different 1D, 2D, and 3D datasets. The project includes the follwoing implementations:  

- Implementation of Tree-Based Oblivious Map from Wang et al.
- Implementation of multidimensional Segment Tree for aggregation functions.
- Implementation of multidimensional Oblivious Segment Tree.
- Implementation of quadtree and octree for height measurements on the datasets.


## Table of Contents

<!-- - [Overview](#overview) -->
- [Prerequisites](#prerequisites)
- [Installation](#installation)<!-- - [Project Structure](#project-structure) -->
- [Usage](#usage)
  - [Available Datasets](#available-datasets)
  - [Examples](#examples)
  - [Logs](#logs)

---


---

## Prerequisites

- In order to run the code, you must have Python 3.10.13 or newer versions installed. <br />
- Recommended: A virtual environment (e.g., `venv` or `conda`) to isolate dependencies.
- The `req.txt` file located in the main directory lists all the required dependencies for executing the project. You can install these dependencies by using the command `pip install -r req.txt`.


## Installation
1. **Clone the repository**:
```bash
   git clone https://github.com/your-username/your-project.git
```

2. **Navigate to the project directory**:
```bash
cd your-project
```

3. **(Optional) Create and activate a virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

4. **Install dependencies (if you have req.txt)**:
```bash
pip install -r req.txt
```

## Usage
To run the cryptogrpahic experiments on the datasets, from the project root directory, run:
```bash
python crypto_exp.py [-q Q_NUM] [-d DATASET_NAME] 
```
**Usage Flags**

- **`-q Q_NUM`** *(Optional)*  
  Number of query samples. Defaults to **1000** if not specified.

- **`-d DATASET_NAME`** *(Optional)*  
  Dataset to run experiments on. Defaults to **all** (runs experiments for all 1D, 2D, and 3D datasets). You can choose the dataset name from the follwoing available datasets. 

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
  python crypto_exp.py -q 500
```
3. **Run a single dataset** (e.g., `spitz-1024x1024`):
```bash
python crypto_exp.py -d spitz-1024x1024
```

4. **Combine both** (e.g., 2000 queries with `amazon-books`):
```bash
python crypto_exp.py -q 2000 -d amazon-books
```

## Logs

Logs are automatically generated in the `log/` folder:

- **1D datasets**: `log/1d_{dataset}_crypto.log`
- **2D datasets**: `log/2d_{dataset}_crypto.log`
- **3D datasets**: `log/3d_{dataset}_crypto.log`

