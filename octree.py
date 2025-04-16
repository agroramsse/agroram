import numpy as np
from typing import List, Tuple, Dict, Optional
from treelib import Tree, Node

class Point3D:
    def __init__(self, x: int, y: int, z: int, value: int):
        """
        Represents a 3D point with integer coordinates and value.
        
        :param x: x-coordinate of the point
        :param y: y-coordinate of the point
        :param z: z-coordinate of the point
        :param value: Integer value associated with the point
        """
        self.x = x
        self.y = y
        self.z = z
        self.value = value
    
    def __repr__(self):
        return f"Point3D(x={self.x}, y={self.y}, z={self.z}, value={self.value})"

class OctTreeNode:
    def __init__(self, 
                 x_range: Tuple[int, int],  # (start_idx, end_idx) in sorted x array
                 y_range: Tuple[int, int],  # (start_idx, end_idx) in sorted y array
                 z_range: Tuple[int, int],  # (start_idx, end_idx) in sorted z array
                 depth: int = 0,
                 is_leaf: bool = True):
        """
        Data container for an OctTree node.
        
        :param x_range: (start_idx, end_idx) defining x-range indices
        :param y_range: (start_idx, end_idx) defining y-range indices
        :param z_range: (start_idx, end_idx) defining z-range indices
        :param depth: Current depth of the node in the tree
        :param is_leaf: Whether this node is a leaf node
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.depth = depth
        self.is_leaf = is_leaf
        self.points = []  # Will store points if this is a leaf node
        self.aggregate_value = 0  # Will store the sum of all point values in this node and its children
        
    def __repr__(self):
        return f"OctTreeNode(x_range={self.x_range}, y_range={self.y_range}, z_range={self.z_range}, depth={self.depth}, points={len(self.points)}, agg_value={self.aggregate_value})"

class IndexBasedHyperOctTree:
    def __init__(self, 
                 max_points_per_node: int = 1,
                 max_depth: int = 100):
        """
        Initialize an empty OctTree with a specified capacity and depth.
        
        :param max_points_per_node: Maximum number of points in a leaf node before partitioning
        :param max_depth: Maximum depth of the tree
        """
        self.tree = Tree()
        self.max_points = max_points_per_node
        self.max_depth = max_depth
        self.all_points = []  # Keep a reference to all points
        self.sorted_x_values = []  # Sorted unique x values
        self.sorted_y_values = []  # Sorted unique y values
        self.sorted_z_values = []  # Sorted unique z values
        
    def build_from_points(self, points: List[Point3D]) -> None:
        """
        Build an OctTree using index-based partitioning from a list of 3D points.
        
        :param points: List of Point3D objects to build the tree from
        """
        if not points:
            return
            
        # Store all points for verification
        self.all_points = points.copy()
            
        # Extract and sort unique x, y, and z values
        x_values = list(set(p.x for p in points))
        y_values = list(set(p.y for p in points))
        z_values = list(set(p.z for p in points))
        
        self.sorted_x_values = sorted(x_values)
        self.sorted_y_values = sorted(y_values)
        self.sorted_z_values = sorted(z_values)
        
        # Create dictionaries to quickly map from value to index
        self.x_to_idx = {val: idx for idx, val in enumerate(self.sorted_x_values)}
        self.y_to_idx = {val: idx for idx, val in enumerate(self.sorted_y_values)}
        self.z_to_idx = {val: idx for idx, val in enumerate(self.sorted_z_values)}
        
        # Root node covers the entire index range
        x_range = (0, len(self.sorted_x_values) - 1)
        y_range = (0, len(self.sorted_y_values) - 1)
        z_range = (0, len(self.sorted_z_values) - 1)
        
        # Create root node
        root_node = OctTreeNode(x_range, y_range, z_range)
        self.tree.create_node("root", "root", data=root_node)
        
        # Recursively build the tree
        self._build_recursive("root", points, 0)
        
        # Update aggregate values bottom-up
        self._update_aggregate_values("root")
    
    def _build_recursive(self, node_id: str, points: List[Point3D], depth: int) -> None:
        """
        Recursively build the OctTree by partitioning the data based on indices.
        
        :param node_id: ID of the current node
        :param points: List of points to partition
        :param depth: Current depth in the tree
        """
        node = self.tree.get_node(node_id)
        node_data = node.data
        node_data.depth = depth
        
        # Convert point coordinates to indices
        for p in points:
            p.x_idx = self.x_to_idx[p.x]
            p.y_idx = self.y_to_idx[p.y]
            p.z_idx = self.z_to_idx[p.z]
        
        # Filter points that are within this node's index ranges
        node_points = [p for p in points if self._contains_point_by_index(
            node_data.x_range, node_data.y_range, node_data.z_range, p)]
        
        # If few enough points or max depth reached, make it a leaf node
        if len(node_points) <= self.max_points or depth >= self.max_depth:
            node_data.is_leaf = True
            node_data.points = node_points
            return
        
        # Otherwise, partition the space into eight octants based on indices
        node_data.is_leaf = False
        x_start, x_end = node_data.x_range
        y_start, y_end = node_data.y_range
        z_start, z_end = node_data.z_range
        
        # Calculate mid-indices
        x_mid = (x_start + x_end) // 2
        y_mid = (y_start + y_end) // 2
        z_mid = (z_start + z_end) // 2
        
        # Define octants based on indices
        # Each octant is defined by (x_range, y_range, z_range)
        octant_ranges = [
            # 0: Bottom-Southwest (x_start to x_mid, y_start to y_mid, z_start to z_mid)
            ((x_start, x_mid), (y_start, y_mid), (z_start, z_mid)),
            # 1: Bottom-Southeast (x_mid+1 to x_end, y_start to y_mid, z_start to z_mid)
            ((x_mid + 1, x_end), (y_start, y_mid), (z_start, z_mid)),
            # 2: Bottom-Northwest (x_start to x_mid, y_mid+1 to y_end, z_start to z_mid)
            ((x_start, x_mid), (y_mid + 1, y_end), (z_start, z_mid)),
            # 3: Bottom-Northeast (x_mid+1 to x_end, y_mid+1 to y_end, z_start to z_mid)
            ((x_mid + 1, x_end), (y_mid + 1, y_end), (z_start, z_mid)),
            # 4: Top-Southwest (x_start to x_mid, y_start to y_mid, z_mid+1 to z_end)
            ((x_start, x_mid), (y_start, y_mid), (z_mid + 1, z_end)),
            # 5: Top-Southeast (x_mid+1 to x_end, y_start to y_mid, z_mid+1 to z_end)
            ((x_mid + 1, x_end), (y_start, y_mid), (z_mid + 1, z_end)),
            # 6: Top-Northwest (x_start to x_mid, y_mid+1 to y_end, z_mid+1 to z_end)
            ((x_start, x_mid), (y_mid + 1, y_end), (z_mid + 1, z_end)),
            # 7: Top-Northeast (x_mid+1 to x_end, y_mid+1 to y_end, z_mid+1 to z_end)
            ((x_mid + 1, x_end), (y_mid + 1, y_end), (z_mid + 1, z_end))
        ]
        
        octant_names = ['bsw', 'bse', 'bnw', 'bne', 'tsw', 'tse', 'tnw', 'tne']
        
        # Create child nodes only if the index range is valid
        for i, ((x_range, y_range, z_range), name) in enumerate(zip(octant_ranges, octant_names)):
            # Skip if the index range is invalid
            if x_range[0] > x_range[1] or y_range[0] > y_range[1] or z_range[0] > z_range[1]:
                continue
                
            child_id = f"{node_id}_{name}"
            child_node = OctTreeNode(x_range, y_range, z_range, depth + 1)
            self.tree.create_node(name, child_id, parent=node_id, data=child_node)
            
            # Recursively build this octant
            self._build_recursive(child_id, node_points, depth + 1)
    
    def _update_aggregate_values(self, node_id: str) -> int:
        """
        Update aggregate values bottom-up.
        For leaf nodes, sum the values of contained points.
        For internal nodes, sum the aggregate values of children.
        
        :param node_id: ID of the current node
        :return: The aggregate value for this node
        """
        node = self.tree.get_node(node_id)
        node_data = node.data
        
        if node_data.is_leaf:
            # Leaf node: sum point values
            node_data.aggregate_value = sum(p.value for p in node_data.points)
        else:
            # Internal node: sum child values
            children = self.tree.children(node_id)
            if not children:  # If no children, treat as a leaf
                node_data.is_leaf = True
                node_data.aggregate_value = 0  # No points
            else:
                child_values = [self._update_aggregate_values(child.identifier) for child in children]
                node_data.aggregate_value = sum(child_values)
        
        return node_data.aggregate_value
    
    def _contains_point_by_index(self, x_range: Tuple[int, int], y_range: Tuple[int, int], 
                                z_range: Tuple[int, int], point) -> bool:
        """
        Check if a point's indices are within the given index ranges.
        
        :param x_range: (start_idx, end_idx) for x indices
        :param y_range: (start_idx, end_idx) for y indices
        :param z_range: (start_idx, end_idx) for z indices
        :param point: Point with x_idx, y_idx and z_idx attributes
        :return: True if the point is within bounds
        """
        x_start, x_end = x_range
        y_start, y_end = y_range
        z_start, z_end = z_range
        return (x_start <= point.x_idx <= x_end and 
                y_start <= point.y_idx <= y_end and 
                z_start <= point.z_idx <= z_end)
    
    def query_range_sum(self, query_bounds: Tuple[int, int, int, int, int, int]) -> int:
        """
        Compute sum of point values within a given 3D range efficiently using precomputed aggregates.
        
        :param query_bounds: (min_x, min_y, min_z, max_x, max_y, max_z) defining query region in original coordinates
        :return: Sum of point values within the range
        """
        # Convert query bounds to index ranges
        qmin_x, qmin_y, qmin_z, qmax_x, qmax_y, qmax_z = query_bounds
        
        # Find the closest indices in our sorted arrays
        x_min_idx = self._find_closest_index(self.sorted_x_values, qmin_x, lower=True)
        x_max_idx = self._find_closest_index(self.sorted_x_values, qmax_x, lower=False)
        y_min_idx = self._find_closest_index(self.sorted_y_values, qmin_y, lower=True)
        y_max_idx = self._find_closest_index(self.sorted_y_values, qmax_y, lower=False)
        z_min_idx = self._find_closest_index(self.sorted_z_values, qmin_z, lower=True)
        z_max_idx = self._find_closest_index(self.sorted_z_values, qmax_z, lower=False)
        
        # Perform the query using indices
        total_sum = self._query_range_recursive("root", (x_min_idx, y_min_idx, z_min_idx, 
                                                       x_max_idx, y_max_idx, z_max_idx))
        return total_sum
    
    def _find_closest_index(self, sorted_values: List[int], target: int, lower: bool = True) -> int:
        """
        Find the index of the closest value in the sorted array.
        If lower=True, find the largest value <= target.
        If lower=False, find the smallest value >= target.
        
        :param sorted_values: Sorted list of values
        :param target: Target value to find closest to
        :param lower: If True, find lower bound, else find upper bound
        :return: Index of the closest value
        """
        if not sorted_values:
            return -1
            
        # If target is outside the range, return boundary index
        if target <= sorted_values[0]:
            return 0
        if target >= sorted_values[-1]:
            return len(sorted_values) - 1
            
        # Binary search to find the closest index
        left, right = 0, len(sorted_values) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if sorted_values[mid] == target:
                return mid
            elif sorted_values[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        # At this point, left > right
        # If looking for lower bound, return right (largest value <= target)
        # If looking for upper bound, return left (smallest value >= target)
        return right if lower else left
    
    def _query_range_recursive(self, node_id: str, query_idx_bounds: Tuple[int, int, int, int, int, int]) -> int:
        """
        Recursively find the sum of all point values within the query range.
        
        :param node_id: ID of the current node
        :param query_idx_bounds: (x_min_idx, y_min_idx, z_min_idx, x_max_idx, y_max_idx, z_max_idx) 
                                defining query region in indices
        :return: Sum of point values within the range
        """
        node = self.tree.get_node(node_id)
        if not node:
            return 0
            
        node_data = node.data
        
        # Check if this node intersects with the query range
        if not self._intersects_range_by_index(node_data.x_range, node_data.y_range, 
                                             node_data.z_range, query_idx_bounds):
            return 0
        
        # If query range completely contains this node, use the precomputed aggregate
        if self._contains_range_by_index(query_idx_bounds, node_data.x_range, 
                                       node_data.y_range, node_data.z_range):
            return node_data.aggregate_value
        
        # If leaf node, sum the points that fall within the query range
        if node_data.is_leaf:
            x_min_idx, y_min_idx, z_min_idx, x_max_idx, y_max_idx, z_max_idx = query_idx_bounds
            points_in_range = [p for p in node_data.points if 
                         x_min_idx <= p.x_idx <= x_max_idx and 
                         y_min_idx <= p.y_idx <= y_max_idx and
                         z_min_idx <= p.z_idx <= z_max_idx]
            return sum(p.value for p in points_in_range)
        
        # Otherwise, recurse on children and sum the results
        total_sum = 0
        for child in self.tree.children(node_id):
            total_sum += self._query_range_recursive(child.identifier, query_idx_bounds)
        
        return total_sum
    
    def _intersects_range_by_index(self, x_range: Tuple[int, int], y_range: Tuple[int, int], 
                                 z_range: Tuple[int, int],
                                 query_idx_bounds: Tuple[int, int, int, int, int, int]) -> bool:
        """
        Check if node's index ranges intersect with the query index ranges.
        
        :param x_range: (start_idx, end_idx) of the node's x range
        :param y_range: (start_idx, end_idx) of the node's y range
        :param z_range: (start_idx, end_idx) of the node's z range
        :param query_idx_bounds: (x_min_idx, y_min_idx, z_min_idx, x_max_idx, y_max_idx, z_max_idx) of the query
        :return: True if there is an intersection
        """
        x_min_idx, y_min_idx, z_min_idx, x_max_idx, y_max_idx, z_max_idx = query_idx_bounds
        x_start, x_end = x_range
        y_start, y_end = y_range
        z_start, z_end = z_range
        
        return not (x_max_idx < x_start or x_min_idx > x_end or 
                  y_max_idx < y_start or y_min_idx > y_end or
                  z_max_idx < z_start or z_min_idx > z_end)
    
    def _contains_range_by_index(self, outer: Tuple[int, int, int, int, int, int], 
                               inner_x: Tuple[int, int], inner_y: Tuple[int, int], 
                               inner_z: Tuple[int, int]) -> bool:
        """
        Check if outer index ranges completely contain inner index ranges.
        
        :param outer: (x_min_idx, y_min_idx, z_min_idx, x_max_idx, y_max_idx, z_max_idx) of the outer range
        :param inner_x: (start_idx, end_idx) of the inner x range
        :param inner_y: (start_idx, end_idx) of the inner y range
        :param inner_z: (start_idx, end_idx) of the inner z range
        :return: True if inner ranges are completely contained in outer ranges
        """
        x_min_idx, y_min_idx, z_min_idx, x_max_idx, y_max_idx, z_max_idx = outer
        x_start, x_end = inner_x
        y_start, y_end = inner_y
        z_start, z_end = inner_z
        
        return (x_min_idx <= x_start and x_max_idx >= x_end and
                y_min_idx <= y_start and y_max_idx >= y_end and
                z_min_idx <= z_start and z_max_idx >= z_end)
    
    def height(self) -> int:
        """
        Calculate the height of the octree.
        
        :return: Maximum depth of the tree
        """
        max_depth = 0
        for node in self.tree.all_nodes():
            max_depth = max(max_depth, node.data.depth)
        return max_depth
    
    def count_total_nodes(self) -> int:
        """
        Count the total number of nodes in the octree.
        
        :return: Total number of nodes
        """
        return len(self.tree.all_nodes())
    
    def is_balanced(self) -> bool:
        """
        Check if the octree is balanced.
        A balanced octree has no significant height difference 
        between its octants.
        
        :return: True if the tree is balanced, False otherwise
        """
        # Get depth of all leaf nodes
        leaf_depths = []
        for node in self.tree.all_nodes():
            if node.data.is_leaf:
                leaf_depths.append(node.data.depth)
        
        if not leaf_depths:
            return True
        
        # Tree is balanced if max and min leaf depths differ by at most 1
        return max(leaf_depths) - min(leaf_depths) <= 1

    def get_node_count_by_depth(self) -> Dict[int, int]:
        """
        Count nodes at each depth level.
        
        :return: Dictionary mapping depth to node count
        """
        depth_counts = {}
        for node in self.tree.all_nodes():
            depth = node.data.depth
            if depth in depth_counts:
                depth_counts[depth] += 1
            else:
                depth_counts[depth] = 1
        return depth_counts

    def verify_aggregate_values(self) -> bool:
        """
        Verify that aggregate values are correctly computed by checking all nodes.
        
        :return: True if all aggregate values are correct
        """
        all_correct = True
        for node in self.tree.all_nodes():
            node_id = node.identifier
            node_data = node.data
            x_range = node_data.x_range
            y_range = node_data.y_range
            z_range = node_data.z_range
            
            # Get actual min/max values from indices
            x_min = self.sorted_x_values[x_range[0]] if x_range[0] < len(self.sorted_x_values) else float('inf')
            x_max = self.sorted_x_values[x_range[1]] if x_range[1] < len(self.sorted_x_values) else float('-inf')
            y_min = self.sorted_y_values[y_range[0]] if y_range[0] < len(self.sorted_y_values) else float('inf')
            y_max = self.sorted_y_values[y_range[1]] if y_range[1] < len(self.sorted_y_values) else float('-inf')
            z_min = self.sorted_z_values[z_range[0]] if z_range[0] < len(self.sorted_z_values) else float('inf')
            z_max = self.sorted_z_values[z_range[1]] if z_range[1] < len(self.sorted_z_values) else float('-inf')
            
            # Calculate expected aggregate value directly from all points
            expected_points_in_bounds = [p for p in self.all_points if 
                                       x_min <= p.x <= x_max and 
                                       y_min <= p.y <= y_max and
                                       z_min <= p.z <= z_max]
            expected_sum = sum(p.value for p in expected_points_in_bounds)
            
            # For leaf nodes, directly compare with points
            if node_data.is_leaf:
                leaf_sum = sum(p.value for p in node_data.points)
                if node_data.aggregate_value != leaf_sum:
                    print(f"Leaf node {node_id} has incorrect aggregate: " 
                          f"Expected {leaf_sum}, got {node_data.aggregate_value}")
                    all_correct = False
            
            # For internal nodes, verify recursive sum from children matches aggregate
            if not node_data.is_leaf:
                children = self.tree.children(node_id)
                if children:  # Only check if there are children
                    child_sum = sum(child.data.aggregate_value for child in children)
                    if node_data.aggregate_value != child_sum:
                        print(f"Internal node {node_id} has incorrect aggregate: "
                              f"Expected {child_sum}, got {node_data.aggregate_value}")
                        all_correct = False
        
        return all_correct


# Example usage
def main():
    # Load dataset
    from utils import load_dataset_3d
    from config import query_config, dataset_config
    from termcolor import colored

    # Load the configuration from the file
    dataset_config.load_from_file()

    print(colored('-'*100, 'green'))
    #---------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------
    # *loading 3D dataset


    dataset_path = dataset_config.dataset_path
    print(colored(f'Loading 3D dataset from {dataset_path}...', 'blue'))
    # read the dataset_path extension and store as string in a type_ variable
    type_ = dataset_path.split('.')[-1]
    # from the dataset_path, get the dataset name
    dataset_name = dataset_path.split('/')[-1].split('.')[0]

    print(colored(f'\033[1m{dataset_name} 3d dataset\033[0m', 'blue'))

    pts, pts_dict, x_range, y_range, z_range = load_dataset_3d(dataset_path, type_=type_)

    # Create points from the loaded dataset
    points_from_dataset = []
    for (x, y, z), value in pts_dict.items():
        points_from_dataset.append(Point3D(x, y, z, value))
    
    # Print dataset statistics
    unique_x = len(set(p.x for p in points_from_dataset))
    unique_y = len(set(p.y for p in points_from_dataset))
    unique_z = len(set(p.z for p in points_from_dataset))
    print(f"unique x values: {unique_x}")
    print(f"unique y values: {unique_y}")
    print(f"unique z values: {unique_z}")
    print(f"size of the dataset: {len(points_from_dataset)}")
    print(f"x bound: ({min(p.x for p in points_from_dataset)}, {max(p.x for p in points_from_dataset)})")
    print(f"y bound: ({min(p.y for p in points_from_dataset)}, {max(p.y for p in points_from_dataset)})")
    print(f"z bound: ({min(p.z for p in points_from_dataset)}, {max(p.z for p in points_from_dataset)})")

    # Create the index-based octree
    oct_tree = IndexBasedHyperOctTree(max_points_per_node=1, max_depth=100)
    oct_tree.build_from_points(points_from_dataset)

    # Print tree stats
    print(f'Index-Based HyperOctTree Stats for Dataset: {dataset_name}')
    print('Number of nodes:', oct_tree.count_total_nodes())
    print('Height of the tree:', oct_tree.height())
    print('Is the tree balanced:', oct_tree.is_balanced())
    
    # Distribution of nodes by depth
    depth_counts = oct_tree.get_node_count_by_depth()
    print('Node count by depth:')
    for depth, count in sorted(depth_counts.items()):
        print(f'  Depth {depth}: {count} nodes')

    # # Verify aggregate values
    # print('Verifying aggregate values...')
    # print('All aggregates correct:', oct_tree.verify_aggregate_values())

    # # Perform range sum queries to test
    # query_ranges = [
    #     (0, 0, 0, 64, 64, 64),   # Covering the entire dataset
    #     (0, 0, 0, 30, 40, 40),    # Partial range with specific points
    #     (0, 0, 0, 20, 30, 64)   # Another partial range
    # ]

    # for qrange in query_ranges:
    #     # Get sum using octree
    #     octree_sum = oct_tree.query_range_sum(qrange)
        
    #     # Verify with naive approach (for validation)
    #     naive_sum = sum(p.value for p in points_from_dataset if 
    #                 qrange[0] <= p.x <= qrange[3] and 
    #                 qrange[1] <= p.y <= qrange[4] and
    #                 qrange[2] <= p.z <= qrange[5])
        
    #     print(f"Dataset Range {qrange}:")
    #     print(f"  OctTree Sum: {octree_sum}")
    #     print(f"  Naive Sum:   {naive_sum}")
    #     print(f"  Match:       {octree_sum == naive_sum}\n")

if __name__ == "__main__":
    main()