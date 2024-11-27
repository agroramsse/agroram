from treelib import Node, Tree
from collections import defaultdict
import math



class SegmentTree1D:
    def __init__(self, data):
        self.data = data
        self.indices = sorted(data.keys())
        self.n = len(self.indices)
        self.tree = Tree()
        self.build(data, 0, 0, self.n)
        self.qr_result = [] # query result nodes
        self.qr_h = {} # query result nodes stopping height
        self.access = {}  # accessed nodes in the tree per level when running a query
        self.accessed_nodes={}

    def build(self, data, node_id, start, length):
        if length == 1:
            index = self.indices[start]
            if node_id == 0:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=data[index])
            else:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=data[index], parent=(node_id - 1) // 2)
        else:
            mid = length // 2
            left_child = 2 * node_id + 1
            right_child = 2 * node_id + 2

            if node_id == 0:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=0)
            else:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=0, parent=(node_id - 1) // 2)

            self.build(data, left_child, start, mid)
            self.build(data, right_child, start + mid, length - mid)

            left_data = self.tree[left_child].data
            right_data = self.tree[right_child].data
            self.tree[node_id].data = left_data + right_data

    def update(self, idx, value, node_id=0, start=0, length=None):
        if length is None:
            length = self.n

        if length == 1:
            self.tree[node_id].data = value
            self.data[idx] = value
        else:
            mid = length // 2
            left_child = 2 * node_id + 1
            right_child = 2 * node_id + 2

            if start <= self.indices.index(idx) < start + mid:
                self.update(idx, value, left_child, start, mid)
            else:
                self.update(idx, value, right_child, start + mid, length - mid)

            left_data = self.tree[left_child].data
            right_data = self.tree[right_child].data
            self.tree[node_id].data = left_data + right_data

    # query function to get the sum of values in the range [L, R]
    def query(self, L, R, node_id=0, start=0, length=None):
        if length is None:
            length = self.n
            self.qr_result = []
            self.qr_h = {}
            # initialize self.access dic with default zero values and length of the tree height
            self.access = {level: 0 for level in range(self.tree.depth() + 1)}
            self.accessed_nodes = []

        if R < self.indices[start] or L > self.indices[start + length - 1]:
            return 0

        if 0 != self.tree[node_id].data:
            self.access[self.tree.depth(node_id)] += 1
            self.accessed_nodes.append(node_id)
        
        if L <= self.indices[start] and self.indices[start + length - 1] <= R:
            if self.tree[node_id].data != 0:
                self.qr_result.append(self.tree[node_id].identifier)
                self.qr_h[self.tree[node_id].identifier] = self.tree.depth(self.tree[node_id])                 
            return self.tree[node_id].data

        mid = length // 2
        left_child = 2 * node_id + 1
        right_child = 2 * node_id + 2
        left_sum = self.query(L, R, left_child, start, mid)
        right_sum = self.query(L, R, right_child, start + mid, length - mid)

        return left_sum + right_sum

    def count_nodes(self):
        return len(self.tree.all_nodes())
    
    def height(self):
        return self.tree.depth()
    
    def is_balanced(self, node_id=0):
        def check_balance(node_id):
            if node_id >= len(self.tree.nodes):
                return True, -1  # If node doesn't exist, it's balanced with height -1

            left_child = 2 * node_id + 1
            right_child = 2 * node_id + 2

            left_balanced, left_height = check_balance(left_child)
            right_balanced, right_height = check_balance(right_child)

            current_height = max(left_height, right_height) + 1
            current_balance = abs(left_height - right_height) <= 1

            return left_balanced and right_balanced and current_balance, current_height

        balanced, _ = check_balance(node_id)
        return balanced

class SegmentTree2D:
    def __init__(self, data):
        self.data = data
        self.x_indices = sorted(set(x for x, y in data.keys()))
        self.nx = len(self.x_indices)
        self.tree = Tree()
        self.st1d_h = 0        
        self.build(data, 0, 0, self.nx)
        self.qr_result = []
        self.qr_h = {}
        self.access = {}
        self.accessed_nodes=[]
        self.accessed_nodes_all = [] # list of tuples of accessed nodes in the 2d segment tree
        self.accessed_2d = [] # accessed nodes in the 2d segment tree original node ids

    def build(self, data, node_id, start, length):
        if length == 1:
            x = self.x_indices[start]
            y_data = {y: data[(x, y)] for (xk, y) in data.keys() if xk == x}
            y_unique = set([d[1] for d in data.keys()])
            y_data_prime = {y: y_data[y] if y in y_data.keys() else 0 for y in y_unique}
            st1d = SegmentTree1D(y_data_prime)
            self.st1d_h = max (st1d.height(), self.st1d_h) 

            if node_id == 0:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=st1d, subtree=st1d.tree)
            else:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=st1d, parent=(node_id - 1) // 2)
                self.tree[node_id].subtree = st1d.tree
                
        else:
            mid = length // 2
            left_child = 2 * node_id + 1
            right_child = 2 * node_id + 2

            if node_id == 0:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=0)
            else:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=0, parent=(node_id - 1) // 2)

            self.build(data, left_child, start, mid)
            self.build(data, right_child, start + mid, length - mid)

            left_tree = self.tree[left_child].data.tree
            right_tree = self.tree[right_child].data.tree

            # combined_data = defaultdict(int)
            # for idx in left_tree.data:
            #     combined_data[idx] += left_tree.data[idx]
            # for idx in right_tree.data:
            #     combined_data[idx] += right_tree.data[idx]
            combined_data = defaultdict(int)
            for node in left_tree.all_nodes():
                combined_data[node.identifier] += node.data
            for node in right_tree.all_nodes():
                combined_data[node.identifier] += node.data


            # x = self.x_indices[start]
            # y_data = {y: data[(x, y)] for (xk, y) in data.keys() if xk == x}
            y_unique = set([d[1] for d in data.keys()])
            y_data_prime = {y: 0 for y in y_unique}            
            y_tree_combined = SegmentTree1D(y_data_prime)
            for node in y_tree_combined.tree.all_nodes():
                node.data = combined_data[node.identifier]               

            # combined_tree = SegmentTree1D(combined_data)
            self.tree[node_id].data = y_tree_combined
            self.tree[node_id].subtree = y_tree_combined.tree

    def update(self, x, y, value, node_id=0, start=0, length=None):
        if length is None:
            length = self.nx

        if length == 1:
            x_idx = self.x_indices[start]
            self.tree[node_id].data.update(y, value)
            self.data[(x, y)] = value
        else:
            mid = length // 2
            left_child = 2 * node_id + 1
            right_child = 2 * node_id + 2

            if start <= self.x_indices.index(x) < start + mid:
                self.update(x, y, value, left_child, start, mid)
            else:
                self.update(x, y, value, right_child, start + mid, length - mid)

            left_tree = self.tree[left_child].data
            right_tree = self.tree[right_child].data

            combined_data = defaultdict(int)
            for idx in left_tree.data:
                combined_data[idx] += left_tree.data[idx]
            for idx in right_tree.data:
                combined_data[idx] += right_tree.data[idx]

            combined_tree = SegmentTree1D(combined_data)
            self.tree[node_id].data = combined_tree


    def query(self, x1, x2, y1, y2, node_id=0, start=0, length=None):
        if length is None:
            length = self.nx
            self.qr_result = []
            self.qr_h = {}
            self.access = {level: 0 for level in range(self.height() + 1)}
            self.accessed_nodes = []
            self.accessed_nodes_all = [] # list of tuples of accessed nodes in the 2d segment tree
            self.accessed_2d = [] # accessed nodes in the 2d segment tree original node ids

        if x2 < self.x_indices[start] or x1 > self.x_indices[start + length - 1]:
            return 0

        self.access[self.tree.depth(node_id)] += 1
        self.accessed_nodes.append(node_id)

        if x1 <= self.x_indices[start] and self.x_indices[start + length - 1] <= x2:

            qr_data = self.tree[node_id].data.query(y1, y2)
            # append x and y values to qr_result
            if qr_data != 0:
                for y_result in self.tree[node_id].data.qr_result:
                    self.qr_result.append(f'{node_id}{y_result}')
                    self.qr_h[f'{node_id}{y_result}'] = self.tree.depth(self.tree[node_id])+self.tree[node_id].data.qr_h[y_result]+1
                
                # extend the self.access with the accessed nodes in the 1d segment tree
                acsess_1d = self.tree[node_id].data.access
                current_level = self.tree.depth(node_id)
                for i in range(len(acsess_1d)):
                    self.access[current_level+i+1] += acsess_1d[i]

                access_ = convert_access_list(max_id_x = self.tree.all_nodes()[-1].identifier, 
                                    max_id_y=self.tree[node_id].subtree.all_nodes()[-1].identifier, nodeid_x= node_id, access_y=self.tree[node_id].data.accessed_nodes)
                # append access_y to the accessed_nodes list
                for i in range(len(access_)):
                    self.accessed_nodes.append(access_[i])

                # append the accessed nodes in the 1d segment tree to the accessed_nodes_all list
                accessed_nodes_1d = self.tree[node_id].data.accessed_nodes
                for i in range(len(accessed_nodes_1d)):
                    self.accessed_nodes_all.append((node_id, accessed_nodes_1d[i]))
                
                # append the accessed nodes in the 1d segment tree to the accessed_2d dictionary
                self.accessed_2d.append({node_id:accessed_nodes_1d})
                

            return qr_data

        mid = length // 2
        left_child = 2 * node_id + 1
        right_child = 2 * node_id + 2
        left_sum = self.query(x1, x2, y1, y2, left_child, start, mid)
        right_sum = self.query(x1, x2, y1, y2, right_child, start + mid, length - mid)

        return left_sum + right_sum

    def count_nodes(self):
        total_nodes = len(self.tree.all_nodes())
        for node in self.tree.all_nodes():
            total_nodes += node.data.count_nodes()
        return total_nodes
    
    def height(self):
        # heights = [self.tree.depth()]
        # for node in self.tree.all_nodes():
        #     heights.append(node.data.height())
        return self.tree.depth() + self.st1d_h + 1
    
    def is_balanced(self, node_id=0):
        def check_balance(node_id):
            if node_id >= len(self.tree.nodes):
                return True, -1

            left_child = 2 * node_id + 1
            right_child = 2 * node_id + 2

            left_balanced, left_height = check_balance(left_child)
            right_balanced, right_height = check_balance(right_child)

            current_height = max(left_height, right_height) + 1
            current_balance = abs(left_height - right_height) <= 1

            return left_balanced and right_balanced and current_balance, current_height

        balanced, _ = check_balance(node_id)
        for node in self.tree.all_nodes():
            balanced = balanced and node.data.is_balanced()
        return balanced
    
    def get_all_nodes(self):
        # all_nodes = []
        all_nodes_dict = {}
        primary_nodes = self.tree.all_nodes()  # All primary nodes in the 2D segment tree
        
        for primary_node in primary_nodes:
            # all_nodes.append(primary_node)
            # all_nodes_dict[primary_node.identifier] = primary_node.data.tree[0].data  # Store the 1D segment tree associated with this primary node
            secondary_tree = primary_node.data.tree  # The 1D segment tree associated with this primary node
            secondary_nodes = secondary_tree.all_nodes()
            # all_nodes.extend(secondary_nodes)  # Add all nodes from the secondary 1D segment tree
            for secondary_node in secondary_nodes:
                # the key is the identifier of the primary node concatenated to the secondary node identifier
                # this is to ensure that the key is unique across all nodes
                all_nodes_dict[f"{primary_node.identifier}{secondary_node.identifier}"] = secondary_node.data

        return all_nodes_dict
    # write a function which returns the number of non-zero nodes in the 2d segment tree
    def count_non_empty_nodes(self):
        all_nodes_dict = self.get_all_nodes()
        return sum(1 for node_data in all_nodes_dict.values() if node_data != 0)
    
    # a function which returns non-empty nodes in the 2d segment tree
    def get_non_empty_nodes(self):
        all_nodes_dict = self.get_all_nodes()
        return {node_id: node_data for node_id, node_data in all_nodes_dict.items() if node_data != 0}

class SegmentTree3D:
    def __init__(self, data):
        self.data = data
        self.x_indices = sorted(set(x for x, y, z in data.keys()))
        self.nx = len(self.x_indices)
        self.tree = Tree()
        self.st2d_h = 0
        self.build(data, 0, 0, self.nx)
        self.qr_result = []
        self.qr_h = {}
        self.access = {}
        self.accessed_nodes = []

    def build(self, data, node_id, start, length):
        if length == 1:
            x = self.x_indices[start]
            yz_data = {(y, z): data[(x, y, z)] for (xk, y, z) in data.keys() if xk == x}
            yz_unique = set([(d[1], d[2]) for d in data.keys()])
            yz_data_prime = {yz: yz_data[yz] if yz in yz_data.keys() else 0 for yz in yz_unique}            
            st2d = SegmentTree2D(yz_data_prime)
            self.st2d_h = max(st2d.height(), self.st2d_h)

            if node_id == 0:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=st2d, subtree=st2d.tree)
            else:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=st2d, parent=(node_id - 1) // 2)
                self.tree[node_id].subtree = st2d.tree
        else:
            mid = length // 2
            left_child = 2 * node_id + 1
            right_child = 2 * node_id + 2

            if node_id == 0:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=0)
            else:
                self.tree.create_node(tag=f'{node_id}', identifier=node_id, data=0, parent=(node_id - 1) // 2)

            self.build(data, left_child, start, mid)
            self.build(data, right_child, start + mid, length - mid)

            left_tree = self.tree[left_child].data
            right_tree = self.tree[right_child].data

            # creating a new 2d segment tree and copy the merged values of the left and right 2d segment trees into the new 2d segment tree
            yz_unique = set([(d[1], d[2]) for d in data.keys()])
            yz_data_prime = {yz: 0 for yz in yz_unique}            
            yz_tree_combined = SegmentTree2D(yz_data_prime)
            # yz_tree_combined = SegmentTree2D(yz_data)

            for node_y in yz_tree_combined.tree.all_nodes():
                for node_z in node_y.data.tree.all_nodes():
                    node_z.data = left_tree.tree[node_y.identifier].data.tree[node_z.identifier].data + right_tree.tree[node_y.identifier].data.tree[node_z.identifier].data
    
            self.tree[node_id].data = yz_tree_combined
            self.tree[node_id].subtree = yz_tree_combined.tree


    def update(self, x, y, z, value, node_id=0, start=0, length=None):
        if length is None:
            length = self.nx

        if length == 1:
            x_idx = self.x_indices[start]
            self.tree[node_id].data.update(y, z, value)
            self.data[(x, y, z)] = value
        else:
            mid = length // 2
            left_child = 2 * node_id + 1
            right_child = 2 * node_id + 2

            if start <= self.x_indices.index(x) < start + mid:
                self.update(x, y, z, value, left_child, start, mid)
            else:
                self.update(x, y, z, value, right_child, start + mid, length - mid)

            left_tree = self.tree[left_child].data
            right_tree = self.tree[right_child].data

            combined_data = defaultdict(int)
            for idx in left_tree.data:
                combined_data[idx] += left_tree.data[idx]
            for idx in right_tree.data:
                combined_data[idx] += right_tree.data[idx]

            combined_tree = SegmentTree2D(combined_data)
            self.tree[node_id].data = combined_tree

    def query(self, x1, x2, y1, y2, z1, z2, node_id=0, start=0, length=None):
        if length is None:
            length = self.nx
            self.qr_result = []
            self.qr_h = {}
            self.access = {level: 0 for level in range(self.height() + 1)} 
            self.accessed_nodes = []           

        if x2 < self.x_indices[start] or x1 > self.x_indices[start + length - 1]:
            return 0
        
        self.access[self.tree.depth(node_id)] += 1
        self.accessed_nodes.append(node_id)

        if x1 <= self.x_indices[start] and self.x_indices[start + length - 1] <= x2:
            qr_data = self.tree[node_id].data.query(y1, y2, z1, z2)
            # append x and y values to qr_result
            if qr_data != 0:
                for y_result in self.tree[node_id].data.qr_result:
                    self.qr_result.append(f'{node_id}{y_result}')
                    self.qr_h[f'{node_id}{y_result}'] = self.tree.depth(self.tree[node_id])+self.tree[node_id].data.qr_h[y_result]+1
                            # extend the self.access with the accessed nodes in the 1d segment tree
                acsess_2d = self.tree[node_id].data.access
                current_level = self.tree.depth(node_id)
                for i in range(len(acsess_2d)):
                    self.access[current_level+i+1] += acsess_2d[i]
                
                accessed_nodes_2d = self.tree[node_id].data.accessed_2d
                access_y =[]
                for access in accessed_nodes_2d:
                    # print(access)
                    access_y.append(list(access.keys())[0])
                    
                    acccess_z = convert_access_list_3d(max_id_x = self.tree.all_nodes()[-1].identifier, 
                                    max_id_y=self.tree[node_id].subtree.all_nodes()[-1].identifier, 
                                    max_id_z=self.tree[node_id].subtree.all_nodes()[-1].subtree.all_nodes()[-1].identifier, 
                                    nodeid_x= node_id, nodeid_y=list(access.keys())[0], access_z=list(access.values())[0])
                    for i in range(len(acccess_z)):
                        self.accessed_nodes.append(acccess_z[i])

                access_ = convert_access_list(max_id_x = self.tree.all_nodes()[-1].identifier, 
                                    max_id_y=self.tree[node_id].subtree.all_nodes()[-1].identifier, nodeid_x= node_id, access_y=access_y)
                
                # append access_y to the accessed_nodes list
                for i in range(len(access_)):
                    self.accessed_nodes.append(access_[i])


            return qr_data

        mid = length // 2
        left_child = 2 * node_id + 1
        right_child = 2 * node_id + 2
        left_sum = self.query(x1, x2, y1, y2, z1, z2, left_child, start, mid)
        right_sum = self.query(x1, x2, y1, y2, z1, z2, right_child, start + mid, length - mid)

        return left_sum + right_sum

    def count_nodes(self):
        total_nodes = len(self.tree.all_nodes())
        for node in self.tree.all_nodes():
            total_nodes += node.data.count_nodes()
        return total_nodes
    
    def height(self):
        return  self.tree.depth() + self.st2d_h + 1
    
    def is_balanced(self, node_id=0):
        def check_balance(node_id):
            if node_id >= len(self.tree.nodes):
                return True, -1

            left_child = 2 * node_id + 1
            right_child = 2 * node_id + 2

            left_balanced, left_height = check_balance(left_child)
            right_balanced, right_height = check_balance(right_child)

            current_height = max(left_height, right_height) + 1
            current_balance = abs(left_height - right_height) <= 1

            return left_balanced and right_balanced and current_balance, current_height

        balanced, _ = check_balance(node_id)
        for node in self.tree.all_nodes():
            balanced = balanced and node.data.is_balanced()
        return balanced
# implementing a get_all_nodes function to get all the nodes in the 3d segment tree
    def get_all_nodes(self):
        all_nodes = []
        all_nodes_dict = {}
        primary_nodes = self.tree.all_nodes()  # All primary nodes in the 3D segment tree
        
        for primary_node in primary_nodes:
            # all_nodes.append(primary_node)
            # secondary_tree = primary_node.data.tree  # The 2D segment tree associated with this primary node
            secondary_nodes = primary_node.data.tree.all_nodes()
            # all_nodes.extend(secondary_nodes)  # Add all nodes from the secondary 2D segment tree
            # adding the thid dimension nodes
            for secondary_node in secondary_nodes:
                # all_nodes.extend(secondary_node.data.tree.all_nodes())
                third_nodes = secondary_node.data.tree.all_nodes()
                for third_node in third_nodes:
                    all_nodes_dict[f"{primary_node.identifier}{secondary_node.identifier}{third_node.identifier}"] = third_node.data
                    # all_nodes.append(third_node)

        return all_nodes_dict

    # a function which returns the number of non-empty nodes in the 3d segment tree
    def count_non_empty_nodes(self):
        all_nodes_dict = self.get_all_nodes()
        non_empty_nodes = 0
        for node_id, node_data in all_nodes_dict.items():
            if node_data != 0:
                non_empty_nodes += 1
        return non_empty_nodes

    # a function which returns non-empty nodes in the 2d segment tree
    def get_non_empty_nodes(self):
        all_nodes_dict = self.get_all_nodes()
        return {node_id: node_data for node_id, node_data in all_nodes_dict.items() if node_data != 0}


# a function to convert the 2-dimensional segment tree to a big merged segment tree
import copy 

# ToDo this currently works for 2D segment trees only, and I should implement for 3D segment trees
def merge_dim_trees(sg_tree):

    # copy the sg_tree to the new tree
    merged_tree = copy.deepcopy(sg_tree)

    # get the x_tree
    x_tree = merged_tree.tree
    y_id = len(x_tree.all_nodes())

    max_id_x = x_tree.all_nodes()[-1].identifier
    max_id_y = x_tree.all_nodes()[-1].data.tree.all_nodes()[-1].identifier
    # iterate over all nodes of the x_tree
    x_tree_nodes = x_tree.all_nodes()
    for x_node in x_tree_nodes:
        # get the subtree of the x_node
        y_tree = x_node.subtree
        # iterate over all nodes of the y_tree
        y_tree_nodes = y_tree.all_nodes()
        for y_node in y_tree_nodes: 
            new_y_id = (max_id_x + 1) + (max_id_y + 1) + (x_node.identifier * ((y_tree_nodes[-1].identifier)+1)) + y_node.identifier            
            y_tree.update_node(y_node.identifier, identifier=new_y_id)                              
            # update the tag
            y_node.tag = f'{new_y_id}'
            y_id +=1
        # paste the updated y_tree to x_node as its child
        x_tree.paste(x_node.identifier,y_tree)
    return merged_tree

# a function that converts the accessed_nodes of the 2d segment tree to the bigtree indices
def convert_access_list(max_id_x, max_id_y, nodeid_x, access_y =[]):
    access_list_new =[]

    for i in range(len(access_y)):
        access_list_new.append((max_id_x + 1) + (max_id_y + 1) + ((nodeid_x) * (max_id_y + 1)) + access_y[i])
    
    return access_list_new



import copy

# Function to merge a 3D segment tree
def merge_dim_trees_3d(sg_tree):
    """
    Merges the dimensions of a 3D segment tree into a single hierarchical structure.
    
    :param sg_tree: The 3D segment tree object to merge.
    :return: A merged tree object with updated identifiers and tags.
    """
    # Deep copy the segment tree to avoid modifying the original
    merged_tree = copy.deepcopy(sg_tree)

    # Get the x_tree
    x_tree = merged_tree.tree
    x_tree_nodes = x_tree.all_nodes()

    # check if the sg_tree is an object of SegmentTree3D
    if isinstance(sg_tree, SegmentTree3D):
        # calculate the maximum identifier for each dimension
        max_id_x = x_tree_nodes[-1].identifier
        max_id_y_prev = x_tree_nodes[-1].data.tree.all_nodes()[-1].identifier
        max_id_z_prev = x_tree_nodes[-1].data.tree.all_nodes()[-1].data.tree.all_nodes()[-1].identifier
        max_id_ytree = (max_id_x + 1) + (max_id_y_prev + 1) + ((max_id_x + 1) * (max_id_y_prev + 1 ))

    # Iterate over all nodes of the x_tree
    for x_node in x_tree_nodes:
        # Get the subtree of the x_node (y_tree)
        y_tree = x_node.subtree
        y_tree_nodes = y_tree.all_nodes()

        for y_node in y_tree_nodes:
            if isinstance(sg_tree, SegmentTree3D):
                # Get the subtree of the y_node (z_tree)
                z_tree = y_node.subtree
                z_tree_nodes = z_tree.all_nodes()

                for z_node in z_tree_nodes:
                    # Calculate the new identifier for z_tree nodes
                    new_z_id = (max_id_ytree + (max_id_z_prev +1) + 
                                (x_node.identifier * (max_id_y_prev + 1) *
                                (max_id_z_prev + 1)) + ((y_node.identifier) *
                                (max_id_z_prev + 1)) + z_node.identifier)
                    
                    z_tree.update_node(z_node.identifier, identifier=new_z_id)
                    z_node.tag = f'{new_z_id}'
                
                # Paste the updated z_tree back to y_node
                y_tree.paste(y_node.identifier, z_tree)

            # Update the identifiers for y_tree nodes
            new_y_id = (
                (max_id_x + 1) + (max_id_y_prev + 1) +
                (x_node.identifier * (y_tree_nodes[-1].identifier + 1)) +
                y_node.identifier
            )
            y_tree.update_node(y_node.identifier, identifier=new_y_id)
            y_node.tag = f'{new_y_id}'

        # Paste the updated y_tree back to x_node
        x_tree.paste(x_node.identifier, y_tree)

    return merged_tree

# Function to convert accessed nodes of the 3D segment tree to the merged tree indices
def convert_access_list_3d(max_id_x, max_id_y, max_id_z, nodeid_x, nodeid_y, access_z=[]):
    """
    Converts the accessed nodes in the 3D segment tree to indices in the merged tree.
    
    :param max_id_x: Maximum identifier for x_tree nodes.
    :param max_id_y: Maximum identifier for y_tree nodes.
    :param max_id_z: Maximum identifier for z_tree nodes.
    :param nodeid_x: Identifier of the x_tree node being accessed.
    :param nodeid_y: Identifier of the y_tree node being accessed.
    :param access_z: List of identifiers accessed in the z_tree.
    :return: A list of new indices in the merged tree.
    """
    access_list_new = []

    for z_id in access_z:
        new_id = (
            max_id_x + 1 +  (max_id_y+1) + 
            (max_id_x + 1) * (max_id_y + 1) +
            (nodeid_x * (max_id_y + 1) * (max_id_z + 1)) +
            (nodeid_y * (max_id_z + 1)) +
            z_id
        )
        access_list_new.append(new_id)

    return access_list_new


# *Example usage 3D Segment Tree:
def segment_tree_example_3d():
    

    # data_3d = {
    #     (1, 1, 1): 1, (1, 1, 3): 2, (1, 1, 5): 3,
    #     (1, 3, 1): 4, (1, 3, 3): 5, (1, 3, 5): 6,
    #     (1, 5, 1): 7, (1, 5, 3): 8, (1, 5, 5): 9,
    #     (3, 1, 1): 10, (3, 1, 3): 11, (3, 1, 5): 12,
    #     (3, 3, 1): 13, (3, 3, 3): 14, (3, 3, 5): 15,
    #     (3, 5, 1): 16, (3, 5, 3): 17, (3, 5, 5): 18,
    #     (5, 1, 1): 19, (5, 1, 3): 20, (5, 1, 5): 21,
    #     (5, 3, 1): 22, (5, 3, 3): 23, (5, 3, 5): 24,
    #     (5, 5, 1): 25, (5, 5, 3): 26, (5, 5, 5): 27
    # }

    # creating a 3d_data sample 2^3
    data_3d = {
        (1, 1, 1): 1, (1, 1, 3): 2,
        (1, 3, 1): 3, (1, 3, 3): 4,
        (3, 1, 1): 5, (3, 1, 3): 6,
        (3, 3, 1): 7, (3, 3, 3): 8
        }

    seg_tree_3d = SegmentTree3D(data_3d)

    print("Initial 3D Segment Tree:")
    print(seg_tree_3d.tree)

    # Query the sum of values in the range [(1, 1, 1), (5, 5, 5)]
    print(f"Sum of values in range [(1, 1, 1), (3, 3, 3)]: {seg_tree_3d.query(1, 3, 1, 3, 1, 2)}")

    # Print tree stats
    print('Number of nodes:', len(seg_tree_3d.get_all_nodes()))
    print('Height of the tree:', seg_tree_3d.height())
    print('Is the tree balanced:', seg_tree_3d.is_balanced())

    # pirnt all nodes in the 3d segment tree
    # print('All nodes in the 3D segment tree:', seg_tree_3d.get_all_nodes().__len__())

    # all_nodes_dict = seg_tree_3d.get_all_nodes()
    # print(f"Total nodes (including three dimensions) using the dictionary: {len(all_nodes_dict)}")
    # for node_id, node_data in all_nodes_dict.items():
    #     print(f"Node ID: {node_id}, Data: {node_data}")
    print('query results:', seg_tree_3d.qr_result)
    print('query results height:', seg_tree_3d.qr_h)
    print('access:', seg_tree_3d.access)

    # print the sub-trees of the second level and the third level of the 3d segment tree
    for node in seg_tree_3d.tree.all_nodes():
        print('Primary node:', node.identifier, 'subtree:', node.subtree)
        for sub_node in node.subtree.all_nodes():
            print('Secondary node:', sub_node.identifier, 'subtree:', sub_node.subtree)
            for sub_sub_node in sub_node.subtree.all_nodes():
                print('Third node:', sub_sub_node.identifier, 'data:', sub_sub_node.data)


    # Merge the 3D segment tree
    merged_big_segtree = merge_dim_trees_3d(seg_tree_3d)
    # print the before and after merge trees
    print('Before merge :', seg_tree_3d.tree)
    print('Merged tree:', merged_big_segtree.tree)

    # print children of the node_id 0
    print('Merged tree segment tree 2d children of the node_id 2: ', merged_big_segtree.tree.children(2))

    # print convert access list
    print('Convert access list:', convert_access_list_3d(max_id_x=2, max_id_y=2, max_id_z=2, nodeid_x=1 ,nodeid_y=2, access_z=[0, 1 ,2])) 

    # print accessed nodes in the 3d segment tree
    print('Accessed nodes in the 3d segment tree:', seg_tree_3d.accessed_nodes)   

# *2D Segment Tree Example test case
def segment_tree_example_2d():

    data_2d = {
        (1, 1): 1, (1, 3): 2, (1, 5): 3,
        (3, 1): 4, (3, 3): 5, (3, 5): 6,
        (5, 1): 7, (5, 3): 8, 
        (7, 1): 10, (7, 3): 11, (7, 5): 12
    }

    seg_tree_2d = SegmentTree2D(data_2d)

    # print("Initial 2D Segment Tree:")
    print(seg_tree_2d.tree)
    
    # print subtrees of the primary nodes
    for node in seg_tree_2d.tree.all_nodes():
        print('Primary node:', node.identifier, 'subtree:', node.subtree)

    # Query the sum of values in the range [(1, 1), (7, 3)]
    print(f"Sum of values in range [(1, 3), (1, 3)]: {seg_tree_2d.query(1, 3, 1, 3)}")

    # Print tree stats
    print('Number of nodes:', seg_tree_2d.count_nodes())
    print('Height of the tree:', seg_tree_2d.height())
    print('Is the tree balanced:', seg_tree_2d.is_balanced())    
    
    all_nodes_dict = seg_tree_2d.get_all_nodes()
    # print(f"Total nodes (including both dimensions): {len(all_nodes)}")
    # for node in all_nodes:
    #     print(f"Node ID: {node.identifier}, Data: {node.data}")

    print(f"Total nodes (including both dimensions) using the dictionary: {len(all_nodes_dict)}")
    # for node_id, node_data in all_nodes_dict.items():
    #     print(f"Node ID: {node_id}, Data: {node_data}")

    # print query result nodes
    print('Query result nodes:', seg_tree_2d.qr_result)
    print('Query result nodes height:', seg_tree_2d.qr_h)
    print('Accessed nodes:', seg_tree_2d.access)
    print('Accessed nodes (in the o_segment tree format):', seg_tree_2d.accessed_nodes)
    print('Accessed nodes all:(in tuple format)', seg_tree_2d.accessed_nodes_all)

    # print('segment tree 2d children of the node_id 2: ', seg_tree_2d.tree.children(2))
    # print('segment tree 2d all nodes: ', len(seg_tree_2d.tree.all_nodes()))

    merged_big_segtree = merge_dim_trees(seg_tree_2d)
    # print the before and after merge trees
    print('Before merge :', seg_tree_2d.tree)
    print('Merged tree:', merged_big_segtree.tree)

    # print children of the node_id 0
    print('Merged tree segment tree 2d children of the node_id 2: ', merged_big_segtree.tree.children(2))

    # print the merged tree nodes ids
    for node in seg_tree_2d.tree.all_nodes():
        print(f"Node ID: {node.identifier}, children: {[ch_node.identifier for ch_node in merged_big_segtree.tree.children(node.identifier)]}")

    # print convert access list
    # print('Convert access list:', convert_access_list(6, 6, 3, [0, 1, 5]))

    # print accessed nodes in the 2d segment tree
    # print('Accessed nodes in the 2d segment tree:', seg_tree_2d.accessed_nodes)

# *1D Segment Tree Example test case
def segment_tree_example_1d():

    data = {1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6}
    seg_tree = SegmentTree1D(data)

    print("Initial Segment Tree:")
    print(seg_tree.tree)

    # Query the sum of values in range [2, 9]
    print(f"Sum of values in range [2, 9]: {seg_tree.query(2, 9)}")
    
    # printing the self.qr_result after the query
    print('Query result nodes:', seg_tree.qr_result)

    # Update the value at index 5 to 6
    # seg_tree.update(5, 6)

    # print("Segment Tree after updating index 5 to 6:")
    # print(seg_tree.tree)

    # Query the sum of values in range [1, 7] again
    # print(f"Sum of values in range [1, 7]: {seg_tree.query(1, 7)}")

    print('Number of nodes:', seg_tree.count_nodes())
    # print('Number of leaves:', seg_tree.count_leaves())
    print('Height of the tree:', seg_tree.height())
    print('Is the tree balanced:', seg_tree.is_balanced())
    print('Number of returned nodes:', len(seg_tree.qr_result))
    print('The qr_result:', seg_tree.qr_result)
    print('The qr_h:', seg_tree.qr_h)
    print('the access:', seg_tree.access)
    print('Accessed nodes:', seg_tree.accessed_nodes)


if __name__ == '__main__':
    # segment_tree_example_1d()
    # segment_tree_example_2d()
    segment_tree_example_3d()

