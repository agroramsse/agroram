from typing import Dict, Any, List

# from Crypto.Util.Padding import pad, unpad
from Cryptodome.Util.Padding import pad, unpad
import pickle
import math, os, sys, time
from treelib import Tree
# from memory_profiler import profile
import csv
import copy
import pyoram
from pyoram.oblivious_storage.tree.path_oram import PathORAM
from segtree import SegmentTree1D

# from LSH import LSH
from b4_objs import node_data
from b4_tree import OMapTree

storage_name = "heap"
ext = ".bin"

EARLY_TERMINATION = False

class OSegmentTree(object):

    def __init__(self, files_dir="", accesses_per_level=[]):
        self.files_dir = files_dir
        self.data_dict = None
        self.tree = None
        self.block_size = 256
        self.node_map = None
        self.storage_name = storage_name
        self.oram_map = None
        self.omap = None
        self.oram_handles = None
        self.leaf_nodes = None
        self.tmp_map = []
        self.total_accesses = accesses_per_level
        self.tree_depth = 0
        # for benchmarking purposes
        self.nb_oram_access = 0
        self.time_oram_access = 0
        self.max_block_size = self.block_size


    def set_tree(self, tree):
        self.tree = tree

    def search_root_nodes(self, query):
        return self.maintree.search_root_nodes(query)


    def padding(self, item):
        return pad(item, self.block_size)

    def create_blocks(self, serialized_node):
        if len(serialized_node) > self.max_block_size:
            self.max_block_size = len(serialized_node)
        pad_item = self.padding(serialized_node)
        if len(pad_item) % self.block_size != 0:
            raise ValueError("Did not pad to correct length")

        blocks_list = []
        num_blocks = (len(pad_item) // self.block_size)
        for i in range(num_blocks):
            block = pad_item[i * self.block_size:(i + 1) * self.block_size]
            blocks_list.append(block)

        return blocks_list

    def retrieve_data(self, depth, node):
        current_oram_map = self.oram_map
        current_oram = self.omap
        raw_data = []


        if node not in current_oram_map:
            return None
        else:
            t_start = time.time()
            current_oram_file = self.oram_handles
            blocks_pos = current_oram_map[node]
            for pos in blocks_pos:
                raw_data.append(current_oram_file.read_block(pos))
            self.nb_oram_access += 1

            t_end = time.time()
            self.time_oram_access += t_end - t_start

            rebuilt_node = unpad(b''.join(raw_data), self.block_size)
            orig = pickle.loads(rebuilt_node)

        return orig

        # if things in tree are node_data not actual nodes


    def search(self, nodes_to_access):

        queue = []
        next_level_queue = []
        current_level = 1  # hard coded for now
        accesses_made = 0
        nodes_visited = []
        access_depth = {}
        nodes_to_access = copy.deepcopy(nodes_to_access)
        access_depth = {i: [] for i in range(self.tree_depth + 1)}
        time_per_level = {i: [] for i in range(self.tree_depth + 1)}
        leaf_nodes = []
        self.oram_handles = {}

        if len(nodes_to_access) == 0:
            return [], [], []

        # for depth in range(self.tree_depth+1):
        current_oram = self.omap
        self.oram_handles = PathORAM(self.files_dir + self.storage_name , current_oram.stash,current_oram.position_map, key=current_oram.key, storage_type='file')

        queue.append(nodes_to_access.pop(0))
        rest = self.total_accesses[0] - len(queue)
        if rest < 0:
            queue = queue[:self.total_accesses[current_level-1]]
        else:
            queue += [(0, 2 ** current_level)] * rest
        assert (len(queue) == self.total_accesses[0])


        while queue:
            current_node = queue.pop(0)

            nodes_visited.append(current_node)
            accesses_made += 1
            access_depth[current_level-1].append(current_node)
            #print("Accessing Node",current_level,current_node)
            t_start = time.time()
            original_node_data = self.retrieve_data(current_level, current_node)
            time_per_level[current_level-1].append(time.time() - t_start)
            #print(original_node_data)
            if original_node_data is None:
                print("Was unable to look up data " +  str(current_level) + ", " + str(current_node))
                exit(1)

            # if num accesses == total accesses , break loop
            if accesses_made == self.total_accesses[current_level-1]:
                queue = []
            if len(queue) ==0 and current_level < self.tree_depth+1:
                accesses_made = 0
                current_level += 1
                if nodes_to_access == [] and EARLY_TERMINATION:
                    time_max = []
                    self.oram_handles.close()
                    for depth in time_per_level:
                        if len(time_per_level[depth]) != 0:
                            time_max.append(max(time_per_level[depth]))
                    return nodes_visited, access_depth, time_max
                for node_id in nodes_to_access:
                    if self.tree.depth(node_id) == current_level-1:
                        next_level_queue.append(node_id)

                for node_id in next_level_queue:
                    nodes_to_access.remove(node_id)
                rest = self.total_accesses[current_level-1] - len(next_level_queue)
                if rest < 0:
                    next_level_queue = next_level_queue[:self.total_accesses[current_level-1]]
                else:
                    for i in range(rest):
                        next_level_queue.append(0)

                assert (len(next_level_queue) == self.total_accesses[current_level-1])
                queue = next_level_queue
                next_level_queue = []

        time_max = []
        self.oram_handles.close()
        for depth in time_per_level:
            if len(time_per_level[depth]) != 0:
                time_max.append(max(time_per_level[depth]))
        return nodes_visited, access_depth, time_max

    def init_maps(self):
        nodes_map = []
        t = self.tree

        num_nodes = len(t.all_nodes())
        
        queue: list[Any] = []
        queue.append(t.root)
        depth = 0
        next_level_queue = []
        i=0
        while queue:
            node_id = queue.pop()

            current_node = t.get_node(node_id)
            if current_node is not None and current_node.data is not None and  current_node.data != 0:
                pickled_node = pickle.dumps(node_id)
                # if new depth, create corresponding array
                # if len(nodes_map) < depth + 1:
                #     nodes_map.append([])
                blocks_list = self.create_blocks(pickled_node)
                for block in blocks_list:
                    nodes_map.append((node_id, [block]))

                for child in t.children(node_id):
                    next_level_queue.append(child.identifier)

            if  [] == queue and [] != next_level_queue:
                depth+=1
                queue = next_level_queue
                next_level_queue = []

        return nodes_map

    def build_oram(self, nodes_map):
        self.omap = [None for i in range(self.tree.depth() + 1)]
        self.oram_map = {}
        
        block_count = len(nodes_map)
        block_id = 0
        print(block_count)
        file_name = self.files_dir + self.storage_name 
        if os.path.exists(file_name):
            os.remove(file_name)

        with PathORAM.setup(file_name, self.block_size, block_count, storage_type='file') as f:
            self.omap = f

            for (node_id, blocks_list) in nodes_map:
                if node_id not in self.oram_map:
                    self.oram_map[node_id] = []

                for block in blocks_list:
                    f.write_block(block_id, block)
                    # add node to block mapping for client
                    self.oram_map[node_id].append(block_id)
                    block_id = block_id + 1


            

        print("Finished building ORAM")

    def apply(self, segtree, block_size=256):
        self.set_tree(segtree)
        self.tree_depth = self.tree.depth()
        print("Tree depth", self.tree_depth)
        print("Length of access", len(self.total_accesses))
        assert(self.tree_depth+1 == len(self.total_accesses))

        client_map = self.init_maps()
        self.build_oram(client_map)


if __name__ == "__main__":

    data = []
    with open('datasets/amazon-books.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(int(row[0]))

    data.sort()
    data=data[:3]
    data_dict = {val: 1 for val in data}
    print(data_dict)
    seg_tree = SegmentTree1D(data_dict)
    print(seg_tree.tree)

    osegtree = OSegmentTree("./", accesses_per_level=[1,2,4])
    osegtree.apply(seg_tree.tree)

    (nodes, access, time_max) = osegtree.search([0,1,2,5,6])
    print(nodes)
    print(access)
    print(time_max)
