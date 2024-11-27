
# from Crypto.Util.Padding import pad, unpad
from Cryptodome.Util.Padding import pad, unpad
import pickle
import math, os, sys, time
from treelib import Tree
# from memory_profiler import profile

import pyoram
from pyoram.oblivious_storage.tree.path_oram import PathORAM

# from LSH import LSH
from b4_objs import node_data
from b4_tree import OMapTree

storage_name = "heap"
ext = ".bin"


class OMapE(object):

    def __init__(self, files_dir="", total_accesses=5):
        self.files_dir = files_dir
        self.data_dict = None
        self.tree = None
        self.block_size = 256
        self.node_map = None
        self.storage_name = storage_name
        self.oram_map = None
        self.omap = None
        self.leaf_nodes = None
        self.tmp_map = []
        self.total_accesses = total_accesses
        # for benchmarking purposes
        self.nb_oram_access = 0
        self.time_oram_access = 0

    def build_tree(self, elements=None):
        self.tree = OMapTree.create_tree("1", 2, elements, 5)

    def set_tree(self, tree):
        self.tree = tree
        print("set tree")

    def search_root_nodes(self, query):
        return self.maintree.search_root_nodes(query)

    def padding(self, item):
        if len(item) == self.block_size:
            with_padding = item
        else:
            with_padding = pad(item, self.block_size)
        return with_padding

    def create_blocks(self, serialized_node):
        blocks_list = []
        num_blocks = (len(serialized_node) // self.block_size) + 1

        for i in range(num_blocks):
            block = self.padding(serialized_node[i * self.block_size: (i + 1) * self.block_size])
            blocks_list.append(block)

        return blocks_list

    def retrieve_data(self, depth, node):
        current_oram_map = self.oram_map
        current_oram = self.omap[depth - 1]
        raw_data = []


        if node not in current_oram_map:
            print("Value does not exist")  # for testing
            # print(current_oram_map)
        else:
            t_start = time.time()
            current_oram_file = PathORAM(self.files_dir + self.storage_name + str(depth - 1), current_oram.stash,
                                         current_oram.position_map, key=current_oram.key, storage_type='file')
            blocks_pos = current_oram_map[node]
            for pos in blocks_pos:
                raw_data.append(current_oram_file.read_block(pos))
            self.nb_oram_access += 1

            t_end = time.time()
            self.time_oram_access += t_end - t_start

            rebuilt_node = unpad(b''.join(raw_data), self.block_size)
            orig = pickle.loads(rebuilt_node)

            current_oram_file.close()
        # print("original type:" , type(orig) == list)

        return orig

        # if things in tree are node_data not actual nodes

    def search(self, item):
        queue = []
        next_level_queue = []
        current_level = 1  # hard coded for now
        accesses_made = 0
        nodes_visited = []
        access_depth = {}

        access_depth = {i: [] for i in range(self.tree.depth + 1)}

        leaf_nodes = []

        queue.append(1)

        rest = self.total_accesses - len(queue)
        if rest < 0:
            queue = queue[:self.total_accesses]
        else:
            queue += [(0, 2 ** current_level)] * rest
        assert (len(queue) == self.total_accesses)

        while queue:
            current_node = queue.pop()
            nodes_visited.append(current_node)
            accesses_made += 1
            access_depth[current_level-1].append(current_node)
            original_node_data = self.retrieve_data(current_level, current_node)
            if original_node_data is None:
                print("Was unable to look up data " +  str(current_level) + ", " + str(current_node))
                exit(1)


            if current_level != self.tree.depth+1:
                (lchild, rchild) = original_node_data.get_children()
                if item <= original_node_data.left_max_children:
                    child = lchild
                else:
                    child = rchild
                next_level_queue.append(child)
            elif current_level == self.tree.depth+1:
                if current_node not in leaf_nodes :
                    leaf_nodes.append(original_node_data)
            # if num accesses == total accesses , break loop
            if accesses_made == self.total_accesses:
                queue = []

            if queue == [] and current_level < self.tree.depth+1:
                accesses_made = 0
                current_level += 1
                rest = self.total_accesses - len(next_level_queue)
                if rest < 0:
                    next_level_queue = next_level_queue[:self.total_accesses]
                else:
                    next_level_queue += [(0, 2 ** current_level)] * rest
                assert (len(next_level_queue) == self.total_accesses)
                queue = next_level_queue
                next_level_queue = []

        return nodes_visited, leaf_nodes, access_depth

    def init_maps(self):
        nodes_map = []
        t = self.tree

        for node_id in range(t.root, t.num_nodes + 1):
            current_node = t.get_node_data(node_id)
            if current_node is None:
                raise ValueError("Cannot serialize empty node")
                # default node that doesn't match anything
            pickled_node = pickle.dumps(current_node)
            depth = t.get_depth(node_id)
            # if new depth, create corresponding array
            if len(nodes_map) < depth + 1:
                nodes_map.append([])
            blocks_list = self.create_blocks(pickled_node)
            for block in blocks_list:
                nodes_map[depth].append((node_id, [block]))
        return nodes_map

    def build_oram(self, nodes_map):
        self.omap = [None for i in range(self.tree.get_depth() + 1)]
        self.oram_map = {}

        for (depth, serialized_nodes) in enumerate(nodes_map):
            block_count = len(serialized_nodes)
            block_id = 0

            file_name = self.files_dir + self.storage_name + str(depth)
            if os.path.exists(file_name):
                os.remove(file_name)

            with PathORAM.setup(file_name, self.block_size, block_count, storage_type='file') as f:
                self.omap[depth] = f

                for (node_id, blocks_list) in serialized_nodes:
                    if node_id not in self.oram_map:
                        self.oram_map[node_id] = []

                    for block in blocks_list:
                        f.write_block(block_id, block)
                        # add node to block mapping for client
                        self.oram_map[node_id].append(block_id)
                        block_id = block_id + 1

    def apply(self, elements, block_size=256):
        self.build_tree(elements)
        client_map = self.init_maps()
        self.build_oram(client_map)


if __name__ == "__main__":
    omap = OMapE("./", total_accesses=1)
    temp_dict = {1: "cat", 2: "dog", 3: "horse", 4: "cow"}
    print(type(omap))

    omap.apply(temp_dict)
    (nodes, returned_nodes, access) = omap.search(4)
    print(nodes)
    print(returned_nodes)
    print(access)
