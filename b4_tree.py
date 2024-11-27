# bloom4, each tree is a lsh

from treelib import Node, Tree
import math, os, sys
import pickle
from Cryptodome.Util.Padding import pad, unpad

import pyoram
from pyoram.util.misc import MemorySize
from pyoram.oblivious_storage.tree.path_oram import PathORAM

from b4_objs import node_data

storage_name = "heap.bin"


class OMapTree(object):
    def __init__(self, identifier, branching_f, vector_length):
        self.identifier = identifier
        self.vector_length = vector_length
        self.branching_factor = branching_f
        self.num_nodes = 0
        self.max_elem = None

        self.tree = None
        self.depth = None
        self.root = branching_f - 1

        self.levels = {}

    @staticmethod
    def create_tree(ident, branching_factor, elements, vector_length):
        t = OMapTree(ident, branching_factor, vector_length)
        t.build_tree(elements)
        return t

    # testing funcions
    def show_tree(self):
        self.tree.show(line_type='ascii')

    # calculate the number of max elements based on the size of the given list
    def calculate_max_elem(self, num_elements):
        # leaf nodes are hash output
        self.max_elem = 2 ** (math.ceil(math.log2(num_elements)))

    # calculate depth of the tree
    def calculate_depth(self):
        self.depth = math.ceil(math.log(self.max_elem, self.branching_factor))

    """functions for nodes"""
    def get_node(self, identifier):
        return self.tree.get_node(identifier)

    def get_node_data(self, node):
        # might need to change this later, specifying bloom_filter bc current node object has plaintext and bloom filer
        return self.tree.get_node(node).data

    def get_node_items(self, node):
        return self.get_node(node).data.items

    def add_child(self, parent_node, child):
        node_type = self.tree.get_node(parent_node).data

        if node_type is not None:
            node_type.add_child(child)

    # returns children of current node
    def get_children(self, node):
        return self.tree.children(node)

    def return_root_data(self):
        return self.get_node(self.root).data

    # get depth of tree or node
    def get_depth(self, node=None):
        if node == None:
            return self.depth
        else:
            return self.tree.depth(node)

    def check_leaf(self, node):
        return self.get_node(node).is_leaf()

    def new_node(self, current_node, parent_node, elements=None, leaf=False):
        self.num_nodes += 1

        if current_node == "root":
            # corner case: current_node == "root", parent_node == self.root,
            _node_ = node_data(value=current_node, children=[], left_max_child=None)
            self.tree.create_node(current_node, self.root, data=_node_)
        else:
            if leaf:
                if elements is not None:
                    _node_ = elements
                else:
                    _node_ = [(self.vector_length - 1, 2)] * self.l
            else:
                _node_ = node_data(value=current_node, children=[], left_max_child=None)
            self.add_child(parent_node, current_node)
            self.tree.create_node(str(current_node), current_node, data=_node_, parent=parent_node)

    def build_tree(self, og_elements):
        self.tree = Tree()
        # print(og_elements)
        num_elements = len(og_elements)
        level = 0

        self.calculate_max_elem(num_elements)  # max number of elements WITH calculation of eLSH outputs
        self.calculate_depth()
        elements = [(k, og_elements[k]) for k in og_elements.keys()]
        elements.sort(key=str)
        # initialize root node
        self.new_node("root", self.root, num_elements, elements)
        current_node = self.root
        parent_node = self.root - 1  # -1 is just for it to work overall

        while level != self.depth:
            level += 1
            nodes_in_level = self.branching_factor ** level
            items_in_filter = self.branching_factor ** (self.depth - level)

            if level == self.depth:
                for n in range(nodes_in_level):
                    current_node += 1
                    if current_node % self.branching_factor == 0:
                        parent_node += 1

                    # if n < self.l:
                    if n < num_elements:
                        self.new_node(current_node, parent_node, elements=elements[n], leaf=True)
                    else:
                        self.new_node(current_node, parent_node, elements=None, leaf=True)

            else:
                for n in range(nodes_in_level):
                    current_node += 1
                    if current_node % self.branching_factor == 0:
                        parent_node += 1

                    begin = n * items_in_filter
                    end = (n * items_in_filter) + items_in_filter
                    elements_in_filter = elements[begin:end]
                    self.new_node(current_node, parent_node, elements=elements_in_filter)

        while parent_node > 0:
            parent_node_data = self.get_node_data(parent_node)
            (child_1, child_2) = parent_node_data.get_children()
            if child_1 >= child_2:
                raise ValueError
            parent_node_data.add_children_data(self.get_node_data(child_1), self.get_node_data(child_2))
            parent_node -= 1

    def search(self, item):
        depth = self.tree.depth()
        stack = []
        nodes_visited = []
        leaf_nodes = []
        returned_iris = []
        access_depth = {}
        root_bf = self.tree[self.root].data

        access_depth[0] = []
        access_depth[0].append(self.root)
        stack.append(self.root)
        while stack != []:
            current_node = stack.pop()
            nodes_visited.append(current_node)
            children = self.get_children(current_node)
            if children != []:
                left_subtree_max = self.get_node_data(current_node).left_max_children
                if item <= left_subtree_max:
                    child = children[0].identifier
                else:
                    child = children[1].identifier
                child_depth = self.tree.depth(current_node) + 1
                try:
                    nodes_per_level = access_depth[child_depth]
                except KeyError:
                    nodes_per_level = []
                nodes_per_level.append(child)
                access_depth[child_depth] = nodes_per_level
                stack.append(child)
            else:
                if self.get_node_data(current_node)[0] == item:
                    leaf_nodes.append(current_node)

        result_set = [self.get_node_data(l) for l in leaf_nodes]
        return nodes_visited, result_set, access_depth
