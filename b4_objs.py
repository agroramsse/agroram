#bloom4, each tree is a lsh 

from treelib import Node, Tree
import math, os, sys
import pickle
from Cryptodome.Util.Padding import pad, unpad

class node_data(object):
    def __init__(self, value, children, left_max_child=None):
        self.value = value
        self.children = children
        self.max_children = None
        self.left_max_children = left_max_child

    def __reduce__(self):
        return self.__class__, (self.value, self.children, self.left_max_children)

    def __str__(self):
        return self.value

    def add_child(self, child):
        # child is node identifier number
        self.children.append(child)
    
    def get_children(self):
        return self.children

    def add_children_data(self, lchild, rchild):
        if type(lchild) is tuple and type(rchild) is tuple:
            self.max_children = rchild[0]
            self.left_max_children = lchild[0]
        # if type(lchild) is list and type(rchild) is list:
        #     self.max_children = rchild[0]
        #     self.left_max_children = lchild[0]

        else:
            if type(lchild) is not node_data or type(rchild) is not node_data:
                print(type(lchild))
                print(lchild)
                print(type(rchild))
                print(rchild)
                raise TypeError()

            self.max_children = rchild.max_children
            self.left_max_children = lchild.max_children
