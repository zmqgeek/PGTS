# Environment for interacting with Coq theorems during testing
from serapi import SerAPI, CoqExn, CoqTimeout
import re
from copy import deepcopy
from time import time
from collections import OrderedDict
import json
import sexpdata
from utils import update_env
import os
from glob import glob
import pdb


class ProofSearchTree:
    def __init__(self, current_proof_state, used_tactic=None, parent=None):
        """
        初始化证明搜索树节点。

        :param current_proof_state: 当前证明状态
        :param used_tactic: 使用的战术 (默认为 None)
        :param parent: 父节点 (默认为 None)
        """
        self.current_proof_state = current_proof_state
        self.used_tactic = used_tactic
        self.parent = parent  # 记录父节点
        self.children = []

    def add_child(self, child_node):
        """
        添加子节点。

        :param child_node: 需要添加的子节点
        """
        child_node.parent = self  # 设置子节点的父节点
        self.children.append(child_node)

    def backtrack(self):
        """
        回溯到父节点。

        :return: 父节点，如果没有父节点则返回 None
        """
        return self.parent

    def __repr__(self):
        return f"ProofSearchTree(state={self.current_proof_state}, tactic={self.used_tactic})"

    def print_tree(self, level=0):
        """
        打印树的结构，便于可视化。

        :param level: 当前树的层级
        """
        indent = '  ' * level
        print(f"{indent}Node: {self.current_proof_state}, Tactic: {self.used_tactic}")
        for child in self.children:
            child.print_tree(level + 1)

    def to_dict(self):
        """
        将树节点及其子节点递归转换为字典。
        
        :return: 字典表示的树节点
        """
        return {
            "current_proof_state": self.current_proof_state,
            "used_tactic": self.used_tactic,
            "children": [child.to_dict() for child in self.children]  # 递归转换子节点
        }
