from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import sklearn.model_selection
import sklearn.tree as skl_tree
import torch as th

if TYPE_CHECKING:
    import sklearn.tree._tree


def make_ddtc_params(n_covs: int, n_responses: int, n_leaves: Optional[int]):
    depth: int = 4 if n_leaves is None else int(math.floor(math.log2(n_leaves)))
    n_leaves = 2**depth if n_leaves is None else n_leaves
    comparators: th.Tensor = (1 / n_covs) * th.ones((2**depth - 1, 1))
    weights: th.Tensor = th.rand((2**depth - 1, n_covs))
    leaves: list[tuple[list[int], list[int], th.Tensor]]
    leaves = _make_init_leaves(n_responses, depth)
    return weights, comparators, leaves


def _make_init_leaves(n_responses: int, depth: int):
    # assume complete tree
    last_level: th.Tensor = th.arange(2 ** (depth - 1) - 1, 2**depth - 1)
    go_left: bool = True
    leaf_idx: int = 0
    leaves: list[tuple[list[int], list[int], th.Tensor]] = list()
    for _ in range(2**depth):
        # for current leaf node
        curr_node: int = last_level[leaf_idx].item()  # type:ignore
        turn_left = go_left
        left_path: list[int] = list()
        right_path: list[int] = list()
        # traverse {left, right} parent backward all the way up to root
        while curr_node >= 0:
            path = left_path if turn_left else right_path
            path.append(curr_node)
            prev_node: int = math.ceil(curr_node / 2) - 1
            turn_left = False if curr_node // 2 > prev_node else True
            curr_node = prev_node
        #
        if go_left:
            go_left = False
        else:
            go_left = True
            leaf_idx = leaf_idx + 1
        new_probs: th.Tensor = th.rand(n_responses)
        leaves.append((sorted(left_path), sorted(right_path), new_probs))
    return leaves


def make_ddtc_params_from_dtc(
    dtc: skl_tree.DecisionTreeClassifier, dtype: th.dtype = th.float32
):
    # input shape
    n_covs: int = dtc.n_features_in_
    assert isinstance(dtc.n_classes_, (np.integer, int))
    n_labels: int = dtc.n_classes_
    # list of all nodes
    # (curr_id, parent_id, is_right), is_leaves
    nodes, is_leaves = _traverse_tree_right_first(dtc.tree_)
    features: np.ndarray = dtc.tree_.feature  # type:ignore
    thresholds: np.ndarray = dtc.tree_.threshold  # type:ignore
    assert dtc.tree_.node_count == len(nodes)
    # non-leaf nodes parameters
    init_weights_l: list[np.ndarray] = list()
    init_comparators_l: list[list[float]] = list()
    # i-th element in init_weights corresponds to Node(id=weight_node_map[i])
    # id = weight_node_map[i]
    weight_node_map: list[int] = list()
    # leaf nodes
    # (parents_ids_left, parents_ids_right, probs)
    leaves: list[tuple[list[int], list[int], np.ndarray]] = list()
    # transform each dt node to ddt node
    for id in range(len(nodes)):
        if is_leaves[id]:
            # parameters for leaf node
            # probability of an instance belonging to current leaf
            probs_n: np.ndarray = np.zeros(n_labels)
            probs_n[np.argmax(dtc.tree_.value[id])] = 1.0
            # traverse the tree backward from current leaf to root
            # (left_parents, right_parents, p_xs)
            leaves.append((*_reverse_traverse_tree_from_leaf(id, nodes), probs_n))
        else:
            # parameters for non-leaf node
            init_weight: np.ndarray = np.zeros((n_covs,))
            init_weight[features[id]] = -1.0
            init_weights_l.append(init_weight)
            init_comparators_l.append([-thresholds[id]])
            weight_node_map.append(id)
    # from laves to init_leavs; parent_ids to parent_idxs
    init_leaves: list[tuple[list[int], list[int], th.Tensor]] = list()
    for parent_ids_left, parent_ids_right, probs_n in leaves:
        parent_weight_idxs_left: list[int] = list()
        parent_weight_idxs_right: list[int] = list()
        for id in parent_ids_left:
            parent_weight_idxs_left.append(weight_node_map.index(id))
        for id in parent_ids_right:
            parent_weight_idxs_right.append(weight_node_map.index(id))
        probs: th.Tensor = th.as_tensor(probs_n, dtype=dtype)
        init_leaves.append((parent_weight_idxs_left, parent_weight_idxs_right, probs))
    init_weights: th.Tensor = th.as_tensor(np.stack(init_weights_l), dtype=dtype)
    init_comparators: th.Tensor = th.as_tensor(init_comparators_l, dtype=dtype)
    return init_weights, init_comparators, init_leaves


def _traverse_tree_right_first(
    tree: sklearn.tree._tree.Tree,
) -> tuple[list[tuple[int, int, bool]], np.ndarray]:
    n_nodes: int = tree.node_count
    # id of each nodes
    left_ids: np.ndarray = tree.children_left  # type:ignore
    right_ids: np.ndarray = tree.children_right  # type:ignore
    # list of all nodes
    # (curr_id, parent_id, is_right)
    nodes: list[tuple[int, int, bool]] = []
    is_leaves: np.ndarray = np.zeros((n_nodes,), dtype=bool)
    # traverse the tree in right node first order
    # (curr_id, curr_depth, parent_id, is_right)
    stack: list[tuple[int, int, int, bool]] = [(0, 0, -1, False)]
    while len(stack) > 0:
        # current node info
        curr_id, curr_depth, parent_id, is_right = stack.pop()
        left_id = left_ids[curr_id]
        right_id = right_ids[curr_id]
        # add current node to nodes
        is_leaf: bool = left_id == right_id
        nodes.append((curr_id, parent_id, is_right))
        is_leaves[curr_id] = is_leaf
        # add child nodes to stack
        if not is_leaf:
            # stack is lifo data structure
            stack.append((left_id, curr_depth + 1, curr_id, False))
            stack.append((right_id, curr_depth + 1, curr_id, True))
    return nodes, is_leaves


def _reverse_traverse_tree_from_leaf(
    leaf_id: int, nodes: Sequence[tuple[int, int, bool]]
):
    # traverse the tree backward from current leaf to root
    parent_ids_left = []
    parent_ids_right = []
    curr_id: int = leaf_id
    while curr_id != 0:
        for id, parent_id, is_right in nodes:
            if curr_id == id:
                parents = parent_ids_right if is_right else parent_ids_left
                parents.append(parent_id)
                curr_id = parent_id
                break
    return parent_ids_left, parent_ids_right
