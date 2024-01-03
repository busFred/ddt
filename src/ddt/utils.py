from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import sklearn.model_selection
import sklearn.tree as skl_tree

if TYPE_CHECKING:
    import sklearn.tree._tree


def make_ddtc_params_from_dtc(dtc: skl_tree.DecisionTreeClassifier):
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
    init_weights: list[np.ndarray] = list()
    init_comparators: list[list[float]] = list()
    # i-th element in init_weights corresponds to Node(id=weight_node_map[i])
    # id = weight_node_map[i]
    weight_node_map: list[int] = list()
    # leaf nodes
    # (parents_ids_left, parents_ids_right, probs)
    leaves: list[tuple[list[int], list[int], np.ndarray]] = list()
    init_leaves: list = list()
    # transform each dt node to ddt node
    for id in range(len(nodes)):
        if is_leaves[id]:
            # parameters for leaf node
            # probability of an instance belonging to current leaf
            probs: np.ndarray = np.zeros(n_labels)
            probs[np.argmax(dtc.tree_.value[id])] = 1.0
            # traverse the tree backward from current leaf to root
            # (left_parents, right_parents, p_xs)
            leaves.append((*_reverse_traverse_tree_from_leaf(id, nodes), probs))
        else:
            # parameters for non-leaf node
            init_weight: np.ndarray = np.zeros((n_covs,))
            init_weight[features[id]] = -1.0
            init_weights.append(init_weight)
            init_comparators.append([-thresholds[id]])
            weight_node_map.append(id)
    # from laves to init_leavs; parent_ids to parent_idxs
    for parent_ids_left, parent_ids_right, probs in leaves:
        parent_weight_idxs_left: list[int] = list()
        parent_weight_idxs_right: list[int] = list()
        for id in parent_ids_left:
            parent_weight_idxs_left.append(weight_node_map.index(id))
        for id in parent_ids_right:
            parent_weight_idxs_right.append(weight_node_map.index(id))
        init_leaves.append((parent_weight_idxs_left, parent_weight_idxs_right, probs))
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
