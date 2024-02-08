# %%
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import sklearn.datasets as skl_datasets
import sklearn.model_selection
import sklearn.tree as skl_tree

import ddt.utils

if TYPE_CHECKING:
    import sklearn.tree._tree


# %%
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
    init_leaves: list[tuple[list[int], list[int], np.ndarray]] = list()
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


# %%
def ddt_init_from_dt(estimator: skl_tree.DecisionTreeClassifier):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    num_feats = estimator.n_features_in_

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1, -1, False)]
    master_list = []
    while len(stack) > 0:
        node_id, parent_depth, parent_node_id, right_child = stack.pop()
        node_depth[node_id] = parent_depth + 1
        master_list.append([node_id, parent_node_id, right_child])
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1, node_id, False))
            stack.append((children_right[node_id], parent_depth + 1, node_id, True))
        else:
            is_leaves[node_id] = True

    init_weights = []
    init_comparators = []
    leaves = []
    weight_node_map = []
    init_leaves = []
    for i in range(n_nodes):
        if is_leaves[i]:
            probs = np.zeros(estimator.n_classes_)
            probs[np.argmax(estimator.tree_.value[i])] = 1.0
            new_leaf = [[], [], probs]
            current_id = i
            while current_id != 0:
                for node in master_list:
                    if node[0] == current_id:
                        if node[2]:
                            new_leaf[1].append(node[1])
                        else:
                            new_leaf[0].append(node[1])
                        current_id = node[1]
                        break
            leaves.append(new_leaf)
        else:
            init_weight = np.zeros(num_feats)

            init_weight[feature[i]] = -1.0
            init_weights.append(init_weight)
            init_comparators.append([-threshold[i]])

            weight_node_map.append(i)
    # return init_weights, init_comparators, leaves
    for leaf in leaves:
        new_left = []
        new_right = []
        for left_turn in leaf[0]:
            new_left.append(weight_node_map.index(left_turn))
        for right_turn in leaf[1]:
            new_right.append(weight_node_map.index(right_turn))
        init_leaves.append([new_left, new_right, leaf[2]])
    return init_weights, init_comparators, init_leaves


# %%
xs, ys = skl_datasets.load_iris(return_X_y=True)
xs_train, xs_test, ys_train, ys_test = sklearn.model_selection.train_test_split(xs, ys)

# %%
dtc = skl_tree.DecisionTreeClassifier()
dtc.fit(xs_train, ys_train)

# %%
features: np.ndarray = dtc.tree_.feature  # type:ignore
thresholds: np.ndarray = dtc.tree_.threshold  # type:ignore
# print tree result
nodes, is_leaves = _traverse_tree_right_first(dtc.tree_)
for curr_id, parent_id, is_right in nodes:
    print(
        f"[{features[curr_id]}, {thresholds[curr_id]: .2f}], {curr_id}, {parent_id}, {is_right}, {is_leaves[curr_id]}"
    )
# plot tree
_ = skl_tree.plot_tree(dtc, label="all", node_ids=True, filled=False, impurity=False)

# %%
init_weights_, init_comparators_, init_leaves_ = make_ddtc_params_from_dtc(dtc)
init_weights_t, init_comparators_t, init_leaves_t = ddt_init_from_dt(dtc)

# %%
init_weights_, init_comparators_, init_leaves_ = ddt.utils.make_ddtc_params_from_dtc(
    dtc
)
init_weights_t, init_comparators_t, init_leaves_t = ddt_init_from_dt(dtc)

# %%
