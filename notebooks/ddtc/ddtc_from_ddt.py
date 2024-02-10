# %%
from __future__ import annotations

import numpy as np
import sklearn.datasets as skl_datasets
import sklearn.model_selection
import sklearn.tree as skl_tree
import torch as th
import tqdm

import ddt.nn
import ddt.utils

# %%
xs, ys = skl_datasets.load_iris(return_X_y=True)
xs = xs.astype(np.float32)
xs_train, xs_test, ys_train, ys_test = sklearn.model_selection.train_test_split(xs, ys)

# %%
n_covs: int = xs.shape[1]
n_labels: int = len(np.unique(ys))

# %%
dtc = skl_tree.DecisionTreeClassifier()
dtc.fit(xs_train, ys_train)
yshat_dtc: np.ndarray = dtc.predict_proba(xs_test)  # type:ignore

# %%
weights, comparators, leaves = ddt.utils.make_ddt_params_from_dtc(dtc)

# %%
ddtc = ddt.nn.DDTC.make_ddtc(
    n_covs=n_covs,
    n_labels=n_labels,
    weights=weights,
    comparators=comparators,
    leaves=leaves,
    alpha=1.0,
)

# %%
ddtc.eval()
yshat_ddtc: np.ndarray = ddtc(th.as_tensor(xs_test), return_logits=False).numpy(
    force=True
)

# %%
ddtc.train()
opt = th.optim.Adam(ddtc.parameters())
pbar = tqdm.trange(5000)
for itr in pbar:
    yshat_logits: th.Tensor = ddtc.forward(th.as_tensor(xs_train))
    ce_loss: th.Tensor = th.nn.functional.cross_entropy(
        yshat_logits, th.as_tensor(ys_train)
    )
    opt.zero_grad()
    ce_loss.backward()
    opt.step()
    pbar.set_postfix(ce_loss=ce_loss.item())
ddtc.eval()

# %%
