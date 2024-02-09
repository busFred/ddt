# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import tqdm

import ddt.nn
import ddt.utils

# %%
# make dataset
rng = np.random.RandomState(1)
xs_train: np.ndarray = np.sort(5 * rng.rand(80, 1), axis=0).astype(np.float32)
ys_train: np.ndarray = np.sin(xs_train).ravel()
ys_train[::5] += 3 * (0.5 - rng.rand(16))
xs_test: np.ndarray = np.arange(0.0, 5.0, 0.01)[:, None].astype(np.float32)

# %%
n_covs: int = xs_train.shape[1]
n_responses: int = 1

# %%
weights, comparators, leaves = ddt.utils.make_ddt_params(
    n_covs=n_covs, n_responses=n_responses, max_leaves=4
)

# %%
ddtr = ddt.nn.DDT(
    n_covs=n_covs,
    n_responses=n_responses,
    weights=weights,
    comparators=comparators,
    leaves=leaves,
    alpha=10.0,
)

# %%
ddtr.eval()
ysh_test_ddtr: np.ndarray = (
    ddtr.forward(th.as_tensor(xs_test)).flatten().numpy(force=True)
)

# %%
plt.scatter(xs_train.flatten(), ys_train)
plt.plot(xs_test.flatten(), ysh_test_ddtr)
plt.show()
plt.close()

# %%
ddtr.train()
opt = th.optim.Adam(ddtr.parameters())
pbar = tqdm.trange(10000)
for itr in pbar:
    yhats: th.Tensor = ddtr.forward(th.as_tensor(xs_train))
    mse_loss: th.Tensor = th.nn.functional.mse_loss(
        yhats, th.as_tensor(ys_train[:, None])
    )
    opt.zero_grad()
    mse_loss.backward()
    opt.step()
    pbar.set_postfix(mse_loss=mse_loss.item())
ddtr.eval()

# %%
ysh_test_ddtr: np.ndarray = (
    ddtr.forward(th.as_tensor(xs_test)).flatten().numpy(force=True)
)

# %%
plt.scatter(xs_train.flatten(), ys_train)
plt.plot(xs_test.flatten(), ysh_test_ddtr)
plt.show()
plt.close()

# %%
