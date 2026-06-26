# FlashSAC backbone changes for the RSL-RL-SAC reproduction (v5)

These edits are applied **in the working-tree `FlashSAC/` copy** (this folder's vendored backbone).
They implement parts of arXiv:2605.24975 that live inside the agent rather than the env wrapper.
If you re-clone FlashSAC, re-apply these.

## 1. Controlled actor initialization (paper §3.3, eq 7–8)
`flash_rl/agents/flashSAC/layer.py` — `NormalTanhPolicy`:
- Replaced the weight-normalized `UnitLinear` mean/std heads with **plain `nn.Linear`** heads
  (`mean_head`, `std_head`). Plain Linears have no `normalize_parameters`, so they are automatically
  excluded from `network.Network`'s weight-normalization pass and the controlled init persists.
- Init: `mean_head.weight ~ N(0, 1e-3)`, `mean_head.bias = 0` (start at default pose); `std_head.weight = 0`,
  `std_head.bias = atanh(...)` chosen so the (smooth-bounded) std equals `init_sigma` at init,
  state-independent. New ctor arg `init_sigma` (default 0.15).
- `get_mean_and_std` now uses `self.mean_head(x)` / `self.std_head(x)`.

`flash_rl/agents/flashSAC/network.py` — `FlashSACActor`: new `init_sigma` arg, threaded to `NormalTanhPolicy`.

## 2. Config + temperature LR / init_sigma
`flash_rl/agents/flashSAC/agent.py`:
- `FlashSACConfig`: added `init_sigma: float = 0.15` and `temp_learning_rate: float = 2e-5` (defaults
  keep old configs loadable).
- `_init_flashsac_networks`: pass `init_sigma` to `FlashSACActor`; the **temperature optimizer uses a
  constant `cfg.temp_learning_rate`** (decoupled from the actor/critic cosine schedule; `scheduler=None`).

## Note on checkpoint compatibility
The policy-head rename (`predictor.mean_w/std_w/*_bias` → `predictor.mean_head/std_head.*`) means
**v5 checkpoints are not state-dict-compatible with pre-v5 ones.** `export_policy.py` infers the
action dim from `predictor.mean_head.bias`. Fresh training only.

## 3. Exact n-step bootstrap discount (paper §3.5, eq 11)
`flash_rl/buffers/torch_buffer.py` + `flash_rl/agents/flashSAC/update.py`:
- The buffer now stores a per-transition **bootstrap discount** `_discounts = γ^m`, where `m` = steps
  from `t` to the bootstrap state: `m = n` for windows with no episode boundary, `m = k+1` if the
  episode ends (timeout/failure) at window-step `k`. Computed in `_get_n_step_prev_transition`
  (default `γ^n`, overwritten to `γ^{k+1}` at the earliest done), stored/sampled/saved alongside the
  other fields.
- `update_critic` passes `batch["discount"]` (per-sample `γ^m`) to `_compute_categorical_td_target`
  instead of a fixed `gamma**n_step`; the function's `gamma` arg is now a `(B,)` tensor.
- Removes the only §3.5 approximation (mid-window timeouts were previously discounted by `γ^n`
  instead of `γ^{k+1}`). The reward sum + masking were already exact.

## What is NOT here (lives in the launcher, not the backbone)
- Per-joint asymmetric action bounds (eq 5–6): `Go2IsaacEnvWrapper` in `train.py`/`play.py`/`evaluate.py`.
- Timeout pre-reset observation (eq 9–10): `_reset_idx` monkeypatch in the wrapper. The critic
  masking (`done=terminated`) and masked n-step were already correct in `update.py`/`torch_buffer.py`.
- Hyperparameters (γ0.97, n=5, τ0.003, target entropy ≈−2 via `temp_target_sigma=0.205`, 2M CPU
  buffer, LR 2e-4): set in `train.py`'s `FlashSACConfig`.
