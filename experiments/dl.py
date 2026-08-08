"""
dl.py — Denoising autoencoder imputer (deep-learning baseline).

A masked denoising autoencoder trained per fold on the normalised [0,1] matrix,
with reconstruction loss computed on observed entries only (leakage-safe). This
is the class of model cited in the paper via Gjorshoska2022 (denoising
autoencoders on USDA FoodData Central) but never run there; this script fills
that gap.

Operates in normalised space so it plugs into the same NutrientPreprocessor
pipeline as every other method (fairness). CPU torch is sufficient at this scale.
"""
import numpy as np
import torch
import torch.nn as nn


class _DAE(nn.Module):
    def __init__(self, p, hidden=64, bottleneck=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(p, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, p), nn.Sigmoid(),   # outputs in [0,1]
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def dae_impute(X_input_norm, seed=42, epochs=800, lr=1e-3,
               dropout_frac=0.2, hidden=64, bottleneck=16, verbose=False):
    """
    X_input_norm : (n,p) normalised matrix in [0,1] with NaN at missing entries.
    Returns imputed (n,p) normalised matrix (NaN entries filled by the DAE).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = X_input_norm.astype(np.float32)
    obs = ~np.isnan(X)
    Xz = np.where(obs, X, 0.0).astype(np.float32)      # missing -> 0
    n, p = X.shape

    Xt = torch.tensor(Xz)
    Mt = torch.tensor(obs.astype(np.float32))

    model = _DAE(p, hidden=hidden, bottleneck=bottleneck)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss(reduction='sum')

    model.train()
    for ep in range(epochs):
        # denoising corruption: randomly zero a fraction of OBSERVED inputs,
        # but always score reconstruction against the true observed values.
        corrupt = torch.tensor(
            (rng.random((n, p)) < dropout_frac).astype(np.float32))
        keep_mask = Mt * (1.0 - corrupt)
        x_in = Xt * keep_mask
        opt.zero_grad()
        out = model(x_in)
        # loss on all observed entries (denoised + kept)
        diff = (out - Xt) * Mt
        loss = loss_fn(diff, torch.zeros_like(diff)) / Mt.sum()
        loss.backward()
        opt.step()
        if verbose and ep % 200 == 0:
            print(f"    ep{ep} loss={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        recon = model(Xt).numpy()
    X_out = X.copy()
    X_out[~obs] = recon[~obs]
    return X_out
