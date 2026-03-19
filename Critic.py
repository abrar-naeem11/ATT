import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ── Device selection ──────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CriticNetwork] Using device: {DEVICE}")


class CriticMLP(nn.Module):
    """
    Single hidden-layer MLP implementing the basis function φ_i(X_i).
    Architecture: [6 input → 19 hidden (tanh) → 1 output (linear)]
    as stated in Table I of the paper.

    The hidden activations ARE the basis vector φ_i used in Eq. 29 & 35.
    The scalar output is the value estimate V̂_ic = ŵ_i^T φ_i + B̄_c_i.
    """

    def __init__(self, state_dim: int = 6, hidden: int = 19):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1, bias=False)   # output weight = ŵ_i

        # Xavier initialisation — keeps tanh activations in linear region
        nn.init.uniform_(self.fc1.weight, a=-0.01, b=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 6)  →  V̂: (batch, 1)"""
        return self.fc2(torch.tanh(self.fc1(x)))

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return hidden activations φ_i(x).
        x: (batch, 6)  →  phi: (batch, 19)
        """
        return torch.tanh(self.fc1(x))

    def basis_and_jacobian(self, x: torch.Tensor):
        """
        Return φ_i and its Jacobian ∂φ_i/∂x via autograd.
        x   : (6,) single state  (will be unsqueezed internally)
        Returns
        -------
        phi  : (19, 1) tensor
        dphi : (19, 6) Jacobian  ∂φ_i/∂x
        """
        x = x.detach().requires_grad_(True)        # (6,)
        x_in = x.unsqueeze(0)                      # (1, 6)

        phi_row = torch.tanh(self.fc1(x_in))       # (1, 19)
        m = phi_row.shape[1]

        dphi_rows = []
        for j in range(m):
            grad_j = torch.autograd.grad(
                phi_row[0, j], x,
                retain_graph=True,
                create_graph=False
            )[0]                                    # (6,)
            dphi_rows.append(grad_j.unsqueeze(0))  # (1, 6)

        dphi = torch.cat(dphi_rows, dim=0)          # (19, 6)
        phi  = phi_row.squeeze(0).unsqueeze(1)      # (19, 1)

        return phi, dphi


class ReplayBuffer:
    """
    Fixed-size FIFO replay buffer storing experience tuples.
    All tensors live on DEVICE for zero-copy training.
    """

    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, phi_dot: torch.Tensor,
             Lc: float,
             B_bar_dot: float):
        """Store a single experience (phi_dot, Lc, B_bar_dot)."""
        self.buf.append((
            phi_dot.detach().clone(),          # (m, 1) on DEVICE
            torch.tensor([[Lc]],         dtype=torch.float32, device=DEVICE),
            torch.tensor([[B_bar_dot]],  dtype=torch.float32, device=DEVICE),
        ))

    def sample(self, batch_size: int):
        """
        Return up to `batch_size` most-recent samples as stacked tensors.
        Returns (phi_dots, Lcs, B_bar_dots) each shape (m,B) / (1,B) / (1,B).
        """
        items = list(self.buf)[-batch_size:]
        phi_dots   = torch.cat([e[0] for e in items], dim=1)   # (m, B)
        Lcs        = torch.cat([e[1] for e in items], dim=1)   # (1, B)
        B_bar_dots = torch.cat([e[2] for e in items], dim=1)   # (1, B)
        return phi_dots, Lcs, B_bar_dots

    def __len__(self):
        return len(self.buf)


class CriticNetwork:
    """
    GPU-accelerated critic-only RL network — Algorithm 1 of the paper.

    Equations implemented
    ---------------------
    Eq. 29 / 35  : Value function approximation  V̂_ic = ŵ_i^T φ_i + B̄_c_i
    Eq. 36       : Control policy  û_i = −α_i tanh(Λ̂_i) + β̄_i
    Eq. 37       : Disturbance policy  d̂_i = (1/2δ²) ∇V̂_ic^T
    Eq. 39 / 40  : Hamiltonian estimation error
    Eq. 42       : Gradient-descent weight update with experience replay
    Eq. 30       : Regularised CBF  B̄_c_i
    """

    def __init__(self,
                 state_dim: int   = 6,
                 m: int           = 19,
                 lr: float        = 0.5,
                 delta: float     = 2.0,
                 kappa: float     = 0.1,
                 replay_buffer_size: int = 100_000,
                 sigma_reg: float = 1e-3,
                 mini_batch_size: int = 64):

        self.state_dim       = state_dim
        self.m               = m
        self.lr              = lr
        self.delta           = delta
        self.kappa           = kappa
        self.sigma_reg       = sigma_reg
        self.mini_batch_size = mini_batch_size
        self.device          = DEVICE

        # ── Neural network on GPU ──────────────────────────────────────
        self.net = CriticMLP(state_dim, m).to(DEVICE)

        self.dt = 0.01

        # Adam optimiser — more stable than plain SGD for RL critics
        self.optimiser = optim.SGD(self.net.parameters(), lr=lr * self.dt)

        # ── Replay buffer ──────────────────────────────────────────────
        self.replay = ReplayBuffer(capacity=replay_buffer_size)

        # ── Numerical safety limits ────────────────────────────────────
        self.GRAD_CLIP = 1.0
        self.LC_CLIP   = 1e6
        self.ERR_CLIP  = 1e3
        self.IN_CLIP   = 1e3

        # expose .w for logging (mirrors paper notation ŵ_i)
        self._sync_w()

    # ── Internal helper ───────────────────────────────────────────────
    def _sync_w(self):
        """Keep self.w (numpy) in sync with fc2 weights for external logging."""
        self.w = self.net.fc2.weight.detach().cpu().numpy().T   # (m, 1)

    def _to_tensor(self, x, clip=None):
        """Convert numpy array / scalar to a float32 GPU tensor."""
        t = torch.tensor(
            np.asarray(x, dtype=np.float32),
            device=self.device
        )
        if clip is not None:
            t = t.clamp(-clip, clip)
        return t

    # ──────────────────────────────────────────────────────────────────
    # CBF  (Eq. 30)
    # ──────────────────────────────────────────────────────────────────

    def compute_B_bar(self, ei, k_ie):
        """
        Regularised CBF B̄_c_i — Eq. 30  (runs on CPU / numpy, very cheap).

        Parameters
        ----------
        ei   : (3,) position error
        k_ie : (3,) per-axis safety bounds

        Returns
        -------
        B_bar      : float
        dB_bar_dei : (3,) numpy  ∂B̄_c_i/∂e_i
        """
        ei   = np.asarray(ei,   dtype=np.float64).flatten()
        k_ie = np.asarray(k_ie, dtype=np.float64).flatten()

        B_bar      = 0.0
        dB_bar_dei = np.zeros(3)

        for idx in range(3):
            e_o = np.clip(ei[idx], -(k_ie[idx] - 1e-3), (k_ie[idx] - 1e-3))
            k_o = k_ie[idx]

            num   = (k_o * e_o) ** 2
            denom = (k_o + e_o) * (k_o - e_o) + self.sigma_reg

            B_bar += num / denom

            dnum            = 2.0 * k_o ** 2 * e_o
            dden            = -2.0 * e_o
            dB_bar_dei[idx] = (dnum * denom - num * dden) / (denom ** 2)

        B_bar      = float(np.clip(self.kappa * B_bar,      -1e6, 1e6))
        dB_bar_dei = np.clip(self.kappa * dB_bar_dei, -1e4, 1e4)

        return B_bar, dB_bar_dei

    def compute_dB_bar_dt(self, ei, k_ie, Xi_dot):
        """Time derivative B̄_c_i'(t) = ∇B̄_c_i · Ẋ_i (numpy, scalar)."""
        _, dB = self.compute_B_bar(ei, k_ie)
        dB_dX       = np.zeros(self.state_dim)
        dB_dX[:3]   = dB
        Xi_dot      = np.clip(np.asarray(Xi_dot).flatten(), -1e4, 1e4)
        return float(np.clip(dB_dX @ Xi_dot, -self.LC_CLIP, self.LC_CLIP))

    # ──────────────────────────────────────────────────────────────────
    # Policy  (Eq. 36, 37)
    # ──────────────────────────────────────────────────────────────────

    def compute_policy(self, Xi, ei, k_ie, Gi, alpha_i, beta_bar_i):
        """
        Compute û_i (Eq. 36) and d̂_i (Eq. 37) on GPU.

        Returns
        -------
        u_hat : (3,) numpy
        d_hat : (6,) numpy
        """
        Xi_t = self._to_tensor(Xi, clip=self.IN_CLIP).flatten()  # (6,)

        # φ_i and ∂φ_i/∂X via autograd
        phi_t, dphi_t = self.net.basis_and_jacobian(Xi_t)    # (19,1), (19,6)

        # ∇B̄_c_i w.r.t. full state (numpy → tensor)
        _, dB_dei         = self.compute_B_bar(ei, k_ie)
        dB_dX_np          = np.zeros((1, self.state_dim), dtype=np.float32)
        dB_dX_np[0, :3]   = dB_dei.astype(np.float32)
        dB_dX_t           = self._to_tensor(dB_dX_np)         # (1, 6)

        # ∇V̂_ic = ŵ_i^T ∇φ_i + ∇B̄_c_i   shape (1, 6)
        # fc2.weight is ŵ_i^T, shape (1, 19)
        grad_V = self.net.fc2.weight @ dphi_t + dB_dX_t       # (1, 6)
        grad_V = grad_V.clamp(-1e4, 1e4)

        # Convert to numpy for control law (small 3D matrices — negligible cost)
        grad_V_np  = grad_V.detach().cpu().numpy()             # (1, 6)
        Gi_np      = np.asarray(Gi,         dtype=np.float64)
        alpha_np   = np.asarray(alpha_i,    dtype=np.float64)
        beta_np    = np.asarray(beta_bar_i, dtype=np.float64).flatten()
        alpha_inv  = np.linalg.inv(alpha_np)

        # Λ̂_i = 0.5 α_i⁻¹ G_i^T ∇V̂_ic^T   (3, 1)
        Lambda_hat = 0.5 * alpha_inv @ Gi_np.T @ grad_V_np.T
        Lambda_hat = np.clip(Lambda_hat, -10.0, 10.0)

        # û_i = −α_i tanh(Λ̂_i) + β̄_i
        u_hat = (-alpha_np @ np.tanh(Lambda_hat)
                 + beta_np.reshape(-1, 1))

        # d̂_i = (1 / 2δ²) ∇V̂_ic^T   (6,)
        d_hat = (1.0 / (2.0 * self.delta ** 2)) * grad_V_np.T
        d_hat = np.clip(d_hat, -1e3, 1e3)

        return u_hat.flatten(), d_hat.flatten()

    # ──────────────────────────────────────────────────────────────────
    # Hamiltonian error  (Eq. 39)
    # ──────────────────────────────────────────────────────────────────

    def compute_hamiltonian_error(self, Xi, ei, k_ie, u_hat, d_hat,
                                  Fi, Gi, alpha_i, beta_bar_i, Ri, Xi_dot):
        """
        Hamiltonian estimation error e_i(t) — Eq. 39.

        Returns
        -------
        error   : float
        phi_dot : (m, 1) numpy  ϕ_i(t) = ∇φ_i · ẋ
        """
        Xi_t = self._to_tensor(Xi, clip=self.IN_CLIP).flatten()

        _, dphi_t = self.net.basis_and_jacobian(Xi_t)          # (19, 6)
        dphi_np   = dphi_t.detach().cpu().numpy()              # (19, 6)

        Xi_dot_np = np.clip(
            np.asarray(Xi_dot, dtype=np.float64).flatten(),
            -1e4, 1e4
        )

        # ϕ_i(t) = ∇φ_i · ẋ   (19, 1)
        phi_dot_np = (dphi_np @ Xi_dot_np).reshape(-1, 1)
        phi_dot_np = np.clip(phi_dot_np, -1e4, 1e4)

        # L^c_i
        B_bar, _   = self.compute_B_bar(ei, k_ie)
        Lc         = self._compute_Lc(Xi, u_hat, d_hat, ei, k_ie,
                                      Ri, alpha_i, beta_bar_i, B_bar)
        B_bar_dot  = self.compute_dB_bar_dt(ei, k_ie, Xi_dot_np)

        # e_i(t) = ŵ_i^T ϕ_i(t) + L^c_i + B̄_c_i'
        w_np  = self.net.fc2.weight.detach().cpu().numpy()     # (1, 19)
        error = float(w_np @ phi_dot_np) + Lc + B_bar_dot
        error = float(np.clip(error, -self.ERR_CLIP, self.ERR_CLIP))

        return error, phi_dot_np

    def _compute_Lc(self, Xi, u_hat, d_hat, ei, k_ie,
                    Ri, alpha_i, beta_bar_i, B_bar):
        """Augmented reward L^c_i — Eq. 18."""
        Xi_f = np.clip(np.asarray(Xi).flatten(), -1e3, 1e3)
        sc   = float(Xi_f @ np.asarray(Ri) @ Xi_f)
        ic   = self._compute_Ui(u_hat, alpha_i, beta_bar_i)
        d_full = np.asarray(d_hat).flatten()
        dc = self.delta ** 2 * float(d_full @ d_full)
        Lc   = sc + ic + B_bar - dc
        return float(np.clip(Lc, -self.LC_CLIP, self.LC_CLIP))

    def _compute_Ui(self, ui, alpha_i, beta_bar_i):
        """Nonquadratic input penalty U_i — Eq. 16."""
        ui    = np.asarray(ui).flatten()
        alpha = np.diag(np.asarray(alpha_i))
        beta  = np.asarray(beta_bar_i).flatten()
        total = 0.0
        for s in range(3):
            a_s = alpha[s];  b_s = beta[s];  u_s = ui[s]
            tau  = np.clip((u_s - b_s) / a_s, -1 + 1e-7, 1 - 1e-7)
            total += 2.0 * a_s * (
                (u_s - b_s) * np.arctanh(tau)
                + (a_s / 2.0) * np.log(max(1.0 - tau ** 2, 1e-12))
            )
        return float(np.clip(total, -self.LC_CLIP, self.LC_CLIP))

    # ──────────────────────────────────────────────────────────────────
    # Weight update  (Eq. 42)  — fully on GPU
    # ──────────────────────────────────────────────────────────────────

    def update_weights(self, phi_dot_np, Lc, B_bar_dot):
        """
        Update critic weights via gradient descent with experience replay
        — Eq. 42.  The loss is the squared normalised Hamiltonian error
        summed over the current sample and all replay samples.

        Parameters
        ----------
        phi_dot_np  : (m, 1) numpy  ϕ_i(t)
        Lc          : float
        B_bar_dot   : float
        """
        phi_dot_np = np.clip(
            np.asarray(phi_dot_np, dtype=np.float32).reshape(-1, 1),
            -1e4, 1e4
        )
        phi_dot_t = self._to_tensor(phi_dot_np)                # (m, 1)

        Lc_t        = torch.tensor([[float(np.clip(Lc,        -self.LC_CLIP, self.LC_CLIP))]],
                                   dtype=torch.float32, device=self.device)
        B_bar_dot_t = torch.tensor([[float(np.clip(B_bar_dot, -self.LC_CLIP, self.LC_CLIP))]],
                                   dtype=torch.float32, device=self.device)

        self.optimiser.zero_grad()

        # ── Build loss from current sample + replay ────────────────────
        loss = 0.5 * self._hamiltonian_loss(phi_dot_t, Lc_t, B_bar_dot_t)

        if len(self.replay) >= self.mini_batch_size:
            phi_batch, Lc_batch, B_dot_batch = self.replay.sample(self.mini_batch_size)
            # phi_batch: (m, B),  Lc_batch: (1, B)

            # ŵ_i^T ϕ_i(t_p)   →   (1, B)
            w_row   = self.net.fc2.weight                      # (1, m)
            e_batch = (w_row @ phi_batch                       # (1, B)
                       + Lc_batch + B_dot_batch)

            # Normalise: ϕ̄ = ϕ / (ϕ^T ϕ + 1)
            denom_b  = (phi_batch * phi_batch).sum(dim=0, keepdim=True) + 1.0  # (1, B)
            phi_bar  = phi_batch / denom_b                     # (m, B)

            # Gradient contribution: Σ_p ϕ̄_p · e_p / denom_p  →  mean over batch
            normalized_e_batch = e_batch / denom_b
            replay_loss = 0.5 * (normalized_e_batch ** 2).sum()
            loss = loss + replay_loss

        # ── Backprop + gradient clip + step ───────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.GRAD_CLIP)
        self.optimiser.step()

        # ── Sync numpy .w for external logging ────────────────────────
        self._sync_w()

        # Store current sample in replay buffer
        self.replay.push(phi_dot_t, float(Lc_t), float(B_bar_dot_t))

    def _hamiltonian_loss(self, phi_dot_t, Lc_t, B_bar_dot_t):
        """
        Scalar loss for a single sample — normalised squared Hamiltonian error.

            L = (ŵ^T ϕ̄ + L^c/denom + B̄'/denom)²

        This drives ŵ so that Ĥ → 0  (Eq. 41).
        """
        denom   = (phi_dot_t * phi_dot_t).sum() + 1.0         # scalar
        phi_bar = phi_dot_t / denom                            # (m, 1)

        # ŵ^T ϕ̄   using fc2 (the linear output layer weights)
        w_row   = self.net.fc2.weight                          # (1, m)
        e_t     = (w_row @ phi_bar                             # (1, 1)
                   + Lc_t / denom
                   + B_bar_dot_t / denom)

        return (e_t ** 2).squeeze()

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────

    def get_value_estimate(self, Xi, ei, k_ie):
        """V̂_ic(X_i) = ŵ_i^T φ_i(X_i) + B̄_c_i(X_i)  — Eq. 35."""
        Xi_t    = self._to_tensor(Xi, clip=self.IN_CLIP).unsqueeze(0)  # (1, 6)
        V_net   = self.net(Xi_t).item()
        B_bar,_ = self.compute_B_bar(ei, k_ie)
        return V_net + B_bar

    def check_rank_condition(self, min_samples: int = 19):
        """Verify Assumption 4 : rank(Π_i) = m."""
        if len(self.replay) < min_samples:
            return False, len(self.replay)
        items  = list(self.replay.buf)[-min_samples:]
        Phi    = torch.cat([e[0] for e in items], dim=1)      # (m, l)
        rank   = torch.linalg.matrix_rank(Phi).item()
        return rank == self.m, rank