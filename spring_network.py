"""
Act 2 substrate: 2D spring network with trainable (k, ell) per edge.

This is the Stern, Hexner, Rocks, Liu 2021 spring network — the nonlinear
counterpart to the Act 1 resistor mesh. The substrate's expressivity escapes
the linear regime because the spring length r_e = ||x_i - x_j|| is a
nonlinear function of node coordinates, even though each edge's energy is
still harmonic in strain.

Structure (mirrors MeshCoupledSubstrate where possible):
  - rows x cols grid of nodes; initial positions on a unit grid (col, -row).
  - 4-neighbor connectivity (horizontal + vertical springs only).
  - Two trainable parameters per edge:
      k_e   (spring constant, > k_min)
      ell_e (rest length,     > ell_min)
    Both updated by the same per-edge coupled-learning interface as all
    other parameters in the project.

I/O conventions:
  - Source nodes: left column at rows ``source_rows[k]`` for ``k = 0 .. n_input-1``
    (see ``_default_source_rows`` when omitted). Node ``(source_rows[k], 0)`` is
    clamped to ``(0, -source_rows[k] + x[k])`` in mesh coordinates.
  - Output: differential y-displacement from rest at two designated
    right-boundary nodes,
        y_pred = (pos[out_pos] - rest_pos[out_pos]).y
               - (pos[out_neg] - rest_pos[out_neg]).y
    This is 0 when the network is at rest, so signed targets are natural.
  - All non-source nodes are free in 2D.

Forward = energy minimization over free node coordinates, solved with
L-BFGS-B (quasi-Newton) using analytical gradients. The energy landscape is
strictly convex near the rest configuration, so initializing free nodes at
their rest positions converges quickly. The clamped phase warm-starts from
the free-phase equilibrium since the eta-nudge is a small perturbation.

Backward (``backward_mode``):
  - ``adjoint`` (default): gradient of ``L = (1/2) * mean((y_pred - y)²)``
    through the free elastic equilibrium via a dense analytic Hessian
    ``∂²E/∂z²`` (central-force blocks) plus ``hessian_ridge`` on the diagonal
    so the linearized system stays positive definite after approximate L-BFGS
    convergence. Per-batch cost is a ``2 n_free`` linear solve and edge-local
    dot products.
  - ``fd``: central differences on every ``(k, ℓ)`` with full re-forward — a
    gold-standard check (``spring_grad_check.py``) at
    ``O(n_params × batch × forward)``.

The old squared-strain ``(r^C-ℓ)² - (r^F-ℓ)²`` shortcut was **not** equivalent
to ``∂L/∂θ`` here: Stern et al. (PRX 2021) Eq.~6 uses explicit ``∂_θ(E^F-E^C)``
with the **same** boundary variables that appear in ``E``; a scalar output
nudge on node displacements is a different protocol than their edge-strain
targets, and for nonlinear springs it no longer matches finite-difference
``∇L`` (see ``spring_grad_check.py`` with ``backward_mode='fd'``).
"""
import numpy as np
from scipy.optimize import minimize


class SpringNetworkSubstrate:
    @staticmethod
    def _default_source_rows(rows, n_input):
        """Pick non-consecutive interior rows on column 0 when none given."""
        interior = list(range(1, rows - 1))
        if len(interior) < n_input:
            raise ValueError(
                f"rows={rows} admits only {len(interior)} interior source rows "
                f"(need {n_input}); use a taller grid or pass source_rows=..."
            )
        chosen = []
        for r in interior:
            if len(chosen) == n_input:
                break
            if chosen and r == chosen[-1] + 1:
                continue
            chosen.append(r)
        if len(chosen) < n_input:
            chosen = interior[:n_input]
        return chosen

    def __init__(self, rows=3, cols=4, source_rows=None, n_input=None,
                 out_pos_row=1, out_neg_row=2, eta=0.05,
                 k_init=1.0, ell_init=1.0, init_std=0.05,
                 k_min=0.01, ell_min=0.05, seed=0,
                 backward_mode="adjoint", fd_eps=1e-5,
                 hessian_ridge=1e-5):
        # Source nodes live on column 0. Each must be a strictly interior row
        # (1 .. rows-2) so the node has vertical grid neighbors; otherwise a
        # vertical spring is pinned at a boundary and the topology degenerates.
        if source_rows is None:
            if n_input is None:
                n_input = 2
            source_rows = self._default_source_rows(rows, n_input)
        source_rows = list(source_rows)
        if n_input is None:
            n_input = len(source_rows)
        if n_input != len(source_rows):
            raise ValueError("n_input must equal len(source_rows)")
        for r in source_rows:
            if not (0 <= r < rows):
                raise ValueError(f"source row {r} out of range [0, {rows})")
            if r <= 0 or r >= rows - 1:
                raise ValueError(
                    f"source row {r} must be strictly interior "
                    f"(1 <= row <= {rows - 2}) so vertical springs are not "
                    "degenerate at the mesh boundary"
                )
        if len(set(source_rows)) != len(source_rows):
            raise ValueError("source rows must be distinct")
        if not (0 <= out_pos_row < rows and 0 <= out_neg_row < rows):
            raise ValueError("output rows must be in range [0, rows)")
        if out_pos_row == out_neg_row:
            raise ValueError("output rows must differ")

        self.rows = rows
        self.cols = cols
        self.n_input = n_input
        self.source_rows = source_rows
        self.out_pos_row = out_pos_row
        self.out_neg_row = out_neg_row
        self.eta = eta
        self.k_min = k_min
        self.ell_min = ell_min
        if backward_mode not in ("adjoint", "fd"):
            raise ValueError("backward_mode must be 'adjoint' or 'fd'")
        self.backward_mode = backward_mode
        self.fd_eps = fd_eps
        self.hessian_ridge = hessian_ridge

        rng = np.random.default_rng(seed)

        # Initial node positions on a unit grid; matches MeshCoupledSubstrate's
        # visualization convention so plotting code can be reused.
        self.rest_pos = np.array(
            [[c, -r] for r in range(rows) for c in range(cols)],
            dtype=float,
        )
        self.n_nodes = rows * cols

        # Trainable parameters: spring constants and rest lengths per edge.
        # Layout matches MeshCoupledSubstrate (k_h, k_v) then appended
        # (ell_h, ell_v). The doubled list makes learning rules iterate over
        # all four with no change.
        self.weights = [
            np.clip(k_init + init_std * rng.standard_normal((rows, cols - 1)),
                    k_min, None),
            np.clip(k_init + init_std * rng.standard_normal((rows - 1, cols)),
                    k_min, None),
            # Initial rest length 1.0 matches the unit-spaced grid, so the
            # untrained network starts at rest with zero strain everywhere.
            np.clip(ell_init + init_std * rng.standard_normal((rows, cols - 1)),
                    ell_min, None),
            np.clip(ell_init + init_std * rng.standard_normal((rows - 1, cols)),
                    ell_min, None),
        ]

        self.edges_h = [(self._idx(r, c), self._idx(r, c + 1))
                        for r in range(rows) for c in range(cols - 1)]
        self.edges_v = [(self._idx(r, c), self._idx(r + 1, c))
                        for r in range(rows - 1) for c in range(cols)]

        self.source_nodes = [self._idx(r, 0) for r in self.source_rows]
        self._source_set = frozenset(self.source_nodes)
        self.out_pos_node = self._idx(out_pos_row, cols - 1)
        self.out_neg_node = self._idx(out_neg_row, cols - 1)

        clamped = set(self.source_nodes)
        self.free_nodes = [i for i in range(self.n_nodes) if i not in clamped]
        self.n_free = len(self.free_nodes)
        self.global_to_free = {gi: fi for fi, gi in enumerate(self.free_nodes)}
        self.out_pos_free = self.global_to_free[self.out_pos_node]
        self.out_neg_free = self.global_to_free[self.out_neg_node]

        # Cached arrays for vectorized energy / force evaluation.
        self._edges_h_arr = np.array(self.edges_h, dtype=int)
        self._edges_v_arr = np.array(self.edges_v, dtype=int)

    def _idx(self, r, c):
        return r * self.cols + c

    def _build_full_position(self, free_pos, x_input):
        all_pos = self.rest_pos.copy()
        # Sources: x stays at column 0 rest, y displaced from rest by input.
        for i, node in enumerate(self.source_nodes):
            all_pos[node, 1] = self.rest_pos[node, 1] + x_input[i]
        for fi, node in enumerate(self.free_nodes):
            all_pos[node, :] = free_pos[fi]
        return all_pos

    def _energy_and_grad(self, free_pos_flat, x_input, target):
        """Total energy and gradient w.r.t. free node positions.

        target=None: free phase (just elastic energy).
        target=scalar: clamped phase (adds (eta/2)*(y_pred - target)**2
        nudge term).
        """
        free_pos = free_pos_flat.reshape(self.n_free, 2)
        all_pos = self._build_full_position(free_pos, x_input)

        Wk_h = np.maximum(self.weights[0], self.k_min)
        Wk_v = np.maximum(self.weights[1], self.k_min)
        Well_h = np.maximum(self.weights[2], self.ell_min)
        Well_v = np.maximum(self.weights[3], self.ell_min)

        # Vectorized spring energy + per-node force accumulation.
        force = np.zeros_like(all_pos)
        energy = 0.0

        for edges_arr, Wk, Well in (
            (self._edges_h_arr, Wk_h, Well_h),
            (self._edges_v_arr, Wk_v, Well_v),
        ):
            if edges_arr.shape[0] == 0:
                continue
            i = edges_arr[:, 0]
            j = edges_arr[:, 1]
            d = all_pos[i] - all_pos[j]            # (n_edges, 2)
            r = np.linalg.norm(d, axis=1)          # (n_edges,)
            ke = Wk.ravel()
            le = Well.ravel()
            strain = r - le
            energy += float(0.5 * np.sum(ke * strain ** 2))

            # Force on i: -dE/dx_i = -k*strain*d/r; opposite on j.
            r_safe = np.maximum(r, 1e-12)[:, None]
            fij = (ke * strain)[:, None] * d / r_safe
            np.add.at(force, i, -fij)
            np.add.at(force, j, +fij)

        if target is not None:
            y_pos_disp = (all_pos[self.out_pos_node, 1]
                          - self.rest_pos[self.out_pos_node, 1])
            y_neg_disp = (all_pos[self.out_neg_node, 1]
                          - self.rest_pos[self.out_neg_node, 1])
            y_pred = y_pos_disp - y_neg_disp
            energy += 0.5 * self.eta * (y_pred - target) ** 2
            # d/dy_pos of (eta/2)(y_pred - target)^2 = eta * (y_pred - target)
            # Force = -gradient, applied on the free-node force array.
            d_loss = self.eta * (y_pred - target)
            force[self.out_pos_node, 1] -= d_loss
            force[self.out_neg_node, 1] += d_loss

        # Gradient w.r.t. free-node positions = -force on those nodes.
        grad = np.zeros((self.n_free, 2))
        for fi, node in enumerate(self.free_nodes):
            grad[fi] = -force[node]
        return energy, grad.ravel()

    def _equilibrate(self, x_input, target=None, init_free=None):
        if init_free is None:
            init_free = np.array([self.rest_pos[node]
                                  for node in self.free_nodes])
        result = minimize(
            self._energy_and_grad,
            init_free.ravel(),
            args=(x_input, target),
            method="L-BFGS-B",
            jac=True,
            options={"gtol": 1e-9, "ftol": 1e-12, "maxiter": 2000},
        )
        free_pos_eq = result.x.reshape(self.n_free, 2)
        all_pos_eq = self._build_full_position(free_pos_eq, x_input)
        return all_pos_eq, free_pos_eq, result

    def _compute_y_pred(self, all_pos):
        y_pos_disp = (all_pos[self.out_pos_node, 1]
                      - self.rest_pos[self.out_pos_node, 1])
        y_neg_disp = (all_pos[self.out_neg_node, 1]
                      - self.rest_pos[self.out_neg_node, 1])
        return y_pos_disp - y_neg_disp

    def _hessian_elastic_dense(self, z_flat, x_input):
        """Dense ∂²E/∂z² for elastic energy (central-force springs)."""
        all_pos = self._build_full_position(
            z_flat.reshape(self.n_free, 2), x_input
        )
        n = 2 * self.n_free
        H = np.zeros((n, n))
        Wk_h = np.maximum(self.weights[0], self.k_min)
        Wk_v = np.maximum(self.weights[1], self.k_min)
        Well_h = np.maximum(self.weights[2], self.ell_min)
        Well_v = np.maximum(self.weights[3], self.ell_min)

        for edges_arr, Wk, Well in (
            (self._edges_h_arr, Wk_h, Well_h),
            (self._edges_v_arr, Wk_v, Well_v),
        ):
            if edges_arr.shape[0] == 0:
                continue
            for e in range(edges_arr.shape[0]):
                gi, gj = int(edges_arr[e, 0]), int(edges_arr[e, 1])
                k_e = float(Wk.ravel()[e])
                ell_e = float(Well.ravel()[e])
                self._accumulate_edge_hessian(H, gi, gj, all_pos, k_e, ell_e)

        H[np.diag_indices(n)] += self.hessian_ridge
        return H

    def _accumulate_edge_hessian(self, H, gi, gj, all_pos, k_e, ell_e):
        """Add one spring's contribution to H (free coordinates only)."""
        pi = all_pos[gi]
        pj = all_pos[gj]
        d = pi - pj
        r = float(np.linalg.norm(d))
        r = max(r, 1e-12)
        q = d / r
        I2 = np.eye(2)
        strain = r - ell_e
        s = k_e * strain
        B = k_e * np.outer(q, q) + (s / r) * (I2 - np.outer(q, q))
        idx_i = [self._dof_flat_index(gi, c) for c in (0, 1)]
        idx_j = [self._dof_flat_index(gj, c) for c in (0, 1)]
        free_i = idx_i[0] is not None
        free_j = idx_j[0] is not None
        if free_i and free_j:
            idx = np.array(idx_i + idx_j, dtype=int)
            H[np.ix_(idx, idx)] += np.block([[B, -B], [-B, B]])
        elif free_i and not free_j:
            idx = np.array(idx_i, dtype=int)
            H[np.ix_(idx, idx)] += B
        elif free_j and not free_i:
            idx = np.array(idx_j, dtype=int)
            H[np.ix_(idx, idx)] += B

    def _dof_flat_index(self, global_node, xy_component):
        """Map global node + {0=x,1=y} to flat z index, or None if source."""
        if global_node in self._source_set:
            return None
        fi = self.global_to_free[global_node]
        return 2 * fi + xy_component

    def _scatter_v_param_edge(self, gi, gj, all_pos, k_e, ell_e, which):
        """Sparse column v = ∂(∇E)/∂θ for one edge; which in {'k', 'ell'}."""
        n = 2 * self.n_free
        v = np.zeros(n)
        pi = all_pos[gi]
        pj = all_pos[gj]
        d = pi - pj
        r = float(np.linalg.norm(d))
        r = max(r, 1e-12)
        q = d / r
        strain = r - ell_e
        if which == "k":
            vi = strain * q
            vj = -strain * q
        else:
            vi = -k_e * q
            vj = k_e * q
        for comp in (0, 1):
            idx_i = self._dof_flat_index(gi, comp)
            if idx_i is not None:
                v[idx_i] += vi[comp]
            idx_j = self._dof_flat_index(gj, comp)
            if idx_j is not None:
                v[idx_j] += vj[comp]
        return v

    def forward(self, x):
        """Batched forward. x: (batch, n_input). Returns (y_pred, cache)."""
        batch_size = x.shape[0]
        y_pred = np.zeros((batch_size, 1))
        all_positions_F = np.zeros((batch_size, self.n_nodes, 2))
        free_positions_F = np.zeros((batch_size, self.n_free, 2))
        for b in range(batch_size):
            all_pos, free_pos, _ = self._equilibrate(x[b])
            all_positions_F[b] = all_pos
            free_positions_F[b] = free_pos
            y_pred[b, 0] = self._compute_y_pred(all_pos)
        cache = {
            "x": x,
            "all_positions_F": all_positions_F,
            "free_positions_F": free_positions_F,
        }
        return y_pred, cache

    def backward(self, cache, y_pred, y):
        """Gradient of ``L = (1/2) * mean((y_pred - y)**2)`` through free equilibrium."""
        if self.backward_mode == "fd":
            return self._backward_fd(cache, y_pred, y)
        return self._backward_adjoint(cache, y_pred, y)

    def _backward_fd(self, cache, y_pred, y):
        x = cache["x"]
        h = self.fd_eps

        def loss_after_forward():
            yp, _ = self.forward(x)
            return float(0.5 * np.mean((yp - y) ** 2))

        grads = []
        for W in self.weights:
            g = np.zeros_like(W)
            flat = W.ravel()
            gflat = g.ravel()
            for k in range(flat.size):
                saved = flat[k]
                flat[k] = saved + h
                lp = loss_after_forward()
                flat[k] = saved - h
                lm = loss_after_forward()
                flat[k] = saved
                gflat[k] = (lp - lm) / (2.0 * h)
            grads.append(g)
        return grads

    def _backward_adjoint(self, cache, y_pred, y):
        x = cache["x"]
        batch_size = x.shape[0]
        free_positions_F = cache["free_positions_F"]
        all_positions_F = cache["all_positions_F"]

        g_k_h = np.zeros_like(self.weights[0])
        g_k_v = np.zeros_like(self.weights[1])
        g_ell_h = np.zeros_like(self.weights[2])
        g_ell_v = np.zeros_like(self.weights[3])

        Wk_h = np.maximum(self.weights[0], self.k_min)
        Wk_v = np.maximum(self.weights[1], self.k_min)
        Well_h = np.maximum(self.weights[2], self.ell_min)
        Well_v = np.maximum(self.weights[3], self.ell_min)

        for b in range(batch_size):
            z_flat = free_positions_F[b].ravel()
            xb = x[b]
            all_b = all_positions_F[b]
            H = self._hessian_elastic_dense(z_flat, xb)
            coef = (y_pred[b, 0] - y[b, 0]) / float(batch_size)
            gL = np.zeros(2 * self.n_free)
            gL[2 * self.out_pos_free + 1] = coef
            gL[2 * self.out_neg_free + 1] = -coef
            p = np.linalg.solve(H, gL)

            for k, (gi, gj) in enumerate(self.edges_h):
                r, c = divmod(k, self.cols - 1)
                kk = float(Wk_h[r, c])
                el = float(Well_h[r, c])
                vk = self._scatter_v_param_edge(gi, gj, all_b, kk, el, "k")
                vell = self._scatter_v_param_edge(gi, gj, all_b, kk, el, "ell")
                g_k_h[r, c] -= float(np.dot(p, vk))
                g_ell_h[r, c] -= float(np.dot(p, vell))

            for k, (gi, gj) in enumerate(self.edges_v):
                r, c = divmod(k, self.cols)
                kk = float(Wk_v[r, c])
                el = float(Well_v[r, c])
                vk = self._scatter_v_param_edge(gi, gj, all_b, kk, el, "k")
                vell = self._scatter_v_param_edge(gi, gj, all_b, kk, el, "ell")
                g_k_v[r, c] -= float(np.dot(p, vk))
                g_ell_v[r, c] -= float(np.dot(p, vell))

        return [g_k_h, g_k_v, g_ell_h, g_ell_v]

    def project_weights(self):
        self.weights[0] = np.maximum(self.weights[0], self.k_min)
        self.weights[1] = np.maximum(self.weights[1], self.k_min)
        self.weights[2] = np.maximum(self.weights[2], self.ell_min)
        self.weights[3] = np.maximum(self.weights[3], self.ell_min)

    def copy_params(self):
        return [W.copy() for W in self.weights]

    def set_params(self, params):
        self.weights = [W.copy() for W in params]

    @property
    def num_params(self):
        return sum(W.size for W in self.weights)
