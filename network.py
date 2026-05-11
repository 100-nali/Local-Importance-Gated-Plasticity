"""
Two substrates with the same per-edge interface, so the same learning rules
and metrics can be run on either.

  ANNSubstrate (alias: GraphNetwork)
      Layered linear feedforward network. Forward = matrix multiplication.
      Backward = standard backprop. Per-edge "gradient" is the chain-rule
      derivative dL/dw_e. This is what was used in the first round of
      experiments and is, honestly, an ANN.

  CoupledGraphSubstrate
      Edge-coupled physical network with linear couplings (resistor-network
      analogue). Same layered topology and same number of trainable edges,
      but with a different forward/backward mechanism:

        Forward (free phase): inputs are clamped boundary nodes; hidden and
          output nodes are free degrees of freedom. Solve for the equilibrium
          configuration that minimizes the network energy
              E = (1/2) sum_e w_e (v_i - v_j)^2
          which reduces to a linear system L v_free = f over the free nodes,
          where L is the weighted graph Laplacian.

        Backward (coupled / contrastive learning): solve a second equilibrium
          with the output additionally nudged toward the target with small
          strength eta. The per-edge "force" — the only signal each edge
          ever sees — is the difference in its own per-edge energy
          contribution between the two equilibria, divided by eta:
              F_e = (1/(2 eta)) * [(v_i^C - v_j^C)^2 - (v_i^F - v_j^F)^2]
          In the eta -> 0 limit this is the exact gradient of MSE with
          respect to w_e. No chain rule, no backprop — each edge sees only
          the potentials at its two endpoints in the free and clamped
          equilibria.

The two substrates are not the same parameterization of input -> output:
the ANN computes y = (x W1) W2; the coupled substrate computes y as the
output-node potential of the equilibrium of the linear coupled system,
which is a rational function of W1 and W2. Loss landscapes therefore differ,
and the per-edge signal seen by each learning rule has different statistics.
"""
import numpy as np


class GraphNetwork:
    def __init__(self, layer_sizes, seed=0):
        self.layer_sizes = list(layer_sizes)
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.standard_normal((a, b)) * np.sqrt(1.0 / a)
            for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    def forward(self, x):
        acts = [x]
        h = x
        for W in self.weights:
            h = h @ W
            acts.append(h)
        return h, acts

    def backward(self, acts, y_pred, y):
        n = max(len(y), 1)
        delta = (y_pred - y) / n
        grads = [None] * len(self.weights)
        for i in reversed(range(len(self.weights))):
            grads[i] = acts[i].T @ delta
            delta = delta @ self.weights[i].T
        return grads

    def copy_params(self):
        return [W.copy() for W in self.weights]

    def set_params(self, params):
        self.weights = [W.copy() for W in params]

    @property
    def num_params(self):
        return sum(W.size for W in self.weights)


# Honest alias — the class above is an ANN.
ANNSubstrate = GraphNetwork


class CoupledGraphSubstrate:
    def __init__(self, layer_sizes, seed=0, eta=0.05, ridge=1e-4):
        if len(layer_sizes) != 3:
            raise ValueError(
                "CoupledGraphSubstrate currently supports a single hidden layer "
                "(3-layer topology); got layer_sizes=%r" % (layer_sizes,)
            )
        self.layer_sizes = list(layer_sizes)
        self.eta = eta
        # Tiny diagonal ridge added to L to guarantee invertibility under
        # signed couplings. Physically: a weak coupling of every free node to
        # ground; biases the output toward zero by O(ridge), which is
        # negligible for small ridge.
        self.ridge = ridge
        rng = np.random.default_rng(seed)
        # Signed couplings, Xavier-style scale. Strictly-positive conductance
        # would restrict the realizable input->output map to the positive
        # orthant of coefficient space, which can't fit signed regression
        # targets. Signed couplings correspond to e.g. mechanical networks
        # where springs can act in compression or tension.
        self.weights = [
            rng.standard_normal((a, b)) * np.sqrt(1.0 / a)
            for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    def _build_laplacian(self, W):
        n_hid = self.layer_sizes[1]
        n_out = self.layer_sizes[2]
        n_free = n_hid + n_out
        L = np.zeros((n_free, n_free))
        W1 = W[0]  # (n_in, n_hid)
        W2 = W[1]  # (n_hid, n_out)
        # Hidden-node row: sum_j w_ij coefficient on v_i.
        L[np.arange(n_hid), np.arange(n_hid)] = W1.sum(axis=0) + W2.sum(axis=1)
        # Output-node row: sum_h w_ho coefficient on v_o.
        L[n_hid + np.arange(n_out), n_hid + np.arange(n_out)] = W2.sum(axis=0)
        # Cross terms.
        L[:n_hid, n_hid:] = -W2
        L[n_hid:, :n_hid] = -W2.T
        # Tiny ridge (weak coupling to ground) for invertibility under signed W.
        L[np.arange(n_free), np.arange(n_free)] += self.ridge
        return L

    def _input_drive(self, x, W1):
        n_hid = self.layer_sizes[1]
        n_out = self.layer_sizes[2]
        f = np.zeros((n_hid + n_out, x.shape[0]))
        f[:n_hid, :] = W1.T @ x.T
        return f

    def forward(self, x):
        L = self._build_laplacian(self.weights)
        f = self._input_drive(x, self.weights[0])
        v_free = np.linalg.solve(L, f).T  # (batch, n_free)
        n_out = self.layer_sizes[2]
        y_pred = v_free[:, -n_out:].copy()
        cache = {"x": x, "v_free": v_free, "L": L}
        return y_pred, cache

    def backward(self, cache, y_pred, y):
        x = cache["x"]
        v_free = cache["v_free"]
        n_hid = self.layer_sizes[1]
        n_out = self.layer_sizes[2]
        # Clamped phase: nudge the output toward y with strength eta. For a
        # quadratic loss this is equivalent to adding eta * I to the output
        # diagonal of L and eta * y to the output entries of the bias.
        L_c = cache["L"].copy()
        idx_out = n_hid + np.arange(n_out)
        L_c[idx_out, idx_out] += self.eta
        f_c = self._input_drive(x, self.weights[0])
        f_c[n_hid:, :] = self.eta * y.T
        v_c = np.linalg.solve(L_c, f_c).T

        h_F = v_free[:, :n_hid]
        h_C = v_c[:, :n_hid]
        o_F = v_free[:, n_hid:]
        o_C = v_c[:, n_hid:]

        # Per-edge gradient: g_e ≈ ∂L/∂w_e in the eta -> 0 limit.
        # g_e = (1/(2 eta)) * mean_b [(v_i^C - v_j^C)^2 - (v_i^F - v_j^F)^2]
        d_C1 = x[:, :, None] - h_C[:, None, :]    # (B, n_in, n_hid)
        d_F1 = x[:, :, None] - h_F[:, None, :]
        g_W1 = (d_C1 ** 2 - d_F1 ** 2).mean(axis=0) / (2.0 * self.eta)

        d_C2 = h_C[:, :, None] - o_C[:, None, :]  # (B, n_hid, n_out)
        d_F2 = h_F[:, :, None] - o_F[:, None, :]
        g_W2 = (d_C2 ** 2 - d_F2 ** 2).mean(axis=0) / (2.0 * self.eta)

        return [g_W1, g_W2]

    def copy_params(self):
        return [W.copy() for W in self.weights]

    def set_params(self, params):
        self.weights = [W.copy() for W in params]

    @property
    def num_params(self):
        return sum(W.size for W in self.weights)


class MeshCoupledSubstrate:
    """
    2D mesh of coupled nodes with positive (passive) conductances —
    the canonical substrate from the physical-learning literature
    (resistor network, Stern et al. 2022, Anisetti et al. 2024, etc.).

    Topology:
      - rows × cols grid of nodes, indexed (r, c) with global index r*cols+c
      - 4-neighbor connectivity (N, S, E, W) only
      - Boundary clamps:
          * Inputs:  leftmost column, rows 0..n_input-1, clamped to x[i]
          * Outputs: rightmost column, two designated rows form a
            differential pair: y_pred = v[out_pos] - v[out_neg]
          * All other nodes (interior + remaining boundary) are free
      - Differential output is what lets a positive-conductance system fit
        signed targets without resorting to signed couplings.

    Trainable parameters: edge conductances stored in two 2D arrays
        W_h: (rows, cols-1)    horizontal edges
        W_v: (rows-1, cols)    vertical edges
    Total trainable edges: rows*(cols-1) + (rows-1)*cols.
    With rows=8, cols=10 this is 142 — close to the layered 144.

    Conductances are positive: clipped to >= w_min when constructing the
    Laplacian. The raw weight in self.weights is left unmodified by the
    physics so the rule sees what it "tried" to do; project_weights()
    syncs raw to physical and is called by the train loop after each step.

    Forward = solve Kirchhoff equilibrium given input clamps.
    Backward = contrastive coupled learning, identical mathematics to
    CoupledGraphSubstrate; only the topology and boundary conditions differ.
    """

    def __init__(self, rows=8, cols=10, n_input=8,
                 out_pos_row=3, out_neg_row=4, eta=0.05,
                 init_mean=0.5, init_std=0.05, w_min=0.05, seed=0):
        if n_input > rows:
            raise ValueError("n_input must be <= rows")
        if not (0 <= out_pos_row < rows and 0 <= out_neg_row < rows):
            raise ValueError("output rows must be in range [0, rows)")
        if out_pos_row == out_neg_row:
            raise ValueError("output rows must differ")

        self.rows = rows
        self.cols = cols
        self.n_input = n_input
        self.out_pos_row = out_pos_row
        self.out_neg_row = out_neg_row
        self.eta = eta
        self.w_min = w_min

        rng = np.random.default_rng(seed)
        # Source of truth is self.weights — referenced everywhere via index
        # so a rule that reassigns network.weights[i] is honored. Caching
        # W_h / W_v as separate attributes would break that contract.
        self.weights = [
            np.clip(init_mean + init_std * rng.standard_normal((rows, cols - 1)),
                    w_min, None),
            np.clip(init_mean + init_std * rng.standard_normal((rows - 1, cols)),
                    w_min, None),
        ]

        self.n_nodes = rows * cols
        self.input_global = [self._idx(r, 0) for r in range(n_input)]
        self.output_pos_global = self._idx(out_pos_row, cols - 1)
        self.output_neg_global = self._idx(out_neg_row, cols - 1)

        clamped = set(self.input_global)
        self.free_global = [i for i in range(self.n_nodes) if i not in clamped]
        self.n_free = len(self.free_global)
        self.global_to_free = {gi: fi for fi, gi in enumerate(self.free_global)}
        self.out_pos_free = self.global_to_free[self.output_pos_global]
        self.out_neg_free = self.global_to_free[self.output_neg_global]

        self.edges_h = [(self._idx(r, c), self._idx(r, c + 1))
                        for r in range(rows) for c in range(cols - 1)]
        self.edges_v = [(self._idx(r, c), self._idx(r + 1, c))
                        for r in range(rows - 1) for c in range(cols)]

        # Spatial node positions for visualization (col -> x, -row -> y).
        self.node_positions = np.array(
            [[c, -r] for r in range(rows) for c in range(cols)],
            dtype=float,
        )

    def _idx(self, r, c):
        return r * self.cols + c

    def _build_full_laplacian(self):
        N = self.n_nodes
        L = np.zeros((N, N))
        Wh = np.maximum(self.weights[0], self.w_min)
        Wv = np.maximum(self.weights[1], self.w_min)
        for k, (i, j) in enumerate(self.edges_h):
            r, c = divmod(k, self.cols - 1)
            w = Wh[r, c]
            L[i, i] += w; L[j, j] += w
            L[i, j] -= w; L[j, i] -= w
        for k, (i, j) in enumerate(self.edges_v):
            r, c = divmod(k, self.cols)
            w = Wv[r, c]
            L[i, i] += w; L[j, j] += w
            L[i, j] -= w; L[j, i] -= w
        return L

    def forward(self, x):
        L_full = self._build_full_laplacian()
        L_ff = L_full[np.ix_(self.free_global, self.free_global)]
        L_fc = L_full[np.ix_(self.free_global, self.input_global)]
        v_clamped = x.T  # (n_input, batch)
        rhs = -L_fc @ v_clamped
        v_free = np.linalg.solve(L_ff, rhs).T  # (batch, n_free)
        y_pred = (v_free[:, self.out_pos_free]
                  - v_free[:, self.out_neg_free]).reshape(-1, 1)
        cache = {"x": x, "v_free": v_free, "L_ff": L_ff, "L_fc": L_fc}
        return y_pred, cache

    def backward(self, cache, y_pred, y):
        x = cache["x"]
        v_free_F = cache["v_free"]
        # Clamped phase: nudge v[out_pos]-v[out_neg] toward y with strength eta.
        # Loss aug (eta/2)(v_pos - v_neg - y)^2 -> Hessian eta * e e^T (e = e_pos - e_neg)
        # and bias eta * y * e on the free-node coordinates.
        L_ff_C = cache["L_ff"].copy()
        L_ff_C[self.out_pos_free, self.out_pos_free] += self.eta
        L_ff_C[self.out_neg_free, self.out_neg_free] += self.eta
        L_ff_C[self.out_pos_free, self.out_neg_free] -= self.eta
        L_ff_C[self.out_neg_free, self.out_pos_free] -= self.eta
        rhs_C = -cache["L_fc"] @ x.T
        rhs_C[self.out_pos_free, :] += self.eta * y.ravel()
        rhs_C[self.out_neg_free, :] -= self.eta * y.ravel()
        v_free_C = np.linalg.solve(L_ff_C, rhs_C).T

        v_all_F = self._reconstruct(v_free_F, x)
        v_all_C = self._reconstruct(v_free_C, x)

        g_W_h = np.zeros_like(self.weights[0])
        for k, (i, j) in enumerate(self.edges_h):
            r, c = divmod(k, self.cols - 1)
            d_F = v_all_F[:, i] - v_all_F[:, j]
            d_C = v_all_C[:, i] - v_all_C[:, j]
            g_W_h[r, c] = float(np.mean(d_C ** 2 - d_F ** 2)) / (2.0 * self.eta)
        g_W_v = np.zeros_like(self.weights[1])
        for k, (i, j) in enumerate(self.edges_v):
            r, c = divmod(k, self.cols)
            d_F = v_all_F[:, i] - v_all_F[:, j]
            d_C = v_all_C[:, i] - v_all_C[:, j]
            g_W_v[r, c] = float(np.mean(d_C ** 2 - d_F ** 2)) / (2.0 * self.eta)
        return [g_W_h, g_W_v]

    def _reconstruct(self, v_free, x):
        v_all = np.zeros((x.shape[0], self.n_nodes))
        for fi, gi in enumerate(self.free_global):
            v_all[:, gi] = v_free[:, fi]
        for i, gi in enumerate(self.input_global):
            v_all[:, gi] = x[:, i]
        return v_all

    def project_weights(self):
        """Clip raw weights to physical positive range (call after rule step).
        Uses np.maximum on a fresh array to support rules that reassign the
        list slot rather than updating in place."""
        self.weights[0] = np.maximum(self.weights[0], self.w_min)
        self.weights[1] = np.maximum(self.weights[1], self.w_min)

    def copy_params(self):
        return [W.copy() for W in self.weights]

    def set_params(self, params):
        self.weights = [W.copy() for W in params]

    @property
    def num_params(self):
        return sum(W.size for W in self.weights)
