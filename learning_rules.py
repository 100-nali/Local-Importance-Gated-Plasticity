"""
Three local learning rules.

All rules are strictly local in the sense that the update for parameter w_i
depends only on quantities attached to that parameter (its own gradient, its
own current value, and its own scalar state). No global Fisher matrix, no
replay buffer, no cross-parameter coupling beyond what the forward/backward
solver already provides.

Rules:
  - SGDRule:                 w -= lr * g
  - ThresholdedSGDRule:      w -= lr * g  if |g| > tau else 0
                             (per-edge gradient gate; paper-style structure
                              but applied to per-batch g, not smoothed)
  - ImportanceGatedRule:     w  <- proximal step toward w* with weight lam*I
                             I  <- beta * I + (1-beta) * g^2     (per step)
                             w* <- w                             (at task boundary)

ImportanceGatedRule is the candidate. Each parameter holds two scalars: an
anchor w* (snapshot at the last task boundary) and an importance proxy I
(EMA of squared gradient — a per-parameter, online surrogate for diagonal
Fisher information). The anchor pull is a strictly local analogue of EWC's
penalty: it uses only this parameter's importance and its own deviation from
its own anchor. No global quadratic, no Hessian, no replay.
"""
import numpy as np


class SGDRule:
    """
    No protection, no gating. On the ANN substrate this is plain SGD on
    backprop gradients; on a coupled physical substrate it is pure
    unconstrained contrastive learning (the per-edge force from the
    free/clamped equilibrium difference is followed at full strength).
    Reported as "vanilla" in plots because that meaning is honest on both
    substrates.
    """
    name = "vanilla"

    def __init__(self, lr=0.05):
        self.lr = lr

    def init_state(self, network):
        return {}

    def step(self, network, grads, state):
        for i, g in enumerate(grads):
            network.weights[i] -= self.lr * g
        return state

    def on_task_boundary(self, network, state):
        return state


class ThresholdedSGDRule:
    """
    Per-edge gradient thresholding — block updates whose magnitude is below
    tau. Each edge sees only its own per-batch gradient; the threshold gates
    motion locally. This matches the paper's per-edge force-threshold
    structure (each edge decides independently whether to update), with the
    caveat that our per-batch gradient carries minibatch noise that the
    paper's quasi-static equilibrium force does not.

        w_e <- w_e - lr * g_e   if |g_e| > tau   else unchanged
    """
    name = "thresh"

    def __init__(self, lr=0.05, threshold=4e-2):
        self.lr = lr
        self.threshold = threshold

    def init_state(self, network):
        return {}

    def step(self, network, grads, state):
        for i, g in enumerate(grads):
            mask = (np.abs(g) > self.threshold).astype(g.dtype)
            network.weights[i] -= self.lr * g * mask
        return state

    def on_task_boundary(self, network, state):
        return state


class ImportanceGatedRule:
    """
    Local importance-gated update with a proximal (implicit) anchor pull
    using a SNAPSHOT of importance taken at the task boundary (online EWC).

    Per parameter w_i, with anchor w*_i and snapshot importance I*_i:

        I_i        <- beta * I_i + (1 - beta) * g_i^2          (per step, live EMA)
        w_i        <- (w_i - lr * g_i + lr * lam * I*_i * w*_i)
                       / (1 + lr * lam * I*_i)                 (proximal step)
        at task boundary:
            w*_i   <- w_i
            I*_i   <- I_i

    The use of snapshot importance I*_i rather than live importance I_i is
    the EWC-correct formulation: damping during task k+1 is determined by
    the importance accumulated during task k, not by the importance that
    keeps growing on edges actively learning task k+1. The previous version
    of this rule used live importance and was effectively penalizing
    currently-learning edges with extra damping, costing new-task fit
    without buying old-task retention.

    The proximal form is the stable counterpart of the explicit anchor step
    (w -= lr*(g + lam*I*(w - w*))). The divisor (1 + lr*lam*I*) is always
    >= 1, so high-importance parameters smoothly relax toward their anchor
    without overshoot. lam can be set aggressively without divergence.

    Strictly local: each edge uses only its own w, w*, I, I*, g.
    """
    name = "imp_gated"

    def __init__(self, lr=0.05, lam=80.0, beta=0.95):
        self.lr = lr
        self.lam = lam
        self.beta = beta

    def init_state(self, network):
        return {
            "anchor": None,
            "anchor_importance": None,
            "importance": [np.zeros_like(W) for W in network.weights],
        }

    def step(self, network, grads, state):
        anchor = state["anchor"]
        anchor_imp = state["anchor_importance"]
        importance = state["importance"]
        for i, g in enumerate(grads):
            importance[i] = self.beta * importance[i] + (1.0 - self.beta) * (g * g)
            w = network.weights[i]
            if anchor is None or anchor_imp is None:
                network.weights[i] = w - self.lr * g
            else:
                k = self.lr * self.lam * anchor_imp[i]
                network.weights[i] = (w - self.lr * g + k * anchor[i]) / (1.0 + k)
        return state

    def on_task_boundary(self, network, state):
        state["anchor"] = [W.copy() for W in network.weights]
        state["anchor_importance"] = [I.copy() for I in state["importance"]]
        return state


class CumulativeImportanceGatedRule(ImportanceGatedRule):
    """
    Online-EWC variant: anchor importance accumulates across task boundaries
    instead of being overwritten. Each edge's protection is the sum of its
    importance over every past task, so an edge important to any past task
    keeps being damped — not just edges important to the most recent task.

    Identical step rule to ImportanceGatedRule; differs only in on_task_boundary:

        I*_i  <-  I*_i + I_i        (was: I*_i <- I_i)

    Anchor still snaps to the post-task weights each boundary. Cumulative I*
    grows roughly linearly in number of tasks, so the effective protection
    strength at fixed lambda is ~N x stronger after N tasks than the snapshot
    variant — expect the operating-point lambda to shift down accordingly.
    Still strictly local: every edge holds only its own scalar I*_i.
    """
    name = "cum_imp_gated"

    def on_task_boundary(self, network, state):
        state["anchor"] = [W.copy() for W in network.weights]
        if state["anchor_importance"] is None:
            state["anchor_importance"] = [I.copy() for I in state["importance"]]
        else:
            for i in range(len(state["anchor_importance"])):
                state["anchor_importance"][i] = (
                    state["anchor_importance"][i] + state["importance"][i]
                )
        return state


class MultiAnchorImportanceGatedRule(ImportanceGatedRule):
    """
    Per-task anchors collapsed to two running sums. The anchor loss is
        L_anchor(w) = sum_k (lam/2) * I*_k * (w - w*_k)^2
                    = (lam/2) * [S * w^2 - 2 R * w + const]
    with
        S_i = sum_k I*_k_i             (cumulative importance)
        R_i = sum_k I*_k_i * w*_k_i    (importance-weighted sum of anchors)

    Because the loss is quadratic in w, its gradient depends only on S and
    R. Per-edge memory stays at two scalars regardless of task count — the
    individual (w*_k, I*_k) pairs do not need to be stored.

    Per-step proximal (implicit) update:

        w_new = (w - lr*g + lr*lam*R) / (1 + lr*lam*S)

    Equivalently: each edge is pulled toward the importance-weighted
    centroid of all past task solutions, R/S, rather than the most recent
    endpoint. For orthogonal tasks the centroid is closer to every past
    solution than the last endpoint is — fixes the single-anchor bias that
    keeps cum_imp_gated borderline.

    At task boundary:
        S_i  <-  S_i + I_i
        R_i  <-  R_i + I_i * w_i
    where w_i is the post-task weight for the task just finished.

    Strictly local: every edge uses only its own w, g, S, R, I.
    """
    name = "multi_anchor"

    def init_state(self, network):
        return {
            "S": [np.zeros_like(W) for W in network.weights],
            "R": [np.zeros_like(W) for W in network.weights],
            "importance": [np.zeros_like(W) for W in network.weights],
        }

    def step(self, network, grads, state):
        S = state["S"]
        R = state["R"]
        importance = state["importance"]
        for i, g in enumerate(grads):
            importance[i] = self.beta * importance[i] + (1.0 - self.beta) * (g * g)
            w = network.weights[i]
            k = self.lr * self.lam * S[i]
            network.weights[i] = (w - self.lr * g + self.lr * self.lam * R[i]) / (1.0 + k)
        return state

    def on_task_boundary(self, network, state):
        for i in range(len(state["S"])):
            I_curr = state["importance"][i]
            w_curr = network.weights[i]
            state["S"][i] = state["S"][i] + I_curr
            state["R"][i] = state["R"][i] + I_curr * w_curr
        return state


class HeatCumImportanceGatedRule(ImportanceGatedRule):
    """
    Online-EWC variant whose importance estimator is the per-task cumulative
    heat Q_e = -g_e * delta_w_e (Synaptic Intelligence, Zenke et al. 2017).

    Replaces:
        I_i  <-  beta * I_i + (1 - beta) * g_i^2     (Fisher-diagonal proxy)
    with:
        Q_i  <-  Q_i + (-g_i * delta_w_i)             (per-step path integral)
        at task boundary:
            I_task_k_i  =  max(Q_i, 0) + eps          (clipped + numerical floor)
            I*_i        <-  I*_i + I_task_k_i         (online EWC accumulation)
            Q_i         <-  0                         (reset for next task)

    The same proximal step as ImportanceGatedRule is used (single anchor at
    most-recent task endpoint, cumulative I*). Only the importance signal
    changes: Q measures actual loss reduction attributable to each edge —
    "work done by the contrastive force on this edge during this task" —
    rather than gradient-magnitude noise. SI typically beats Fisher-diagonal
    on orthogonal continual-learning sequences because it captures which
    edges contributed to descent rather than which edges saw large gradients.

    eps prevents zero importance on edges that barely moved (otherwise the
    proximal denominator is exactly 1 and those edges have no anchor pull).

    Strictly local: every edge uses only its own w, g, Q, I*, w*.
    """
    name = "heat_cum_imp_gated"

    def __init__(self, lr=0.05, lam=80.0, eps=1e-3):
        super().__init__(lr=lr, lam=lam, beta=0.95)  # beta unused
        self.eps = eps

    def init_state(self, network):
        return {
            "anchor": None,
            "anchor_importance": None,
            "task_Q": [np.zeros_like(W) for W in network.weights],
        }

    def step(self, network, grads, state):
        anchor = state["anchor"]
        anchor_imp = state["anchor_importance"]
        for i, g in enumerate(grads):
            w_before = network.weights[i].copy()
            if anchor is None or anchor_imp is None:
                network.weights[i] = w_before - self.lr * g
            else:
                k = self.lr * self.lam * anchor_imp[i]
                network.weights[i] = (w_before - self.lr * g + k * anchor[i]) / (1.0 + k)
            delta_w = network.weights[i] - w_before
            state["task_Q"][i] = state["task_Q"][i] + (-g * delta_w)
        return state

    def on_task_boundary(self, network, state):
        state["anchor"] = [W.copy() for W in network.weights]
        I_curr = [np.maximum(Q, 0.0) + self.eps for Q in state["task_Q"]]
        if state["anchor_importance"] is None:
            state["anchor_importance"] = [I.copy() for I in I_curr]
        else:
            for i in range(len(state["anchor_importance"])):
                state["anchor_importance"][i] = (
                    state["anchor_importance"][i] + I_curr[i]
                )
        for i in range(len(state["task_Q"])):
            state["task_Q"][i] = np.zeros_like(state["task_Q"][i])
        return state


