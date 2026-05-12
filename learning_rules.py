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

    def step(self, network, grads, state, cache=None):
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

    def step(self, network, grads, state, cache=None):
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

    def step(self, network, grads, state, cache=None):
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

    def step(self, network, grads, state, cache=None):
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
    Online-EWC variant whose importance estimator follows Synaptic
    Intelligence (Zenke et al. 2017) using only edge-local quantities.

    Replaces:
        I_i  <-  beta * I_i + (1 - beta) * g_i^2     (Fisher-diagonal proxy)
    with:
        Q_i  <-  Q_i + (-g_i * delta_w_i)             (per-step path integral)
        at task boundary:
            D_i         =  w_i - w_i,start            (net task displacement)
            I_task_k_i  =  max(Q_i, 0) / (D_i^2 + xi) (SI normalization)
            I*_i        <-  I*_i + I_task_k_i         (online EWC accumulation)
            Q_i         <-  0                         (reset for next task)
            w_i,start   <-  w_i                       (start next task)

    The same proximal step as ImportanceGatedRule is used (single anchor at
    most-recent task endpoint, cumulative I*). Only the importance signal
    changes: Q measures actual loss reduction attributable to each edge,
    normalized by how far the edge ended up moving over the task. This gives
    high importance to edges that did useful work despite small net motion,
    matching SI's "contribution per displacement" structure.

    xi prevents division by zero when an edge has nearly zero net task
    displacement. Smaller xi makes the SI normalization more aggressive.

    Strictly local: every edge uses only its own w, task-start w, g, Q, I*, w*.
    """
    name = "heat_cum_imp_gated"

    def __init__(self, lr=0.05, lam=80.0, xi=1e-3):
        super().__init__(lr=lr, lam=lam, beta=0.95)  # beta unused
        self.xi = xi

    def init_state(self, network):
        return {
            "anchor": None,
            "anchor_importance": None,
            "task_Q": [np.zeros_like(W) for W in network.weights],
            "task_start": [W.copy() for W in network.weights],
        }

    def step(self, network, grads, state, cache=None):
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
        I_curr = []
        for i, Q in enumerate(state["task_Q"]):
            delta_task = network.weights[i] - state["task_start"][i]
            I_curr.append(np.maximum(Q, 0.0) / (delta_task * delta_task + self.xi))
        if state["anchor_importance"] is None:
            state["anchor_importance"] = [I.copy() for I in I_curr]
        else:
            for i in range(len(state["anchor_importance"])):
                state["anchor_importance"][i] = (
                    state["anchor_importance"][i] + I_curr[i]
                )
        for i in range(len(state["task_Q"])):
            state["task_Q"][i] = np.zeros_like(state["task_Q"][i])
            state["task_start"][i] = network.weights[i].copy()
        return state


class SlowConsolidatedImportanceRule(ImportanceGatedRule):
    """
    Biologically motivated slow-consolidation rule.

    Each edge keeps three local quantities:
        w_e  = expressed, fast conductance used by the substrate
        z_e  = slow consolidated conductance
        S_e  = bounded metaplastic stability / importance

    The fast conductance remains plastic, but high-stability edges learn with
    a smaller effective step and feel a weak pull toward their consolidated
    value. At task boundaries, local SI-style work determines how much this
    task should stabilize the edge, and z_e slowly moves toward the current
    expressed conductance:

        Q_e       <- sum_t -g_e(t) * delta_w_e(t)
        I_e       <- max(Q_e, 0) / ((w_e - w_e,start)^2 + xi)
        tag_e     <- I_e / (I_e + importance_scale)      in [0, 1]
        S_e       <- min(stability_cap, decay*S_e + tag_e)
        z_e       <- z_e + consolidation_rate*tag_e*(w_e - z_e)

    Unlike the EWC-like anchor rules, this does not treat the task endpoint as
    a hard quadratic well. It is closer to synaptic tagging / consolidation:
    useful local work creates a bounded tag, and durable memory is written
    gradually into a slow variable while the expressed weight remains able to
    move for the next task.
    """
    name = "slow_consolidated"

    def __init__(self, lr=0.05, lam=1.0, pull=0.02,
                 consolidation_rate=0.25, stability_decay=0.95,
                 stability_cap=3.0, importance_scale=1.0, xi=1e-3):
        super().__init__(lr=lr, lam=lam, beta=0.95)  # beta unused
        self.pull = pull
        self.consolidation_rate = consolidation_rate
        self.stability_decay = stability_decay
        self.stability_cap = stability_cap
        self.importance_scale = importance_scale
        self.xi = xi

    def init_state(self, network):
        return {
            "consolidated": [W.copy() for W in network.weights],
            "stability": [np.zeros_like(W) for W in network.weights],
            "task_Q": [np.zeros_like(W) for W in network.weights],
            "task_start": [W.copy() for W in network.weights],
        }

    def step(self, network, grads, state, cache=None):
        z = state["consolidated"]
        stability = state["stability"]
        for i, g in enumerate(grads):
            w_before = network.weights[i].copy()
            S = stability[i]

            # Stability gates plasticity, but does not fully freeze the edge.
            gated_lr = self.lr / (1.0 + self.lam * S)
            w_plastic = w_before - gated_lr * g

            # Weak fast-timescale relaxation toward the slow consolidated state.
            k = self.lr * self.pull * S
            network.weights[i] = (w_plastic + k * z[i]) / (1.0 + k)

            delta_w = network.weights[i] - w_before
            state["task_Q"][i] = state["task_Q"][i] + (-g * delta_w)
        return state

    def on_task_boundary(self, network, state):
        for i, Q in enumerate(state["task_Q"]):
            delta_task = network.weights[i] - state["task_start"][i]
            I_task = np.maximum(Q, 0.0) / (delta_task * delta_task + self.xi)
            tag = I_task / (I_task + self.importance_scale)

            state["stability"][i] = np.minimum(
                self.stability_cap,
                self.stability_decay * state["stability"][i] + tag,
            )
            state["consolidated"][i] = (
                state["consolidated"][i]
                + self.consolidation_rate * tag
                * (network.weights[i] - state["consolidated"][i])
            )
            state["task_Q"][i] = np.zeros_like(state["task_Q"][i])
            state["task_start"][i] = network.weights[i].copy()
        return state


class ActivityCumulativeImportanceRule(CumulativeImportanceGatedRule):
    """
    Cumulative online-EWC variant whose importance signal is per-edge
    free-phase *activity* rather than squared contrastive gradient.

    Activity of an edge in the free phase is the squared endpoint voltage
    drop in the free-phase equilibrium, averaged across the batch:

        I_e  <-  beta * I_e + (1 - beta) * mean_b (v_i^F - v_j^F)^2

    For parameters indexed by a context axis (per-edge gain components
    u_e[j]), per-axis importance is the same edge activity scaled by
    c[j]^2 — the same chain-rule scaling squared-gradient importance
    picks up implicitly, applied explicitly here.

    Unlike g^2, activity does not decay to zero when a task converges:
    edges that carry significant voltage drops in the task's equilibrium
    keep registering as important. The intent is that the anchor pull
    latches onto "edges the substrate uses to express this task" rather
    than "edges that were transiently moving during optimization."

    Requires the substrate to provide compute_activity_importance(cache);
    raises if a forward cache is not supplied or the substrate does not
    support it.

    Strictly local: every edge uses only its own free-phase endpoint
    voltages and the broadcast context vector.
    """
    name = "act_cum_imp_gated"

    def step(self, network, grads, state, cache=None):
        if cache is None or not hasattr(network, "compute_activity_importance"):
            raise RuntimeError(
                "ActivityCumulativeImportanceRule requires a forward cache "
                "and a substrate exposing compute_activity_importance()."
            )
        activity = network.compute_activity_importance(cache)
        anchor = state["anchor"]
        anchor_imp = state["anchor_importance"]
        importance = state["importance"]
        for i, g in enumerate(grads):
            importance[i] = (
                self.beta * importance[i]
                + (1.0 - self.beta) * activity[i]
            )
            w = network.weights[i]
            if anchor is None or anchor_imp is None:
                network.weights[i] = w - self.lr * g
            else:
                k = self.lr * self.lam * anchor_imp[i]
                network.weights[i] = (w - self.lr * g + k * anchor[i]) / (1.0 + k)
        return state
