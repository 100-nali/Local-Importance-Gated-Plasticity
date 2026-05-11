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


