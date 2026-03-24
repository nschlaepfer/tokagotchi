"""DAPO (Dynamic Asymmetric Policy Optimisation) clipping utilities.

Implements the *Clip-Higher* variant that uses an asymmetric clipping range
for positive advantages, preventing entropy collapse while still training
effectively on negative signals.

Reference: the DAPO paper extends standard PPO/GRPO clipping by widening the
upper clip bound when the advantage is positive, encouraging the policy to
explore high-reward regions more aggressively.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import DAPOConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional torch import
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


class DAPOClipper:
    """Applies DAPO's asymmetric clipping to policy-gradient ratio terms.

    Standard GRPO clips the importance-sampling ratio to
    ``[1 - epsilon, 1 + epsilon]``.  DAPO's *Clip-Higher* strategy widens
    the upper bound to ``1 + epsilon_high`` when the advantage is positive,
    giving the policy more room to reinforce exploratory actions.

    Parameters
    ----------
    config:
        A :class:`DAPOConfig` supplying ``epsilon_low`` and ``epsilon_high``.
    """

    def __init__(self, config: DAPOConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Scalar clip (useful for debugging / unit tests)
    # ------------------------------------------------------------------

    def clip_ratio(
        self,
        ratio: float,
        advantage: float,
        config: DAPOConfig | None = None,
    ) -> float:
        """Clip a single importance-sampling ratio using DAPO rules.

        Parameters
        ----------
        ratio:
            The importance-sampling ratio ``pi_new / pi_old``.
        advantage:
            The GRPO advantage for the corresponding trajectory.
        config:
            Optional override config; defaults to ``self.config``.

        Returns
        -------
        float
            The clipped ratio.
        """
        cfg = config or self.config

        if advantage > 0:
            # Clip-Higher: wider upper bound encourages exploration
            low = 1.0 - cfg.epsilon_low
            high = 1.0 + cfg.epsilon_high
        else:
            # Standard symmetric clip for negative advantages
            low = 1.0 - cfg.epsilon_low
            high = 1.0 + cfg.epsilon_low

        return max(low, min(high, ratio))

    # ------------------------------------------------------------------
    # Full policy loss (tensor-level)
    # ------------------------------------------------------------------

    def compute_policy_loss(
        self,
        log_probs_new: Any,
        log_probs_old: Any,
        advantages: Any,
        config: DAPOConfig | None = None,
    ) -> Any:
        """Compute the DAPO-clipped policy gradient loss.

        Parameters
        ----------
        log_probs_new:
            Log-probabilities under the *current* policy.  Shape ``(B,)``.
        log_probs_old:
            Log-probabilities under the *reference* (old) policy.  Shape ``(B,)``.
        advantages:
            Per-example GRPO advantages.  Shape ``(B,)``.
        config:
            Optional override config; defaults to ``self.config``.

        Returns
        -------
        torch.Tensor
            Scalar loss (mean over the batch), ready for ``.backward()``.
        """
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for compute_policy_loss but is not installed."
            )

        cfg = config or self.config

        # Importance sampling ratio
        ratio = torch.exp(log_probs_new - log_probs_old)  # (B,)

        # Build per-element clip bounds
        eps_low = cfg.epsilon_low
        eps_high = cfg.epsilon_high

        # Positive-advantage mask
        pos_mask = (advantages > 0).float()

        # Upper clip bound: epsilon_high when advantage > 0, else epsilon_low
        upper_bound = 1.0 + pos_mask * eps_high + (1.0 - pos_mask) * eps_low
        lower_bound = torch.full_like(ratio, 1.0 - eps_low)

        clipped_ratio = torch.clamp(ratio, min=lower_bound, max=upper_bound)

        # Surrogate objectives
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages

        # PPO-style pessimistic bound
        loss = -torch.min(surr1, surr2).mean()
        return loss

    # ------------------------------------------------------------------
    # Entropy bonus
    # ------------------------------------------------------------------

    @staticmethod
    def entropy_bonus(
        logits: Any,
        config: Any,
    ) -> Any:
        """Compute an entropy bonus to prevent premature convergence.

        If the policy entropy drops below ``config.min_entropy``, an
        additional bonus term is added to the loss to encourage broader
        exploration.

        Parameters
        ----------
        logits:
            Raw logits from the policy head.  Shape ``(B, V)`` where *V*
            is the vocabulary size.
        config:
            Must expose ``min_entropy`` (floor) and ``entropy_coeff``
            (scaling coefficient) attributes.

        Returns
        -------
        torch.Tensor
            Scalar entropy bonus (non-negative).  Add this to the loss
            to encourage exploration.
        """
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for entropy_bonus but is not installed."
            )

        # Policy distribution entropy: H = -sum(p * log(p))
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()  # scalar

        # Apply entropy floor: bonus kicks in when entropy is too low
        min_entropy = getattr(config, "min_entropy", 0.01)
        entropy_coeff = getattr(config, "entropy_coeff", 0.01)

        # Bonus = coeff * max(0, min_entropy - entropy)
        # This pushes entropy upward when it falls below the floor.
        bonus = entropy_coeff * torch.clamp(min_entropy - entropy, min=0.0)

        # Also add a small general entropy bonus to always encourage diversity
        bonus = bonus + entropy_coeff * entropy

        return bonus
