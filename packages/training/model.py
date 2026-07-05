"""
model.py — CatanNet: a small MLP for Catan policy + value estimation.

Architecture
────────────
Input  obs [batch, 1603]  — normalised game-state observation (4-seat)
           (from encoding.ts encodeObservation, perspective of the acting player)

Shared body:
  Linear(1603 → 512) → LayerNorm → ReLU
  Linear(512  → 256) → LayerNorm → ReLU
  Linear(256  → 256) → ReLU

Policy head:  Linear(256 → 376)  — raw logits; apply legal mask + softmax outside
Value  net:   SEPARATE trunk Linear(1603 → 256) → LN → ReLU → Linear(256 → 256)
              → ReLU → Linear(256 → 4) — per-seat win logits, seat-relative
              (slot 0 = acting seat); train with cross-entropy to the winner's
              relative seat offset. Kept separate from the policy body so critic
              gradients cannot reshape the features under a BC-pretrained head.

Notes
─────
• The legal mask is NOT applied inside the network; it is applied by the TS inference
  code (or the PPO rollout loop) so the ONNX export stays simple.
• For PPO, use separate param groups (see make_optimizer) so the value head can track
  bootstrapped returns faster than the policy head updates.
• At inference time, to get a win probability: value.sigmoid().
"""

from __future__ import annotations
import torch
import torch.nn as nn

OBS_SIZE = 1659
ACT_SIZE = 396
N_SEATS  = 4   # value head width: per-seat win logits, seat-relative


class CatanNet(nn.Module):
    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        act_size: int = ACT_SIZE,
        hidden:   int = 256,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            # NO LayerNorm here: bc_v1 checkpoints were trained without it, and
            # inserting one breaks warm-start compatibility (a fresh LayerNorm
            # mid-body scrambles the pretrained policy). Value-head stability
            # is handled by PPO's critic warm-up instead.
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, act_size)

        # Separate critic trunk. The value loss must NOT backprop through the
        # policy's body: with a shared trunk, critic gradients steadily
        # reshape the features under the (BC-pretrained) policy head and
        # degrade it — measured directly: shared-trunk PPO decayed 50% → 38%
        # vs greedy while a frozen body held 50%. Raw scalar output, no
        # Sigmoid (it kills gradients exactly at terminal states).
        self.value_net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_SEATS),
        )

    def forward(self, obs: torch.Tensor):
        """
        obs: [batch, obs_size]

        Returns
        -------
        policy_logits : [batch, act_size]  — before softmax / masking
        value         : [batch, N_SEATS]   — per-seat win logits, seat-relative
                                             (slot 0 = acting seat); softmax for P
        """
        h = self.body(obs)
        return self.policy_head(h), self.value_net(obs)

    def win_prob(self, obs: torch.Tensor) -> torch.Tensor:
        """P(the acting player wins) ∈ (0, 1). Display/logging only."""
        _, v = self(obs)
        return v.softmax(dim=-1)[:, 0]


def make_optimizer(
    model: CatanNet,
    lr_body:   float = 3e-4,
    lr_policy: float = 3e-4,
    lr_value:  float = 1e-3,   # value head benefits from a faster lr
) -> torch.optim.Adam:
    """
    Returns an Adam optimizer with separate learning rates for the body,
    policy head, and value head.

    Typical PPO usage
    -----------------
    optimizer = make_optimizer(model)

    # inside the PPO update loop:
    policy_loss = ppo_clip_loss(...)
    value_loss  = F.mse_loss(value.squeeze(-1), returns)   # returns ∈ [0, 1]
    entropy_bonus = -(probs * probs.log()).sum(-1).mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    """
    return torch.optim.Adam([
        {'params': model.body.parameters(),        'lr': lr_body},
        {'params': model.policy_head.parameters(), 'lr': lr_policy},
        {'params': model.value_net.parameters(),   'lr': lr_value},
    ])