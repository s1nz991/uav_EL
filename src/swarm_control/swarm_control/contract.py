"""Observation/action contract and checkpoint loader for the hybrid swarm controller.

The contract in this file is `swarm-coop-enu-v1`. It is an explicit, documented
23-D observation / 4-D action schema. It is not a claim of compatibility with the
missing `maddpg_iter00460.pt` checkpoint; see README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = "swarm-coop-enu-v1"
OBS_DIM = 23
ACT_DIM = 4
N_AGENTS = 3

# Fixed scales from the contract. A checkpoint may override these in its metadata.
GOAL_SCALE = 15.0
VEL_SCALE = 0.8
PEER_VEL_SCALE = 1.6
OBSTACLE_SCALE = 15.0


@dataclass
class Contract:
    schema: str = SCHEMA
    obs_dim: int = OBS_DIM
    act_dim: int = ACT_DIM
    agents: int = N_AGENTS
    hidden: Sequence[int] = (128, 128)
    goal_scale: float = GOAL_SCALE
    vel_scale: float = VEL_SCALE
    peer_vel_scale: float = PEER_VEL_SCALE
    obstacle_scale: float = OBSTACLE_SCALE
    speed: float = 0.8
    accel: float = 1.2
    yaw_rate: float = 0.6
    radius: float = 0.30
    pair_safe: float = 1.0
    obs_margin: float = 0.25
    alt: float = 2.5
    clip_obs: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "agents": self.agents,
            "hidden": list(self.hidden),
            "goal_scale": self.goal_scale,
            "vel_scale": self.vel_scale,
            "peer_vel_scale": self.peer_vel_scale,
            "obstacle_scale": self.obstacle_scale,
            "speed": self.speed,
            "accel": self.accel,
            "yaw_rate": self.yaw_rate,
            "radius": self.radius,
            "pair_safe": self.pair_safe,
            "obs_margin": self.obs_margin,
            "alt": self.alt,
            "clip_obs": self.clip_obs,
        }


DEFAULT_CONTRACT = Contract()


class MissingPolicyDependency(RuntimeError):
    pass


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        raise MissingPolicyDependency(
            "PyTorch is required for mode:=learned. Install a CPU wheel, e.g. "
            "python3 -m pip install 'torch>=2.6,<3' --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def build_actor(contract: Contract):
    torch = _require_torch()
    from torch import nn

    layers: list[nn.Module] = []
    last = contract.obs_dim
    for width in contract.hidden:
        layers += [nn.Linear(last, int(width)), nn.ReLU()]
        last = int(width)
    layers += [nn.Linear(last, contract.act_dim), nn.Tanh()]
    return nn.Sequential(*layers)


def _state_dict_from_payload(payload: Any, actor_index: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint is not a dictionary or TorchScript module")
    if "actors" in payload:
        actors = payload["actors"]
        if not isinstance(actors, (list, tuple)) or len(actors) <= actor_index:
            raise ValueError(f"checkpoint has no actor at index {actor_index}")
        return actors[actor_index]
    for key in ("actor", "actor_state_dict", "state_dict", "model"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    # A bare state dict is accepted only with an explicit legacy override.
    if all(isinstance(v, (dict,)) or hasattr(v, "shape") for v in payload.values()):
        return payload
    raise ValueError("could not locate actor weights in checkpoint")


def _infer_hidden(state: dict[str, Any], contract: Contract) -> tuple[int, ...]:
    keys = sorted(
        (
            key
            for key in state
            if key.endswith(".weight") and key.split(".")[0].isdigit()
        ),
        key=lambda key: int(key.split(".")[0]),
    )
    if not keys:
        return tuple(int(width) for width in contract.hidden)
    widths: list[int] = []
    previous_out = None
    for key in keys:
        weight = state[key]
        shape = tuple(int(value) for value in weight.shape)
        if len(shape) != 2:
            raise ValueError(f"unexpected actor weight shape for {key}: {shape}")
        out_features, in_features = shape
        if previous_out is None:
            if in_features != contract.obs_dim:
                raise ValueError(
                    f"actor input width {in_features} != observation dimension {contract.obs_dim}"
                )
        elif in_features != previous_out:
            raise ValueError(f"actor layer {key} input width {in_features} != {previous_out}")
        widths.append(out_features)
        previous_out = out_features
    if widths[-1] != contract.act_dim:
        raise ValueError(
            f"actor output width {widths[-1]} != action dimension {contract.act_dim}"
        )
    return tuple(widths[:-1])


class ActorPolicy:
    """Strict checkpoint loader.

    Accepted formats:
      * TorchScript module (`.pt` produced with torch.jit.script/trace)
      * dict with `actors` list of state_dicts (swarm-coop trainer format)
      * dict with `actor` / `actor_state_dict` / `state_dict` / `model`
      * bare state_dict (requires allow_legacy=True)

    A checkpoint carrying `contract` metadata must match this contract exactly.
    """

    def __init__(
        self,
        path: str | Path,
        contract: Contract = DEFAULT_CONTRACT,
        actor_index: int = 0,
        allow_legacy: bool = False,
        device: str = "cpu",
    ):
        torch = _require_torch()
        self.torch = torch
        self.contract = contract
        self.actor_index = int(actor_index)
        self.device = torch.device(device)
        self.scripted = None
        self.module = None
        self.source = str(path)
        self.metadata: dict[str, Any] = {}

        checkpoint = Path(path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"policy checkpoint not found: {checkpoint}")

        # TorchScript is preferred: it carries no arbitrary pickle state.
        try:
            scripted = torch.jit.load(str(checkpoint), map_location=self.device)
            scripted.eval()
            with torch.no_grad():
                probe = scripted(torch.zeros(1, contract.obs_dim, device=self.device))
            if tuple(probe.shape) != (1, contract.act_dim):
                raise ValueError(
                    f"TorchScript output shape {tuple(probe.shape)} != (1, {contract.act_dim})"
                )
            self.scripted = scripted
            return
        except Exception:
            pass

        try:
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                "could not load checkpoint without executing pickle code; "
                "export it as TorchScript or a plain state_dict"
            ) from exc

        if isinstance(payload, dict) and "contract" in payload:
            meta = payload["contract"]
            if isinstance(meta, dict):
                self.metadata = dict(meta)
                declared = meta.get("schema")
                if declared is not None and declared != contract.schema:
                    raise ValueError(
                        f"checkpoint schema {declared!r} != controller schema {contract.schema!r}"
                    )
                for name in ("goal_scale", "vel_scale", "peer_vel_scale", "obstacle_scale"):
                    if name in meta:
                        setattr(self.contract, name, float(meta[name]))
                if "hidden" in meta:
                    self.contract.hidden = tuple(int(x) for x in meta["hidden"])

        if isinstance(payload, dict):
            for key, expected in (
                ("obs_dim", self.contract.obs_dim),
                ("action_dim", self.contract.act_dim),
                ("num_drones", self.contract.agents),
            ):
                if key in payload and int(payload[key]) != int(expected):
                    raise ValueError(
                        f"checkpoint {key}={payload[key]} != controller {key}={expected}"
                    )

        state = _state_dict_from_payload(payload, self.actor_index)
        if state and all(str(key).startswith("net.") for key in state):
            state = {str(key)[4:]: value for key, value in state.items()}
        inferred_hidden = _infer_hidden(state, self.contract)
        if tuple(int(width) for width in self.contract.hidden) != inferred_hidden:
            self.contract.hidden = inferred_hidden
        if not isinstance(payload, dict) or "contract" not in payload:
            if not allow_legacy:
                raise ValueError(
                    "checkpoint has no contract metadata. The original "
                    "maddpg_iter00460.pt requires its exact actor class, observation "
                    "order and action decoding. Set allow_legacy:=true only if you "
                    "have independently verified those semantics."
                )
        module = build_actor(self.contract)
        try:
            module.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ValueError(
                "actor state_dict is incompatible with the configured architecture "
                f"({self.contract.obs_dim}->{list(self.contract.hidden)}->{self.contract.act_dim}). "
                "Supply the original Actor class or export a TorchScript module."
            ) from exc
        module.to(self.device)
        module.eval()
        self.module = module

    @property
    def network(self):
        return self.scripted if self.scripted is not None else self.module

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        torch = self.torch
        obs = np.asarray(observation, dtype=np.float32).reshape(1, self.contract.obs_dim)
        with torch.no_grad():
            out = self.network(torch.as_tensor(obs, device=self.device))
        action = out.detach().cpu().numpy().reshape(self.contract.act_dim)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def normalize_observation(raw: Sequence[float], contract: Contract = DEFAULT_CONTRACT) -> np.ndarray:
    obs = np.asarray(raw, dtype=np.float32).reshape(contract.obs_dim)
    if contract.clip_obs:
        obs = np.clip(obs, -1.0, 1.0)
    return obs


def action_to_velocity(action: np.ndarray, contract: Contract = DEFAULT_CONTRACT):
    action = np.clip(np.asarray(action, dtype=np.float32).reshape(contract.act_dim), -1.0, 1.0)
    velocity = action[:3] * contract.speed
    yaw_rate = float(action[3] * contract.yaw_rate)
    return velocity.astype(np.float64), yaw_rate
