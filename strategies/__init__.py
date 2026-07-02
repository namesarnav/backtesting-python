"""Strategy registry.

Strategies are declared in `configs/strategies.yaml` -- class path plus
parameters -- and built by name here. Nothing in `engine/` imports a concrete
strategy, so adding a fourth idea means writing one file and adding one YAML
block, with no change to engine code.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from strategies.base import Strategy

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STRATEGY_CONFIG = REPO_ROOT / "configs" / "strategies.yaml"

__all__ = ["Strategy", "load_strategy", "load_all_strategies", "available_strategies"]


def _read_config(path: Path | str = DEFAULT_STRATEGY_CONFIG) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def available_strategies(config_path: Path | str = DEFAULT_STRATEGY_CONFIG) -> list[str]:
    """Names registered in the strategy config."""
    return sorted(_read_config(config_path))


def load_strategy(
    name: str,
    config_path: Path | str = DEFAULT_STRATEGY_CONFIG,
    **overrides,
) -> Strategy:
    """Build one strategy by its config name.

    `overrides` beat the YAML params, which is handy in tests and for
    parameter sweeps without editing the config file.
    """
    config = _read_config(config_path)
    if name not in config:
        raise KeyError(f"unknown strategy {name!r}; registered: {sorted(config)}")

    entry = config[name]
    class_path = entry["class"]
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"strategy {name!r} has a malformed class path: {class_path!r}")

    strategy_class = getattr(importlib.import_module(module_name), class_name)
    params = {**(entry.get("params") or {}), **overrides}
    return strategy_class(**params)


def load_all_strategies(config_path: Path | str = DEFAULT_STRATEGY_CONFIG) -> dict[str, Strategy]:
    """Build every registered strategy, keyed by name."""
    return {name: load_strategy(name, config_path) for name in available_strategies(config_path)}
