from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


class ConfigError(ValueError):
    """Raised when the profiler configuration is invalid."""


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return data


def load_config(default_path: Path, user_path: Path | None = None) -> Dict[str, Any]:
    config = load_yaml(default_path)
    if user_path is not None:
        _deep_merge(config, load_yaml(user_path))
    validate_config(config)
    return config


def require_path(config: Mapping[str, Any], dotted_path: str) -> Any:
    node: Any = config
    for key in dotted_path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            raise ConfigError(f"Missing required configuration key: {dotted_path}")
        node = node[key]
    return node


def validate_config(config: Mapping[str, Any]) -> None:
    required = [
        "models.siglip2.enabled",
        "models.segformer.enabled",
        "models.open_vocab.enabled",
        "models.sam2.enabled",
        "runtime.device",
        "semantic_axes.environment.classes",
        "semantic_axes.time.classes",
        "semantic_axes.content.classes",
        "segmentation_groups",
        "classification.dynamic_range_proxy.calibration_mode",
        "classification.brightness.calibration_mode",
    ]
    for key in required:
        require_path(config, key)

    for axis_name in ("environment", "time", "content", "illumination_semantic"):
        axis = require_path(config, f"semantic_axes.{axis_name}")
        if not isinstance(axis.get("classes"), Mapping) or not axis["classes"]:
            raise ConfigError(f"semantic_axes.{axis_name}.classes must be a non-empty mapping")
        for label, prompts in axis["classes"].items():
            if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
                raise ConfigError(
                    f"semantic_axes.{axis_name}.classes.{label} must be a non-empty string list"
                )

    for name in ("dynamic_range_proxy", "brightness"):
        mode = require_path(config, f"classification.{name}.calibration_mode")
        if mode not in {"fixed", "dataset_quantile"}:
            raise ConfigError(
                f"classification.{name}.calibration_mode must be fixed or dataset_quantile"
            )

    open_vocab_enabled = bool(require_path(config, "models.open_vocab.enabled"))
    sam2_enabled = bool(require_path(config, "models.sam2.enabled"))
    if sam2_enabled and not open_vocab_enabled:
        raise ConfigError("SAM2 refinement requires models.open_vocab.enabled=true")


def set_dotted(config: MutableMapping[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    node = config
    for key in parts[:-1]:
        child = node.get(key)
        if not isinstance(child, MutableMapping):
            child = {}
            node[key] = child
        node = child
    node[parts[-1]] = value


def apply_cli_overrides(config: Dict[str, Any], overrides: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    result = deepcopy(config)
    for dotted_path, value in overrides:
        if value is not None:
            set_dotted(result, dotted_path, value)
    validate_config(result)
    return result
