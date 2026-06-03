"""YAML 配置加载 + 命令行覆盖。"""
import os
import yaml
from typing import Any, Dict, List


def _parse_cli_overrides(extra: List[str]) -> Dict[str, Any]:
    """Parse --key.sub value pairs into nested dict."""
    overrides = {}
    i = 0
    while i < len(extra):
        arg = extra[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 >= len(extra):
                raise ValueError(f"Missing value for {arg}")
            value = extra[i + 1]
            i += 2

            # Try to cast value
            if value.lower() in ("true", "yes"):
                casted = True
            elif value.lower() in ("false", "no"):
                casted = False
            else:
                try:
                    if "." in value:
                        casted = float(value)
                    else:
                        casted = int(value)
                except ValueError:
                    casted = value

            # Build nested dict
            parts = key.split(".")
            d = overrides
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = casted
        else:
            i += 1
    return overrides


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


def load_config(config_path: str, cli_extra: List[str] = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if cli_extra:
        overrides = _parse_cli_overrides(cli_extra)
        cfg = _deep_update(cfg, overrides)
    cfg = _resolve_env_vars(cfg)
    return cfg
