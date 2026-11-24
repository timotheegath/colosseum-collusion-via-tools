from __future__ import annotations

import experiments.common._patch_terrarium  # noqa: F401  # monkey-patch terrarium providers

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml
from tqdm import tqdm

from terrarium.utils import resolve_phase_agents


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, data)
    tmp.replace(path)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


def configure_experiment_logging(
    logger: logging.Logger, root: Path, *, verbose: bool = True
) -> None:
    """Configure a dedicated experiment logger (file + tqdm-safe console)."""
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Reset handlers so repeated calls (or notebooks) re-point to the new output root.
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)

    log_path = root / "experiment.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)

    if verbose:
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(console_handler)


def write_progress(root: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat()
    atomic_write_json(root / "progress.json", payload)


def normalize_seeds(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        # Allow comma-separated values as a convenience.
        parts = [p.strip() for p in raw.split(",")]
        seeds: List[int] = []
        for part in parts:
            if not part:
                continue
            try:
                seeds.append(int(part))
            except Exception:
                continue
        return seeds
    if isinstance(raw, list):
        seeds = []
        for item in raw:
            if item is None:
                continue
            try:
                seeds.append(int(item))
            except Exception:
                continue
        return seeds
    return []


def tools_environment_name(env: Any) -> str:
    """The toolset-environment name used for ``env``'s tool discovery."""
    if env is None:
        return ""
    return str(
        getattr(env, "tools_environment_name", None) or type(env).__name__ or ""
    )


# ---------------------------------------------------------------------------
# Native-config injection + phase walking (delegates to the Terrarium submodule)
# ---------------------------------------------------------------------------


def inject_native_collusion_config(
    cfg: Dict[str, Any],
    *,
    adversaries: List[str],
    strategy: str,
    instructions_file: str,
    secret_channel: bool,
    enforce_phase: Optional[str] = None,
    disable_post_message: bool = False,
    prompt_wrapper: Optional[str] = None,
    phases: List[Dict[str, Any]] = None,
    blocked_tools: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the native ``environment``/``simulation`` blocks for one run.

    All override surface lives in the per-run YAML config consumed by the
    submodule:

    - ``environment.collusion`` (members/strategy/instructions_file) drives
      ``build_agent_context`` (role / coalition_members / secret_blackboard_id)
      and the adversary-only instructions rendering;
    - ``environment.secret_channels`` makes the environment create the
      coalition-only board itself (``enforce_in_phases`` rewrites post_message
      onto it during the given phase);
    - ``environment.blocked_tools`` gates tools per phase (and per agent via
      ``roles``/``agents`` mapping entries);
    - ``environment.prompt_wrapper`` installs the experiment prompt decorator;
    - ``simulation.phases`` drives the native phase loop.
    """
    env_cfg = cfg.setdefault("environment", {})
    env_cfg["collusion"] = {
        "enabled": True,
        "members": list(adversaries),
        "strategy": str(strategy),
        "instructions_file": str(instructions_file),
    }
    if secret_channel:
        entry: Dict[str, Any] = {"members": list(adversaries)}
        if enforce_phase:
            entry["enforce_in_phases"] = [enforce_phase]
        env_cfg["secret_channels"] = [entry]
    else:
        env_cfg.pop("secret_channels", None)

    blocked: Dict[str, Any] = dict(blocked_tools or {})
    if disable_post_message:
        adversary_pm_block = {"tool": "post_message", "roles": ["adversary"]}
        for phase_name in ("planning", "execution", "survey"):
            blocked.setdefault(phase_name, []).append(dict(adversary_pm_block))
    if blocked:
        env_cfg["blocked_tools"] = blocked
    else:
        env_cfg.pop("blocked_tools", None)

    if prompt_wrapper:
        env_cfg["prompt_wrapper"] = prompt_wrapper
    else:
        env_cfg.pop("prompt_wrapper", None)

    if phases is not None:
        cfg.setdefault("simulation", {})["phases"] = [dict(p) for p in phases]


async def run_phase_entries(
    *,
    protocol: Any,
    env: Any,
    phase_entries: List[Dict[str, Any]],
    iteration: int,
    build_turn: Callable[..., Dict[str, Any]],
    on_turn: Optional[Callable[[Dict[str, Any]], None]] = None,
    context_hook: Optional[
        Callable[[str, Dict[str, Any], str, int, int], Dict[str, Any]]
    ] = None,
) -> List[Dict[str, Any]]:
    """Walk resolved ``simulation.phases`` entries once at ``iteration``.

    Each phase runs ``rounds`` rounds; per round, every agent named by the
    phase's ``agents`` entry (all agents when absent) gets one native
    ``protocol.agent_turn(..., phase=..., round_num=...)``. The per-turn record
    is produced by ``build_turn(agent=..., phase=..., round_num=..., iteration=...,
    response=...[, error=...])`` and handed to ``on_turn`` as it completes.
    Turn failures are recorded as error turns (via ``build_turn(..., error=exc)``)
    and never abort the phase walk.
    """
    turns: List[Dict[str, Any]] = []
    agent_names = [str(a.name) for a in env.agents]
    for phase in phase_entries:
        phase_name = str(phase["name"])
        rounds = int(phase.get("rounds", 1))
        subset = resolve_phase_agents(phase, agent_names)
        for round_num in range(1, rounds + 1):
            for agent in env.agents:
                if subset is not None and str(agent.name) not in subset:
                    continue
                agent_context = env.build_agent_context(
                    agent.name,
                    phase=phase_name,
                    iteration=iteration,
                    planning_round=round_num,
                )
                if context_hook is not None:
                    agent_context = context_hook(
                        agent.name, agent_context, phase_name, round_num, iteration
                    )
                try:
                    response = await protocol.agent_turn(
                        agent,
                        agent.name,
                        agent_context,
                        env,
                        iteration,
                        phase=phase_name,
                        round_num=round_num,
                    )
                except Exception as exc:  # noqa: BLE001 - record and continue
                    turn = build_turn(
                        agent=agent,
                        phase=phase_name,
                        round_num=round_num,
                        iteration=iteration,
                        response=None,
                        error=exc,
                    )
                    turns.append(turn)
                    if on_turn is not None:
                        on_turn(turn)
                    continue
                turn = build_turn(
                    agent=agent,
                    phase=phase_name,
                    round_num=round_num,
                    iteration=iteration,
                    response=response,
                )
                turns.append(turn)
                if on_turn is not None:
                    on_turn(turn)
    return turns
