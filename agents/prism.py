"""Prism agent entrypoint for oss-crs.

Configures ChatLiteLLM models from the LiteLLM proxy settings and runs the
ported PrismAgent against the libCRS-backed environment.
"""

import logging
import os
from pathlib import Path
from typing import Any

from crete.analyzer.jvm_stackoverflow import JVMStackOverflowStacktraceAnalyzer
from crete.analyzer.jvm_timeout import JVMTimeoutStacktraceAnalyzer
from crete.atoms.action import Action, choose_best_action, get_score
from crete.atoms.detection import AIxCCChallengeDeltaMode, BlobInfo, Detection
from crete.environment.environment_pool import EnvironmentPool
from crete.evaluator.default_evaluator import DefaultEvaluator

logger = logging.getLogger(__name__)

_LANGUAGE_MAP: dict[str, str] = {
    "c": "c",
    "c++": "cpp",
    "cpp": "cpp",
    "jvm": "jvm",
    "java": "jvm",
}

_primary_llm: Any = None
_backup_llm: Any = None


def setup(source_dir: Path, config: dict[str, str]) -> None:
    """Configure the Prism agent's primary and backup chat models."""
    del source_dir

    global _primary_llm, _backup_llm

    from langchain_community.chat_models import ChatLiteLLM

    llm_api_url = config.get("llm_api_url", "")
    llm_api_key = config.get("llm_api_key", "")

    os.environ["OPENAI_API_BASE"] = llm_api_url
    os.environ["OPENAI_API_KEY"] = llm_api_key

    primary_model = os.environ.get("PRISM_MODEL", "o4-mini")
    backup_model = os.environ.get(
        "PRISM_BACKUP_MODEL", "claude-sonnet-4-20250514"
    )

    _primary_llm = ChatLiteLLM(model=f"openai/{primary_model}")
    _backup_llm = ChatLiteLLM(model=f"openai/{backup_model}")

    logger.info(
        "Prism agent configured: primary=%s backup=%s api_base=%s",
        primary_model,
        backup_model,
        llm_api_url,
    )


def run(
    source_dir: Path,
    povs: list[tuple[Path, str]],
    harness: str,
    patches_dir: Path,
    work_dir: Path,
    *,
    language: str = "c",
    sanitizer: str = "address",
    builder: str,
    ref_diff: str | None = None,
) -> bool:
    """Run Prism and write the best diff to ``patches_dir``."""
    from crete.prism import PrismAgent
    from libCRS.cli.main import init_crs_utils

    crs = init_crs_utils()
    pool = EnvironmentPool(
        crs=crs,
        builder=builder,
        source_directory=source_dir,
    )

    crete_language = _LANGUAGE_MAP.get(language, language)
    blobs = [
        BlobInfo(
            harness_name=harness,
            sanitizer_name=sanitizer,
            blob=pov_path.read_bytes(),
        )
        for pov_path, _crash_log in povs
    ]

    mode = None
    if ref_diff is not None:
        mode = AIxCCChallengeDeltaMode(
            base_ref="HEAD",
            delta_ref="HEAD~1",
        )

    detection = Detection(
        mode=mode,
        vulnerability_identifier=harness,
        project_name=os.environ.get("OSS_CRS_TARGET", source_dir.name),
        language=crete_language,
        blobs=blobs,
        sarif_report=None,
    )

    evaluator = DefaultEvaluator(
        pool=pool,
        response_dir=work_dir / "eval_response",
    )
    output_directory = work_dir / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    context: dict[str, object] = {
        "pool": pool,
        "evaluator": evaluator,
        "logger": logger,
        "output_directory": output_directory,
        "jvm_timeout_analyzer": JVMTimeoutStacktraceAnalyzer(),
        "jvm_stackoverflow_analyzer": JVMStackOverflowStacktraceAnalyzer(),
    }

    agent = PrismAgent(
        llm=_primary_llm,
        backup_llm=_backup_llm,
        recursion_limit=256,
        max_n_evals=6,
    )

    try:
        actions = list(agent.act(context, detection))
    except Exception:
        logger.exception("Prism agent failed during execution")
        return False

    if not actions:
        logger.warning("Prism agent produced no actions")
        return False

    best_action = choose_best_action(actions)
    logger.info(
        "Best action: %s (score=%d)",
        type(best_action).__name__,
        get_score(best_action),
    )

    if not _has_diff(best_action):
        logger.info("Best action has no diff; skipping patch output")
        return False

    diff_bytes = best_action.diff  # type: ignore[union-attr]
    patches_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patches_dir / "patch_001.diff"
    patch_path.write_bytes(diff_bytes)
    logger.info("Patch written to %s (%d bytes)", patch_path, len(diff_bytes))
    return True


def _has_diff(action: Action) -> bool:
    return hasattr(action, "diff") and action.diff is not None  # type: ignore[union-attr]
