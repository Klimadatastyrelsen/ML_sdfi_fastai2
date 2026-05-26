#!/usr/bin/env python3
"""
Run verification steps from README "## Verify that everything works".
Runs CUDA check then training for all configs in configs/example_configs/ that start with "test".
Output is written to verification.log (or path given as first argument).
Exit code 0 on success, non-zero on failure.
"""
import os
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "verification.log"
if len(sys.argv) > 1:
    LOG_PATH = Path(sys.argv[1])

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from verify_subprocess_streaming import log_message, log_section, stream_run  # noqa: E402

TIMEOUT_PER_CONFIG = int(os.environ.get("VERIFY_TRAIN_TIMEOUT", "600"))
CUDA_CHECK_CMD = (
    f'{sys.executable} -c "import torch; print(\'CUDA available:\', torch.cuda.is_available()); '
    f"print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\""
)


def require_cuda() -> int:
    log_section(LOG_PATH, "=== CUDA check ===")
    ret = stream_run(CUDA_CHECK_CMD, LOG_PATH, timeout=30)
    log_message(LOG_PATH, f"Exit code: {ret}")
    if ret != 0:
        return ret
    try:
        import torch

        if not torch.cuda.is_available():
            log_message(LOG_PATH, "CUDA_UNAVAILABLE: CUDA required but not available")
            return 1
    except Exception as exc:
        log_message(LOG_PATH, f"CUDA_UNAVAILABLE: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _repo_root() -> Path:
    root = Path(__file__).parent
    # On Windows, resolve() can turn a mapped drive (F:) into a UNC path that cmd rejects.
    if os.name == "nt" and str(root.resolve()).startswith("\\\\"):
        cwd = Path(os.getcwd())
        if cwd.name == root.name:
            return cwd
    return root


def main():
    repo_root = _repo_root()
    configs_dir = repo_root / "configs" / "example_configs"

    with open(LOG_PATH, "w", encoding="utf-8"):
        pass

    try:
        env_geotiff = {**os.environ, "GTIFF_SRS_SOURCE": "EPSG", "PYTHONUNBUFFERED": "1"}

        ret = require_cuda()
        if ret != 0:
            return ret

        test_configs = sorted(configs_dir.glob("test*.ini"))
        if not test_configs:
            log_section(LOG_PATH, "=== No test* configs found ===")
            return 0

        for config_path in test_configs:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            rel = config_path.relative_to(repo_root)
            log_section(LOG_PATH, f"=== train.py --config {rel} ===")
            ret = stream_run(
                f"{sys.executable} src/ML_sdfi_fastai2/train.py --config {rel}",
                LOG_PATH,
                cwd=repo_root,
                env=env_geotiff,
                timeout=TIMEOUT_PER_CONFIG,
            )
            log_message(LOG_PATH, f"Exit code: {ret}")
            if ret != 0:
                return ret

        return 0
    except Exception as exc:
        log_message(LOG_PATH, f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
