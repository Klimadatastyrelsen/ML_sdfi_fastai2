#!/usr/bin/env python
# coding: utf-8
"""
Upload trained models to Hugging Face Hub.

Reads train_*.ini configs from a directory, resolves each model path
(experiment_root / job_name / models / job_name.pth), and uploads existing
.pth files to a Hugging Face repository.

Requires: pip install huggingface_hub
"""

import argparse
import configparser
import sys
from pathlib import Path

from huggingface_hub import HfApi


def get_model_path_from_config(config_path: Path) -> tuple[Path | None, str | None]:
    """
    Parse a train config and return (model_path, job_name).
    Only reads [DATASET] experiment_root and [NAME] job_name.
    Returns (None, None) if sections/keys are missing.
    """
    parser = configparser.ConfigParser()
    if not config_path.is_file():
        return None, None
    parser.read(config_path)
    if "DATASET" not in parser or "NAME" not in parser:
        return None, None
    if "experiment_root" not in parser["DATASET"] or "job_name" not in parser["NAME"]:
        return None, None
    experiment_root = Path(parser["DATASET"]["experiment_root"].strip())
    job_name = parser["NAME"]["job_name"].strip().strip('"')
    model_path = (experiment_root / job_name / "models" / f"{job_name}.pth").resolve()
    return model_path, job_name


def main():
    usage = (
        "Upload models to Hugging Face from train configs.\n"
        "Example:\n"
        "  python upload_models_to_huggingface.py --config_dir /mnt/T/mnt/config_files/bygnings_udpegning/2026_production/\n"
        "  python upload_models_to_huggingface.py --config_dir /path/to/configs --dry_run\n"
    )
    parser = argparse.ArgumentParser(
        description="Upload trained .pth models to a Hugging Face repo using train_*.ini configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=usage,
    )
    parser.add_argument(
        "--config_dir",
        type=Path,
        default=Path("/mnt/T/mnt/config_files/bygnings_udpegning/2026_production"),
        help="Directory containing train_*.ini config files",
    )
    parser.add_argument(
        "--token_file",
        type=Path,
        default=None,
        help="Path to file containing Hugging Face write token. Default: ../laz-superpoint_transformer/hftoken_write.txt relative to this script's parent.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="rasmuspjohansson/KDS_buildings",
        help="Hugging Face repo id (e.g. username/repo_name)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print which configs and model paths would be uploaded",
    )
    args = parser.parse_args()

    config_dir = args.config_dir.resolve()
    if not config_dir.is_dir():
        print(f"Error: config_dir is not a directory: {config_dir}", file=sys.stderr)
        sys.exit(1)

    token_path = args.token_file
    if token_path is None:
        # Default: ../laz-superpoint_transformer/hftoken_write.txt relative to repo root
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent  # project root (ML_sdfi_fastai2)
        token_path = (repo_root.parent / "laz-superpoint_transformer" / "hftoken_write.txt").resolve()
    else:
        token_path = token_path.resolve()

    config_files = sorted(config_dir.glob("train_*.ini"))
    if not config_files:
        print(f"No train_*.ini files found in {config_dir}")
        return

    if not args.dry_run and not token_path.is_file():
        print(f"Error: token file not found: {token_path}", file=sys.stderr)
        sys.exit(1)

    token = None
    if not args.dry_run:
        token = token_path.read_text().strip()
        if not token:
            print("Error: token file is empty", file=sys.stderr)
            sys.exit(1)

    api = HfApi(token=token) if token else None
    failed = 0
    uploaded = 0

    for cfg_path in config_files:
        model_path, job_name = get_model_path_from_config(cfg_path)
        if model_path is None or job_name is None:
            print(f"Skip (no experiment_root/job_name): {cfg_path.name}")
            continue
        if not model_path.exists():
            print(f"Skip (file missing): {model_path}")
            continue
        path_in_repo = f"{job_name}.pth"
        if args.dry_run:
            print(f"Would upload: {model_path} -> {args.repo_id}/{path_in_repo}")
            uploaded += 1
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="model",
            )
            print(f"Uploaded: {path_in_repo}")
            uploaded += 1
        except Exception as e:
            print(f"Failed to upload {path_in_repo}: {e}", file=sys.stderr)
            failed += 1

    print(f"Done. Uploaded: {uploaded}, failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
