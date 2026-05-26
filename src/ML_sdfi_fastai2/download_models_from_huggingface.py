#!/usr/bin/env python
# coding: utf-8
"""
Download trained models from Hugging Face Hub.

Downloads .pth model files from a Hugging Face repository into a local directory.
Can download a single file by name or all .pth files in the repo.

Requires: pip install huggingface_hub
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    usage = (
        "Download models from Hugging Face to a local folder.\n"
        "Examples:\n"
        "  python download_models_from_huggingface.py --output_dir /mnt/T/mnt/logs_and_models/bygningsudpegning\n"
        "  python download_models_from_huggingface.py --output_dir ./models --model_file andringsudpegning_1km2benchmark_iter_73.pth\n"
    )
    parser = argparse.ArgumentParser(
        description="Download .pth model files from a Hugging Face repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=usage,
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="rasmuspjohansson/KDS_buildings",
        help="Hugging Face repo id (e.g. username/repo_name)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Local directory where model files will be saved",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="Optional: download only this filename (e.g. andringsudpegning_1km2benchmark_iter_73.pth). If omitted, download all .pth files in the repo.",
    )
    parser.add_argument(
        "--token_file",
        type=Path,
        default=None,
        help="Optional: path to file containing Hugging Face token (for private repos). Default: ../laz-superpoint_transformer/hftoken_write.txt relative to repo root.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    token = None
    if args.token_file is not None:
        token_path = args.token_file.resolve()
        if token_path.is_file():
            token = token_path.read_text().strip()
    else:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent
        default_token = repo_root.parent / "laz-superpoint_transformer" / "hftoken_write.txt"
        if default_token.is_file():
            token = default_token.read_text().strip()

    api = HfApi(token=token)

    if args.model_file:
        files_to_download = [args.model_file]
        if not args.model_file.endswith(".pth"):
            print("Warning: --model_file does not end with .pth", file=sys.stderr)
    else:
        try:
            repo_files = api.list_repo_files(args.repo_id, repo_type="model")
        except Exception as e:
            print(f"Failed to list repo files: {e}", file=sys.stderr)
            sys.exit(1)
        files_to_download = [f for f in repo_files if f.endswith(".pth")]
        if not files_to_download:
            print(f"No .pth files found in {args.repo_id}")
            return

    for filename in files_to_download:
        try:
            path = hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(out_dir),
                token=token,
            )
            print(f"Downloaded: {path}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Done. Saved to {out_dir}")


if __name__ == "__main__":
    main()
